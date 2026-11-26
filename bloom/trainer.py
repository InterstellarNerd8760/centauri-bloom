"""One canonical training engine for the Bloom Colab notebook."""
import json
import math
import random
import shutil
import time
from pathlib import Path

from bloom.notebook_workflow import parent_total, save_corpus, restore_corpus, sha256
from bloom.trainer_data import RECIPES, prepare

REPO_ID = "Centauri-Bloom/Bloom-Text-0.1"

def collate_instruction(batch):
    import torch
    x,y = (torch.stack([b[i] for b in batch]) for i in (0,1))
    n = int(torch.where(y != -100)[1].max())+1
    return x[:,:n].contiguous(),y[:,:n].contiguous()

def evaluate(model, loader, device, max_batches=32):
    import torch
    from torch.nn import functional as F
    from itertools import islice
    model.eval()
    total,count = 0.,0
    with torch.no_grad():
        for x,y in islice(loader,max_batches):
            x,y = x.to(device),y.to(device)
            with torch.amp.autocast(device_type=device,dtype=torch.float16,enabled=device=="cuda"):
                loss = F.cross_entropy(model(x).flatten(0,1),y.flatten(),reduction="sum")
            total += loss.item()
            count += int((y != -100).sum())
    if not count or not math.isfinite(total):
        raise RuntimeError("Empty or nonfinite validation loss.")
    return total/count,count

def generate_samples(model,tok,context,eot,device):
    import torch
    prompts = ["Hello","The Earth orbits","Once upon a time",
               "User: What is Earth?\nAssistant:",
               "User: What is 12 times 5?\nAssistant:",
               "User: Write a Python function that adds two numbers.\nAssistant:"]
    model.eval()
    results = {}
    with torch.no_grad():
        for prompt in prompts:
            ids = tok.encode(prompt).ids or [eot]
            new = []
            for _ in range(32):
                x = torch.tensor([ids[-context:]],device=device,dtype=torch.long)
                nxt = int(model(x)[0,-1].argmax())
                if nxt == eot:
                    break
                ids.append(nxt)
                new.append(nxt)
            results[prompt] = tok.decode(new)
    return results

def train_model(model, train_loader, val_loader, source_loaders, device, steps, grad_accum,
                lr, seed, checkpoint, samples, max_eval_batches=32):
    """Train a fresh optimizer; count actual input/target exposures and successful updates."""
    import torch
    from torch.nn import functional as F
    torch.manual_seed(seed)
    random.seed(seed)
    before,count = evaluate(model,val_loader,device,max_eval_batches)
    source_before = {k:evaluate(model,v,device,max_eval_batches)[0] for k,v in source_loaders.items()}
    before_samples = samples()
    print("Parent loss:",round(before,4),"| evaluated tokens:",count,flush=True)
    print(json.dumps(before_samples,indent=2),flush=True)
    opt = torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda",enabled=device=="cuda")
    it = iter(train_loader)
    inputs = targets = updates = 0
    warmup = min(50,max(1,steps//10))
    t0 = time.monotonic()
    for step in range(1,steps+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        progress = max(0,step-warmup)/max(1,steps-warmup)
        rate = lr*step/warmup if step<=warmup else lr*(0.1+0.9*0.5*(1+math.cos(math.pi*progress)))
        for pg in opt.param_groups:
            pg["lr"] = rate
        batches = []
        for _ in range(grad_accum):
            b = next(it,None)
            if b is None:
                it = iter(train_loader)
                b = next(it)
            batches.append(b)
        target_count = sum(int((y!=-100).sum()) for _,y in batches)
        if not target_count:
            raise RuntimeError("No supervised tokens in optimizer batch.")
        loss_sum = 0.
        for x,y in batches:
            # Last supervised position is the last real input; exclude padding.
            positions = torch.arange(y.shape[1]).expand_as(y)
            inputs += int((positions.masked_fill(y==-100,-1).max(dim=1).values+1).sum())
            targets += int((y!=-100).sum())
            x,y = x.to(device),y.to(device)
            with torch.amp.autocast(device_type=device,dtype=torch.float16,enabled=device=="cuda"):
                loss = F.cross_entropy(model(x).flatten(0,1),y.flatten(),reduction="sum")/target_count
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; saved checkpoints retained.")
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=device!="cuda")
        old_scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        updates += int(scaler.get_scale()>=old_scale)
        if step==1 or step%50==0 or step==steps:
            elapsed = time.monotonic()-t0
            print(f"step {step}/{steps} | loss {loss_sum:.4f} | input tokens {inputs:,} | {elapsed:.1f}s",flush=True)
        if step%500==0 or step==steps:
            checkpoint(step,inputs,targets,updates,loss_sum,time.monotonic()-t0)
    after,count_after = evaluate(model,val_loader,device,max_eval_batches)
    source_after = {k:evaluate(model,v,device,max_eval_batches)[0] for k,v in source_loaders.items()}
    report = {"parent_val_loss":before,"candidate_val_loss":after,"evaluated_target_tokens":count_after,
              "parent_source_val_losses":source_before,"candidate_source_val_losses":source_after,
              "parent_samples":before_samples,"candidate_samples":samples(),
              "note":"Same validation windows and greedy qualitative prompts; prior training overlap possible. Not a scored benchmark."}
    return report

def run_training(api, recipe, steps, repo_dir, out_dir, jaron_path=None, recover=False):
    import torch
    from torch.utils.data import DataLoader
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    from bloom.model import Bloom,Config
    from bloom.checkpoint import config_hash,load_weights,save_weights
    if recipe not in RECIPES or type(steps) is not int or not 1<=steps<=10000:
        raise ValueError("Choose a known dataset and 1–10,000 steps.")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime; this is the GPU trainer.")
    out = Path(out_dir)
    if (out/"metrics.json").exists() or (out/"partial.json").exists() or (out/"partial.safetensors").exists():
        raise RuntimeError("This run has checkpoints. Keep them; use upload recovery if completed, otherwise start a new run.")
    out.mkdir(parents=True,exist_ok=True)
    track,_ = RECIPES[recipe]
    instruction = track=="instruction-tuning"
    revision = api.model_info(REPO_ID).sha
    if not revision:
        raise RuntimeError("Canonical parent revision unavailable.")
    parent = Path(snapshot_download(REPO_ID,revision=revision,token=api.token,
                   allow_patterns=["config.json","tokenizer.json","metrics.json","model.safetensors"]))
    pm = json.loads((parent/"metrics.json").read_text())
    cfg = Config(**json.loads((parent/"config.json").read_text()))
    tok = Tokenizer.from_file(str(parent/"tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None or tok.get_vocab_size()!=cfg.vocab_size or config_hash(cfg)!=pm["config_sha256"]:
        raise ValueError("Canonical tokenizer/config verification failed.")
    seed = 47+int(pm["step"])
    torch.manual_seed(seed)
    model = Bloom(cfg)
    model.load_state_dict(load_weights(parent/"model.safetensors",pm["weights_sha256"],model.state_dict()),strict=True)
    model.to("cuda")
    params = sum(p.numel() for p in model.parameters())
    print("Parent revision:",revision,"| parent step:",pm["step"],"| parameters:",f"{params:,}",flush=True)
    batch_size,grad_accum,lr = 16,2,5e-5
    settings = {"recipe":recipe,"steps":steps,"parent_revision":revision,"seed":seed,"batch_size":batch_size,
                "grad_accum":grad_accum,"learning_rate":lr,"tokenizer_sha256":sha256(parent/"tokenizer.json")}
    settings_path = out/"run-settings.json"
    cache = out/"processed-corpus.npz"
    provenance = out/"data-provenance.json"
    if recover:
        if not settings_path.is_file() or json.loads(settings_path.read_text())!=settings:
            raise RuntimeError("Recovery settings/parent differ from saved run.")
        t,v,sources = restore_corpus(cache,revision,cfg.context,cfg.vocab_size)
        if not provenance.is_file():
            raise RuntimeError("Missing corpus provenance.")
    else:
        if settings_path.exists():
            raise RuntimeError("Run directory already initialized. Use recovery or a new folder.")
        settings_path.write_text(json.dumps(settings,indent=2))
        t,v,sources,meta = prepare(recipe,repo_dir,tok,cfg.context,eot,seed,jaron_path)
        provenance.write_text(json.dumps(meta,indent=2))
        save_corpus(cache,t,v,revision,{k:DataLoader(s) for k,s in sources.items()})
    shutil.copyfile(parent/"tokenizer.json",out/"tokenizer.json")
    (out/"config.json").write_text(json.dumps(cfg.__dict__,indent=2,sort_keys=True))
    collate = collate_instruction if instruction else None
    train = DataLoader(t,batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=collate,
                       generator=torch.Generator().manual_seed(seed))
    # Fixed random validation order avoids evaluating only the first corpus in a mixture.
    val = DataLoader(v,batch_size=batch_size,shuffle=False,collate_fn=collate,
                     sampler=torch.randperm(len(v),generator=torch.Generator().manual_seed(seed)).tolist())
    source_loaders = {k:DataLoader(s,batch_size=batch_size,shuffle=False) for k,s in sources.items()}
    if not len(train):
        raise RuntimeError("Too few training examples.")
    print("Dataset:",recipe,"| train examples/windows:",len(t),"| validation:",len(v),flush=True)
    latest = {}
    def checkpoint(step,inputs,targets,updates,loss,elapsed):
        partial = out/"partial.safetensors"
        tmp = out/"partial.tmp"
        digest = save_weights(tmp,model.state_dict(),model.state_dict())
        tmp.replace(partial)
        latest.update({"additional_steps":step,"train_tokens":inputs,"supervised_target_tokens":targets,
                       "optimizer_updates":updates,"last_train_loss":loss,"elapsed_seconds":elapsed,
                       "weights_sha256":digest})
        (out/"partial.json").write_text(json.dumps(latest,indent=2))
        print("Saved step",step,"to",out,flush=True)
    report = train_model(model,train,val,source_loaders,"cuda",steps,grad_accum,lr,seed,checkpoint,
                         lambda:generate_samples(model,tok,cfg.context,eot,"cuda"))
    report["parent_revision"] = revision
    (out/"evaluation.json").write_text(json.dumps(report,indent=2))
    previous = parent_total(pm)
    metrics = {**latest,"run_name":out.name,"step":int(pm["step"])+steps,
               "step_scope":"attempted optimizer steps; see optimizer_updates",
               "parent_revision":revision,"parent_weights_sha256":pm["weights_sha256"],
               "parent_reported_train_tokens":pm.get("train_tokens"),
               "parameters":params,"config_sha256":config_hash(cfg),
               "train_tokens_scope":"this continuation only; non-padding input tokens",
               "cumulative_train_tokens":previous+latest["train_tokens"] if previous is not None else None,
               "cumulative_train_tokens_scope":"reported prior plus this run's non-padding exposures; repeats included",
               "training_track":track,"dataset":recipe,"device":"cuda",
               "validation_loss_scope":"response targets only" if instruction else "next-token targets",
               "val_loss":report["candidate_val_loss"],"parent_val_loss_same_windows":report["parent_val_loss"],
               "source_val_losses":report["candidate_source_val_losses"],
               "train_examples":len(t),"validation_examples":len(v),"seed":seed,"learning_rate":lr,
               "batch_size":batch_size,"grad_accum":grad_accum,
               "optimizer_state_restored":False,"use_tinystories":"TinyStories" in recipe,
               "data_provenance_sha256":sha256(provenance),"evaluation_sha256":sha256(out/"evaluation.json")}
    if instruction:
        metrics["prompt_format"] = "User: {prompt}\\nAssistant:"
    # Final checkpoint is the last saved step; no duplication of the large file.
    (out/"partial.safetensors").replace(out/f"step-{steps:06d}.safetensors")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True,allow_nan=False))
    (out/"partial.json").unlink()
    print(json.dumps(report,indent=2),flush=True)
    print("COMPLETED:",out,"| step",metrics["step"],"| input tokens",metrics["train_tokens"],flush=True)
    return metrics
