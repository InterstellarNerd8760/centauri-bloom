"""One canonical training engine for the Bloom Colab notebook."""
import json
import math
import random
import shutil
import time
from pathlib import Path

from bloom.notebook_workflow import parent_total, pick_device, save_corpus, restore_corpus, sha256
from bloom.trainer_data import RECIPES, INSTRUCT_MIX, prepare

REPO_ID = "Centauri-Bloom/Bloom-Text-0.1"
TEXT_REPOS = {"14M": REPO_ID, "50M": "Centauri-Bloom/Bloom-Text-0.2"}
CHAT_REPOS = {"14M": "Centauri-Bloom/Bloom-Chat-0.1", "50M": "Centauri-Bloom/Bloom-Chat-0.2"}
TARGET_REPOS = TEXT_REPOS
CONFIG_50M = "configs/bloom-0.2-50m.json"
PARAM_COUNT_50M = 54_807_040

def target_repo(model_size, recipe):
    """Pretraining stays on Bloom-Text-*; 50M instruction and the open instruct mix use Bloom-Chat-*."""
    if model_size not in TEXT_REPOS or recipe not in RECIPES:
        raise ValueError("Choose 14M or 50M and a known dataset.")
    track, _ = RECIPES[recipe]
    if track == "instruction-tuning" and (model_size == "50M" or recipe == INSTRUCT_MIX):
        return CHAT_REPOS[model_size]
    return TEXT_REPOS[model_size]

def uses_chat_repo(model_size, recipe):
    return target_repo(model_size, recipe) in CHAT_REPOS.values()

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

def load_model_config(path):
    raw = json.loads(Path(path).read_text())
    return {k: raw[k] for k in ("vocab_size", "context", "width", "heads", "layers")}

def _move_optimizer(opt, device):
    import torch
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def train_model(model, train_loader, val_loader, source_loaders, device, steps, grad_accum,
                lr, seed, checkpoint, samples, max_eval_batches=32, start_step=0,
                optimizer_file=None, token_counts=None):
    """Train; optionally resume weights/optimizer already loaded by the caller."""
    import torch
    from torch.nn import functional as F
    torch.manual_seed(seed)
    random.seed(seed)
    before,count = evaluate(model,val_loader,device,max_eval_batches)
    source_before = {k:evaluate(model,v,device,max_eval_batches)[0] for k,v in source_loaders.items()}
    before_samples = samples()
    print("Eval loss:",round(before,4),"| evaluated tokens:",count,flush=True)
    print(json.dumps(before_samples,indent=2),flush=True)
    opt = torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda",enabled=device=="cuda")
    if start_step:
        if optimizer_file and Path(optimizer_file).is_file():
            state = torch.load(optimizer_file, map_location="cpu", weights_only=False)
            if int(state.get("step", -1)) != int(start_step):
                raise RuntimeError("Optimizer checkpoint step mismatch.")
            opt.load_state_dict(state["opt"])
            _move_optimizer(opt, device)
            if device == "cuda" and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            print("Restored optimizer at step", start_step, flush=True)
        else:
            print("Resuming weights without optimizer state at step", start_step, flush=True)
    it = iter(train_loader)
    inputs = targets = updates = 0
    if token_counts:
        inputs, targets, updates = (int(v) for v in token_counts)
    warmup = min(50,max(1,steps//10))
    t0 = time.monotonic()
    for step in range(start_step+1,steps+1):
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
            if optimizer_file:
                tmp = Path(optimizer_file).with_suffix(".tmp")
                torch.save({"opt":opt.state_dict(),"scaler":scaler.state_dict(),"step":step}, tmp)
                tmp.replace(Path(optimizer_file))
    after,count_after = evaluate(model,val_loader,device,max_eval_batches)
    source_after = {k:evaluate(model,v,device,max_eval_batches)[0] for k,v in source_loaders.items()}
    report = {"parent_val_loss":before,"candidate_val_loss":after,"evaluated_target_tokens":count_after,
              "parent_source_val_losses":source_before,"candidate_source_val_losses":source_after,
              "parent_samples":before_samples,"candidate_samples":samples(),
              "note":"Same validation windows and greedy qualitative prompts; prior training overlap possible. Not a scored benchmark."}
    return report


def load_local_parent(path, model_size, repo_dir):
    """Load a completed trainer run folder as the instruction-tuning parent."""
    from bloom.model import Config
    from bloom.checkpoint import config_hash
    path = Path(path)
    metrics_path, config_path = path/"metrics.json", path/"config.json"
    if not metrics_path.is_file() or not config_path.is_file() or not (path/"tokenizer.json").is_file():
        raise RuntimeError("PARENT_RUN_DIR must contain metrics.json, config.json, and tokenizer.json.")
    pm = json.loads(metrics_path.read_text())
    cfg = Config(**json.loads(config_path.read_text()))
    steps = pm.get("additional_steps", pm.get("step"))
    weights = path/f"step-{int(steps):06d}.safetensors"
    if not weights.is_file():
        weights = path/"model.safetensors"
    if not weights.is_file():
        raise RuntimeError("PARENT_RUN_DIR has no completed SafeTensors weights.")
    digest = pm.get("weights_sha256")
    if not digest or sha256(weights)!=digest:
        raise ValueError("PARENT_RUN_DIR weights hash mismatch.")
    if config_hash(cfg)!=pm.get("config_sha256"):
        raise ValueError("PARENT_RUN_DIR config hash mismatch.")
    if model_size=="50M":
        expected = load_model_config(Path(repo_dir)/CONFIG_50M)
        shape = (cfg.vocab_size,cfg.context,cfg.width,cfg.heads,cfg.layers)
        want = tuple(expected[k] for k in ("vocab_size","context","width","heads","layers"))
        if shape!=want:
            raise ValueError("PARENT_RUN_DIR is not the 50M Bloom-0.2 config.")
        if pm.get("parameters") not in (None, PARAM_COUNT_50M):
            raise ValueError("PARENT_RUN_DIR is not a 54.8M checkpoint.")
    elif (cfg.width, cfg.layers)==(640, 10):
        raise ValueError("PARENT_RUN_DIR looks like the 50M model; set MODEL to 50M.")
    revision = "local:"+path.name+":"+digest
    return path, cfg, pm, weights, revision

def load_hub_parent(api, repo, token=None, revision=None):
    """Load weights from a Hugging Face model repo, optionally a PR or other revision."""
    from huggingface_hub import snapshot_download
    from bloom.model import Config
    try:
        info = api.model_info(repo, revision=revision) if revision else api.model_info(repo)
        resolved = info.sha
    except Exception as exc:
        if revision:
            raise RuntimeError(
                "Could not load %s at revision %s. Check PARENT_HF_REPO and PARENT_HF_REVISION, or set PARENT_RUN_DIR."
                % (repo, revision)
            ) from exc
        raise RuntimeError(
            "Could not load %s. Set PARENT_HF_REVISION (for example refs/pr/1) or PARENT_RUN_DIR, or wait until that repo has merged weights."
            % repo
        ) from exc
    if not resolved:
        raise RuntimeError("Parent revision unavailable: "+repo)
    pin = revision or resolved
    parent = Path(snapshot_download(repo,revision=pin,token=token,
                   allow_patterns=["config.json","tokenizer.json","metrics.json","model.safetensors"]))
    if not (parent/"model.safetensors").is_file() or not (parent/"metrics.json").is_file() or not (parent/"config.json").is_file():
        raise RuntimeError(
            "%s revision %s does not have complete weights. Set PARENT_HF_REVISION to a candidate PR such as refs/pr/1, or PARENT_RUN_DIR to the completed run folder."
            % (repo, pin)
        )
    pm = json.loads((parent/"metrics.json").read_text())
    cfg = Config(**json.loads((parent/"config.json").read_text()))
    return parent, cfg, pm, parent/"model.safetensors", resolved

def run_training(api, recipe, steps, repo_dir, out_dir, jaron_path=None, recover=False,
                 model_size="14M", device=None, parent_run_dir=None,
                 parent_hf_repo=None, parent_hf_revision=None):
    import torch
    from torch.utils.data import DataLoader
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    from bloom.model import Bloom,Config
    from bloom.checkpoint import config_hash,load_weights,save_weights
    if model_size not in TEXT_REPOS or recipe not in RECIPES or type(steps) is not int or not 1<=steps<=10000:
        raise ValueError("Choose 14M or 50M, a known dataset, and 1–10,000 steps.")
    if parent_run_dir and (parent_hf_repo or parent_hf_revision):
        raise ValueError("Set PARENT_RUN_DIR or PARENT_HF_REPO/PARENT_HF_REVISION, not both.")
    if (parent_run_dir or parent_hf_repo or parent_hf_revision) and model_size=="50M" and RECIPES[recipe][0]!="instruction-tuning":
        raise ValueError("PARENT_RUN_DIR and PARENT_HF_* are for instruction fine-tuning, not from-scratch pretraining.")
    device = device or pick_device()
    if device not in ("cuda","mps","cpu"):
        raise ValueError("Unsupported device.")
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device=="mps":
        mps = getattr(torch.backends,"mps",None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS is not available.")
    out = Path(out_dir)
    completed = (out/"metrics.json").exists()
    has_partial = (out/"partial.json").exists() or (out/"partial.safetensors").exists()
    if completed:
        if recover:
            from bloom.notebook_workflow import repair_completed_metrics
            metrics = repair_completed_metrics(out, steps=steps)
            if metrics.get("dataset") not in (None, recipe):
                raise RuntimeError("Completed run dataset differs from the current notebook settings.")
            if metrics.get("model_size", "14M") != model_size:
                raise RuntimeError("Completed run model size differs from the current notebook settings.")
            if metrics.get("additional_steps") != steps:
                raise RuntimeError("Completed run step count differs from STEPS after repair.")
            print("Completed run found; skipping training:", out, flush=True)
            return metrics
        raise RuntimeError("This run already completed. Use upload recovery if you need to submit the PR.")
    if has_partial and not recover:
        raise RuntimeError("This run has checkpoints. Set RECOVER_RUN_DIR to this folder to resume, or start a new run.")
    out.mkdir(parents=True,exist_ok=True)
    track,_ = RECIPES[recipe]
    instruction = track=="instruction-tuning"
    from_scratch = model_size=="50M" and not instruction
    target_repo_id = target_repo(model_size, recipe)
    # Always read the canonical 14M tokenizer; never write back to Bloom-Text-0.1.
    tokenizer_revision = api.model_info(REPO_ID).sha
    if not tokenizer_revision:
        raise RuntimeError("Canonical tokenizer revision unavailable.")
    tokenizer_root = Path(snapshot_download(REPO_ID,revision=tokenizer_revision,token=api.token,
                   allow_patterns=["config.json","tokenizer.json","metrics.json","model.safetensors"]))
    tok = Tokenizer.from_file(str(tokenizer_root/"tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    parent = tokenizer_root
    target_revision = None
    if from_scratch:
        cfg = Config(**load_model_config(Path(repo_dir)/CONFIG_50M))
        if eot is None or tok.get_vocab_size()!=cfg.vocab_size:
            raise ValueError("Tokenizer/config verification failed.")
        seed = 47
        torch.manual_seed(seed)
        model = Bloom(cfg)
        with torch.no_grad():
            torch.nn.init.normal_(model.tokens.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(model.positions.weight, mean=0.0, std=0.02)
        try:
            revision = api.model_info(target_repo_id).sha
        except Exception:
            revision = None
        target_revision = revision
        pm = {"step":0,"weights_sha256":None,"train_tokens":0}
        batch_size,grad_accum,lr = 8,4,3e-4
        params_expected = PARAM_COUNT_50M
    else:
        if parent_run_dir:
            parent, cfg, pm, weight_path, revision = load_local_parent(parent_run_dir, model_size, repo_dir)
            tok = Tokenizer.from_file(str(parent/"tokenizer.json"))
            eot = tok.token_to_id("<|endoftext|>")
        elif parent_hf_repo or parent_hf_revision:
            hub_repo = parent_hf_repo or TEXT_REPOS[model_size]
            if model_size=="50M" and hub_repo in {TEXT_REPOS["14M"], CHAT_REPOS["14M"]}:
                raise ValueError("50M instruction tuning cannot start from Bloom 0.1.")
            parent, cfg, pm, weight_path, revision = load_hub_parent(
                api, hub_repo, token=api.token, revision=parent_hf_revision or None)
            if (parent/"tokenizer.json").is_file():
                tok = Tokenizer.from_file(str(parent/"tokenizer.json"))
                eot = tok.token_to_id("<|endoftext|>")
        elif model_size=="50M":
            parent, cfg, pm, weight_path, revision = load_hub_parent(api, TEXT_REPOS["50M"], token=api.token)
        else:
            revision = tokenizer_revision
            pm = json.loads((parent/"metrics.json").read_text())
            cfg = Config(**json.loads((parent/"config.json").read_text()))
            weight_path = parent/"model.safetensors"
        if eot is None or tok.get_vocab_size()!=cfg.vocab_size or config_hash(cfg)!=pm["config_sha256"]:
            raise ValueError("Parent tokenizer/config verification failed.")
        seed = 47+int(pm["step"])
        torch.manual_seed(seed)
        model = Bloom(cfg)
        model.load_state_dict(load_weights(weight_path,pm["weights_sha256"],model.state_dict()),strict=True)
        if model_size=="50M":
            batch_size,grad_accum,lr = 8,4,5e-5
            params_expected = PARAM_COUNT_50M
        else:
            batch_size,grad_accum,lr = 16,2,5e-5
            params_expected = None
        try:
            target_revision = api.model_info(target_repo_id).sha
        except Exception:
            target_revision = None
        if target_repo_id==TEXT_REPOS[model_size] and not parent_run_dir:
            target_revision = revision
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    if params_expected is not None and params!=params_expected:
        raise RuntimeError(f"50M config produced {params} parameters, expected {params_expected}.")
    print("Model:",model_size,"| target:",target_repo_id,"| device:",device,
          "| revision:",revision,"| parent step:",pm["step"],"| parameters:",f"{params:,}",flush=True)
    if from_scratch:
        settings = {"recipe":recipe,"steps":steps,"model_size":model_size,
                    "tokenizer_revision":tokenizer_revision,"target_revision":revision,
                    "seed":seed,"batch_size":batch_size,"grad_accum":grad_accum,
                    "learning_rate":lr,"tokenizer_sha256":sha256(parent/"tokenizer.json")}
        cache_revision = tokenizer_revision
    elif uses_chat_repo(model_size, recipe) or parent_run_dir:
        settings = {"recipe":recipe,"steps":steps,"model_size":model_size,
                    "parent_revision":revision,"target_repo":target_repo_id,
                    "target_revision":target_revision,"parent_run_dir":str(parent_run_dir or ""),
                    "parent_hf_repo":parent_hf_repo or "","parent_hf_revision":parent_hf_revision or "",
                    "seed":seed,"batch_size":batch_size,"grad_accum":grad_accum,
                    "learning_rate":lr,"tokenizer_sha256":sha256(parent/"tokenizer.json")}
        cache_revision = revision
    else:
        settings = {"recipe":recipe,"steps":steps,"parent_revision":revision,"seed":seed,"batch_size":batch_size,
                    "grad_accum":grad_accum,"learning_rate":lr,"tokenizer_sha256":sha256(parent/"tokenizer.json")}
        cache_revision = revision
    settings_path = out/"run-settings.json"
    cache = out/"processed-corpus.npz"
    provenance = out/"data-provenance.json"
    start_step = 0
    token_counts = None
    restored_optimizer = False
    if recover:
        if not settings_path.is_file() or json.loads(settings_path.read_text())!=settings:
            raise RuntimeError("Recovery settings/parent differ from saved run.")
        t,v,sources = restore_corpus(cache,cache_revision,cfg.context,cfg.vocab_size)
        if not provenance.is_file():
            raise RuntimeError("Missing corpus provenance.")
        if has_partial:
            if not (out/"partial.json").is_file() or not (out/"partial.safetensors").is_file():
                raise RuntimeError("Incomplete partial checkpoint; keep the files and inspect them before retrying.")
            latest_partial = json.loads((out/"partial.json").read_text())
            start_step = int(latest_partial["additional_steps"])
            if start_step>=steps:
                raise RuntimeError("This run already reached the requested step count.")
            model.load_state_dict(load_weights(out/"partial.safetensors",latest_partial["weights_sha256"],model.state_dict()),strict=True)
            token_counts = (latest_partial["train_tokens"],latest_partial["supervised_target_tokens"],
                            latest_partial["optimizer_updates"])
            restored_optimizer = (out/"partial-optimizer.pt").is_file()
            print("Resuming from step",start_step,"| optimizer state:",restored_optimizer,flush=True)
    else:
        if settings_path.exists():
            raise RuntimeError("Run directory already initialized. Use recovery or a new folder.")
        settings_path.write_text(json.dumps(settings,indent=2))
        t,v,sources,meta = prepare(recipe,repo_dir,tok,cfg.context,eot,seed,jaron_path)
        provenance.write_text(json.dumps(meta,indent=2))
        save_corpus(cache,t,v,cache_revision,{k:DataLoader(s) for k,s in sources.items()})
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
    report = train_model(model,train,val,source_loaders,device,steps,grad_accum,lr,seed,checkpoint,
                         lambda:generate_samples(model,tok,cfg.context,eot,device),
                         start_step=start_step,optimizer_file=str(out/"partial-optimizer.pt"),
                         token_counts=token_counts)
    report["parent_revision"] = revision
    (out/"evaluation.json").write_text(json.dumps(report,indent=2))
    previous = parent_total(pm)
    if from_scratch:
        cumulative = latest["train_tokens"]
        step_total = steps
        train_scope = "this from-scratch run only; non-padding input tokens"
        cumulative_scope = "non-padding exposures in this 50M run; repeats included"
    else:
        cumulative = previous+latest["train_tokens"] if previous is not None else None
        step_total = int(pm["step"])+steps
        train_scope = "this continuation only; non-padding input tokens"
        cumulative_scope = "reported prior plus this run's non-padding exposures; repeats included"
    metrics = {**latest,"run_name":out.name,"step":step_total,
               "step_scope":"attempted optimizer steps; see optimizer_updates",
               "parent_revision":revision,"parent_weights_sha256":pm["weights_sha256"],
               "parent_reported_train_tokens":pm.get("train_tokens"),
               "parameters":params,"config_sha256":config_hash(cfg),
               "train_tokens_scope":train_scope,
               "cumulative_train_tokens":cumulative,
               "cumulative_train_tokens_scope":cumulative_scope,
               "training_track":track,"dataset":recipe,"device":device,
               "model_size":model_size,"target_repo":target_repo_id,"target_revision":target_revision,
               "validation_loss_scope":"response targets only" if instruction else "next-token targets",
               "val_loss":report["candidate_val_loss"],"parent_val_loss_same_windows":report["parent_val_loss"],
               "source_val_losses":report["candidate_source_val_losses"],
               "train_examples":len(t),"validation_examples":len(v),"seed":seed,"learning_rate":lr,
               "batch_size":batch_size,"grad_accum":grad_accum,
               "optimizer_state_restored":restored_optimizer,"use_tinystories":"TinyStories" in recipe,
               "data_provenance_sha256":sha256(provenance),"evaluation_sha256":sha256(out/"evaluation.json")}
    if from_scratch:
        metrics["init"] = "random"
        metrics["tokenizer_source"] = REPO_ID
        metrics["tokenizer_revision"] = tokenizer_revision
    if instruction:
        metrics["prompt_format"] = "User: {prompt}\\nAssistant:"
    # Final checkpoint is the last saved step; no duplication of the large file.
    (out/"partial.safetensors").replace(out/f"step-{steps:06d}.safetensors")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True,allow_nan=False))
    (out/"partial.json").unlink()
    print(json.dumps(report,indent=2),flush=True)
    print("COMPLETED:",out,"| step",metrics["step"],"| input tokens",metrics["train_tokens"],flush=True)
    return metrics
