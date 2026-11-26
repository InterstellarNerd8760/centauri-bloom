"""Bounded CPU-only instruction training from canonical Bloom weights.

Run from the repository root: python -m bloom.local_instruction --data PATH
No credentials, no uploads, no execution of text from the dataset.
"""
import argparse
import json
import math
import random
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from huggingface_hub import HfApi, snapshot_download
from tokenizers import Tokenizer

from bloom.model import Bloom, Config
from bloom.checkpoint import config_hash, load_weights, save_weights
from bloom.notebook_workflow import parent_total, preflight, sha256

REPO_ID = "Centauri-Bloom/Bloom-Text-0.1"
DATA_SHA = "2b9f251e0c0ae172e867845e9accaed7a2d59af16b047eb24e0c4176f62b60da"

def split_pairs(rows):
    import hashlib
    seen, train, val = set(), [], []
    for row in rows:
        if not isinstance(row, dict):
            continue
        q, a = row.get("prompt"), row.get("response")
        if not isinstance(q, str) or not isinstance(a, str) or not q.strip() or not a.strip():
            continue
        q, a = q.strip(), a.strip()
        norm = " ".join(unicodedata.normalize("NFKC", q).casefold().split()).rstrip(" .!?")
        key = (norm, a)
        if key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha256(norm.encode()).hexdigest()
        (val if int(digest[:8], 16) % 10 == 0 else train).append((q, a))
    return train, val

def encode_pairs(rows, tokenizer, context, eot):
    result = []
    for q, a in rows:
        p = tokenizer.encode("User: " + q + "\nAssistant:").ids
        r = tokenizer.encode(" " + a).ids + [eot]
        if p and len(p) + len(r) <= context + 1:
            result.append((p + r[:-1], [-100] * (len(p) - 1) + r))
    return result

def batch(rows, eot):
    length = max(len(x) for x, _ in rows)
    x = torch.tensor([x + [eot] * (length-len(x)) for x, _ in rows], dtype=torch.long)
    y = torch.tensor([y + [-100] * (length-len(y)) for _, y in rows], dtype=torch.long)
    return x, y

def evaluate(model, rows, eot, size):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for i in range(0, len(rows), size):
            x, y = batch(rows[i:i+size], eot)
            total += F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum").item()
            count += int((y != -100).sum())
    if not count or not math.isfinite(total):
        raise RuntimeError("Invalid validation loss")
    return total / count

def samples(model, tokenizer, context, eot):
    model.eval()
    answers = {}
    with torch.no_grad():
        for q in ("What is Earth?", "What is 12 times 5?", "Write a sentence about a robot."):
            ids = tokenizer.encode("User: " + q + "\nAssistant:").ids
            new = []
            for _ in range(16):
                x = torch.tensor([ids[-context:]], dtype=torch.long)
                nxt = int(model(x)[0, -1].argmax())
                if nxt == eot:
                    break
                ids.append(nxt)
                new.append(nxt)
            answers[q] = tokenizer.decode(new)
    return answers

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2, choices=(1,2))
    parser.add_argument("--batch-size", type=int, default=2, choices=(1,2))
    parser.add_argument("--grad-accum", type=int, default=4, choices=(1,2,4))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args()
    if not 1 <= args.steps <= 1500:
        parser.error("steps must be between 1 and 1500")
    data = args.data.expanduser()
    if sha256(data) != DATA_SHA:
        raise ValueError("Expected the original Jaren JSON; dataset hash differs.")
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < 1024**3:
        raise RuntimeError("At least 1 GiB of free space is required for checkpoints and cache.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(47)
    preflight("cpu")
    revision = HfApi().model_info(REPO_ID).sha
    if not revision:
        raise RuntimeError("Cannot resolve canonical parent.")
    parent = Path(snapshot_download(REPO_ID, revision=revision,
        allow_patterns=["config.json","tokenizer.json","metrics.json","model.safetensors"]))
    pm = json.loads((parent/"metrics.json").read_text())
    cfg = Config(**json.loads((parent/"config.json").read_text()))
    tok = Tokenizer.from_file(str(parent/"tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None or tok.get_vocab_size() != cfg.vocab_size or config_hash(cfg) != pm["config_sha256"]:
        raise ValueError("Parent config or tokenizer mismatch")
    model = Bloom(cfg)
    model.load_state_dict(load_weights(parent/"model.safetensors", pm["weights_sha256"], model.state_dict()))
    print("Parent:", revision, "| step:", pm.get("step"), "| parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    raw = json.loads(data.read_text())
    if not isinstance(raw, list):
        raise ValueError("Expected a JSON list of prompt/response examples.")
    train, val = split_pairs(raw)
    train, val = encode_pairs(train,tok,cfg.context,eot), encode_pairs(val,tok,cfg.context,eot)
    if not train or not val:
        raise ValueError("No usable train or validation pairs.")
    rng = random.Random(47)
    valid = rng.sample(val, min(32,len(val)))
    out = args.output/("bloom-jaren-cpu-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    out.mkdir(exist_ok=False)
    shutil.copyfile(parent/"tokenizer.json",out/"tokenizer.json")
    (out/"config.json").write_text(json.dumps(cfg.__dict__,indent=2))
    provenance = {"dataset":"Jaren instruction expanded","source_sha256":DATA_SHA,
        "source_count":len(raw),"train_count":len(train),"validation_count":len(val),
        "evaluated_validation_examples":len(valid),"split":"normalized prompt hash; 10% held out",
        "note":"Contributor data stays local. Use only with permission."}
    (out/"data-provenance.json").write_text(json.dumps(provenance,indent=2))
    before = evaluate(model, valid, eot, args.batch_size)
    before_samples = samples(model,tok,cfg.context,eot)
    print("Initial held-out loss:",round(before,4),"| output:",out,flush=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(0.9,0.95),weight_decay=0.1)
    inp = targets = 0
    t0 = time.monotonic()
    for step in range(1,args.steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for _ in range(args.grad_accum):
            rows = rng.choices(train,k=args.batch_size)
            x,y = batch(rows,eot)
            loss = F.cross_entropy(model(x).flatten(0,1),y.flatten())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            (loss/args.grad_accum).backward()
            loss_sum += loss.item()/args.grad_accum
            inp += sum(len(x) for x,_ in rows)
            targets += int((y!=-100).sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        print(f"step {step}/{args.steps} | loss {loss_sum:.4f} | input tokens {inp:,} | {time.monotonic()-t0:.1f}s",flush=True)
        if step % 10 == 0 or step == args.steps:
            # Keep a single intermediate checkpoint to limit disk usage.
            path=out/"partial.safetensors"
            save_weights(path,model.state_dict(),model.state_dict())
            (out/"partial.json").write_text(json.dumps({"additional_steps":step,"train_tokens":inp,"parent_revision":revision},indent=2))
    after=evaluate(model,valid,eot,args.batch_size)
    report={"parent_revision":revision,"parent_instruction_loss":before,"candidate_instruction_loss":after,
        "parent_samples":before_samples,"candidate_samples":samples(model,tok,cfg.context,eot),
        "note":"32 or fewer held-out contributor examples; possible previous exposure; greedy 16-token qualitative samples."}
    (out/"evaluation.json").write_text(json.dumps(report,indent=2))
    digest=save_weights(out/f"step-{args.steps:06d}.safetensors",model.state_dict(),model.state_dict())
    previous=parent_total(pm)
    metrics={"run_name":out.name,"step":int(pm["step"])+args.steps,"additional_steps":args.steps,
        "optimizer_updates":args.steps,"parent_revision":revision,"parent_weights_sha256":pm["weights_sha256"],
        "parameters":sum(p.numel() for p in model.parameters()),"config_sha256":config_hash(cfg),
        "weights_sha256":digest,"train_tokens":inp,"supervised_target_tokens":targets,
        "train_tokens_scope":"this continuation only; non-padding input tokens",
        "cumulative_train_tokens":previous+inp if previous is not None else None,
        "cumulative_train_tokens_scope":"reported prior exposures plus non-padding input tokens; repeated exposures included",
        "training_track":"instruction-tuning","dataset":"Jaren instruction expanded","device":"cpu",
        "prompt_format":"User: {prompt}\\nAssistant:","val_loss":after,"parent_val_loss_same_windows":before,
        "validation_loss_scope":"response targets only; small fixed subset","last_train_loss":loss_sum,
        "train_examples":len(train),"validation_examples":len(val),"seed":47,
        "batch_size":args.batch_size,"grad_accum":args.grad_accum,"learning_rate":5e-5,
        "elapsed_seconds":round(time.monotonic()-t0,2),
        "data_provenance_sha256":sha256(out/"data-provenance.json"),"evaluation_sha256":sha256(out/"evaluation.json")}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    (out/"partial.safetensors").unlink(missing_ok=True)
    (out/"partial.json").unlink(missing_ok=True)
    print(json.dumps(report,indent=2))
    print("Completed local candidate:",out,"\nNo upload performed.")
if __name__ == "__main__":
    main()
