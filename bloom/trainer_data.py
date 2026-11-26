"""Dataset recipes for the canonical Bloom trainer. Raw contributor data stays local."""
import hashlib
import json
import random
import subprocess
import sys
import unicodedata
from itertools import islice
from pathlib import Path

from bloom.notebook_workflow import sha256

JARON_SHA = "2b9f251e0c0ae172e867845e9accaed7a2d59af16b047eb24e0c4176f62b60da"
RECIPES = {
    "Jaron": ("instruction-tuning", ()),
    "FineWeb-Edu + TinyStories": ("fineweb-edu-replay", ("FineWeb-Edu", "TinyStories")),
    "FineWeb-Edu": ("continued-pretraining", ("FineWeb-Edu",)),
    "TinyStories": ("continued-pretraining", ("TinyStories",)),
    "Cosmopedia": ("knowledge-pretraining", ("Cosmopedia",)),
    "Public-domain literature": ("continued-pretraining", ("Public-domain literature",)),
    "Public-domain + TinyStories": ("continued-pretraining", ("Public-domain literature", "TinyStories")),
    "Arithmetic instruction pilot": ("instruction-tuning", ()),
}
SOURCES = {
    "FineWeb-Edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "ODC-By 1.0; underlying web rights vary", 7_500_000),
    "TinyStories": ("roneneldan/TinyStories", None, "CDLA-Sharing-1.0 (dataset card)", 2_500_000),
    "Cosmopedia": ("HuggingFaceTB/cosmopedia", "stanford", "Apache-2.0 (dataset card)", 3_000_000),
}

def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def instruction_pairs(rows):
    """Keep duplicate prompt variants in the same split; remove repeated pairs."""
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
        (val if int(digest(norm)[:8],16)%10 == 0 else train).append((q,a))
    return train, val

def arithmetic_rows():
    rows = []
    for a in range(1,51):
        for b in range(1,31):
            for q, r in [(f"What is {a} plus {b}?",a+b),(f"Subtract {b} from {a}.",a-b),(f"Multiply {a} by {b}.",a*b)]:
                rows.append({"prompt":q,"response":str(r)})
    for q,a in [
        ("What planet do humans live on?","Earth."),
        ("What is the nearest star to Earth?","The Sun."),
        ("What is the capital of France?","Paris."),
        ("What gas do plants use in photosynthesis?","Carbon dioxide."),
        ("What is a triangle?","A polygon with three sides."),
        ("Define evaporation.","Evaporation is the process by which a liquid becomes a gas at its surface."),
        ("Write the word hello in uppercase.","HELLO."),
        ("Name three colors.","Red, green and blue."),
        ("Say hello politely.","Hello! How can I help?"),
        ("What is an orbit?","An orbit is the path of an object around another object due to gravity."),
        ("What is a computer?","A machine that processes information according to instructions."),
        ("Can you be certain about every answer?","No. I can make mistakes and should acknowledge uncertainty."),
    ]:
        rows.append({"prompt":q,"response":a})
    return rows

def encode_pairs(pairs, tok, context, eot):
    import torch
    from torch.utils.data import TensorDataset
    xs, ys = [], []
    for q,a in pairs:
        prefix = tok.encode("User: "+q+"\nAssistant:").ids
        response = tok.encode(" "+a).ids + [eot]
        if not prefix or len(prefix)+len(response)>context+1:
            continue
        x = (prefix+response)[:-1]
        y = [-100]*(len(prefix)-1)+response
        pad = context-len(x)
        xs.append(x+[eot]*pad)
        ys.append(y+[-100]*pad)
    if not xs:
        raise ValueError("No instruction examples fit the parent context.")
    return TensorDataset(torch.tensor(xs,dtype=torch.long),torch.tensor(ys,dtype=torch.long))

def encode_windows(docs, context, eot):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    buf = []
    for ids in docs:
        buf.extend(ids)
        buf.append(eot)
    arr = np.asarray(buf,dtype=np.int64)
    n = (len(arr)-1)//context
    if n < 1:
        raise ValueError("Not enough corpus tokens for a full context window.")
    windows = np.lib.stride_tricks.sliding_window_view(arr[:n*context+1],context+1)[::context]
    return TensorDataset(torch.from_numpy(windows[:,:-1].copy()),torch.from_numpy(windows[:,1:].copy()))

def stream_source(name, tok, context, seed, seen, budget=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    repo, subset, license_note, default_budget = SOURCES[name]
    budget = budget or default_budget
    revision = HfApi().dataset_info(repo).sha
    if not revision:
        raise RuntimeError("Dataset revision unavailable: "+repo)
    print("Streaming:",repo,subset,revision,flush=True)
    data = load_dataset(repo,subset,split="train",revision=revision,streaming=True)
    data = data.shuffle(seed=seed,buffer_size=1024)
    cap = {"train":budget,"validation":max(context*64,budget//20)}
    docs = {"train":[],"validation":[]}
    used = {"train":0,"validation":0}
    records = []
    for row in islice(data,150_000):
        text = row.get("text")
        if not isinstance(text,str) or not text.strip():
            continue
        ids = tok.encode(text.strip()).ids[:4096]
        if len(ids)<16:
            continue
        key = digest(json.dumps(ids,separators=(",",":")))
        if key in seen:
            continue
        split = "validation" if int(key[:8],16)%20==0 else "train"
        if used[split]>=cap[split]:
            if all(used[k]>=cap[k] for k in cap):
                break
            continue
        seen.add(key)
        docs[split].append(ids)
        used[split]+=len(ids)+1
        records.append({"token_ids_sha256":key,"split":split,"source_id":str(row.get("id",""))})
        if len(records)%2000==0:
            print(name,used,flush=True)
    if not all(used[k]>=cap[k] for k in cap):
        raise RuntimeError(f"Incomplete bounded corpus for {name}: {used}. No training started.")
    meta = {"dataset":repo,"subset":subset,"revision":revision,"license_note":license_note,
            "dataset_card":"https://huggingface.co/datasets/"+repo,
            "tokens":used,"documents":{k:len(v) for k,v in docs.items()},"records":records}
    print(name,used,flush=True)
    return docs["train"],docs["validation"],meta

def public_domain(repo, tok):
    root = Path(repo)/"runs"/"public-domain-v1"
    sources = Path(repo)/"datasets"/"public-domain-v1"/"sources.json"
    if not (root/"manifest.json").is_file():
        subprocess.run([sys.executable,"-m","bloom.dataset","build",str(sources),str(root)],cwd=repo,check=True)
    subprocess.run([sys.executable,"-m","bloom.dataset","verify",str(root)],cwd=repo,check=True)
    def read(split):
        with (root/"data"/f"{split}-00000.jsonl").open(encoding="utf-8") as f:
            return [tok.encode(json.loads(line)["text"]).ids for line in f if line.strip()]
    train,val = read("train"),read("validation")
    meta = {"dataset":"public-domain-v1","license_note":"Public domain in the USA; review local rights",
            "source_manifest_sha256":sha256(sources),"built_manifest_sha256":sha256(root/"manifest.json"),
            "sources":json.loads(sources.read_text())}
    return train,val,meta

def prepare(recipe, repo, tokenizer, context, eot, seed, jaron_path=None):
    """Return train/validation datasets, per-source validation sets, and provenance."""
    import torch
    from torch.utils.data import TensorDataset
    track,names = RECIPES[recipe]
    if track=="instruction-tuning":
        if recipe=="Jaron":
            path = Path(jaron_path)
            if sha256(path)!=JARON_SHA:
                raise ValueError("Jaron JSON hash differs from the reviewed original.")
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows,list):
                raise ValueError("Expected a JSON array.")
            source = {"dataset":"Jaron instruction expanded","source_sha256":JARON_SHA,
                      "note":"Private contributor input; raw examples are not uploaded. Confirm permission and review accuracy."}
        else:
            rows = arithmetic_rows()
            source = {"dataset":"Original arithmetic and basic instruction pilot"}
        train,val = instruction_pairs(rows)
        random.Random(seed).shuffle(train)
        random.Random(seed).shuffle(val)
        t,v = encode_pairs(train,tokenizer,context,eot),encode_pairs(val,tokenizer,context,eot)
        source.update(source_count=len(rows),deduplicated_count=len(train)+len(val),
                      train_count=len(t),validation_count=len(v),split="normalized prompt hash; 10% held out")
        return t,v,{},{"recipe":recipe,"training_track":track,"sources":[source]}
    seen, train_sets, val_sets, metadata = set(), [], {}, []
    for name in names:
        if name=="Public-domain literature":
            tr,va,meta = public_domain(repo,tokenizer)
        else:
            tr,va,meta = stream_source(name,tokenizer,context,seed,seen)
        random.Random(seed).shuffle(tr)
        random.Random(seed).shuffle(va)
        train_sets.append(encode_windows(tr,context,eot))
        val_sets[name] = encode_windows(va,context,eot)
        metadata.append(meta)
    def combine(sets):
        return TensorDataset(*(torch.cat([s.tensors[i] for s in sets],dim=0) for i in (0,1)))
    return combine(train_sets),combine(list(val_sets.values())),val_sets,{"recipe":recipe,"training_track":track,"sources":metadata}
