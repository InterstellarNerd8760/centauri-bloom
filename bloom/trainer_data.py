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
# Reviewed contributor files only. A later file can be added here after a quality check;
# load_jaron_sources combines allowlisted files and instruction_pairs de-duplicates them.
KNOWN_JARON_SHAS = {
    JARON_SHA: "bloom_instruction_expanded",
}
INSTRUCT_MIX = "Instruct mix (Jaron + open)"
RECIPES = {
    "Jaron": ("instruction-tuning", ()),
    "FineWeb-Edu + TinyStories": ("fineweb-edu-replay", ("FineWeb-Edu", "TinyStories")),
    "FineWeb-Edu": ("continued-pretraining", ("FineWeb-Edu",)),
    "TinyStories": ("continued-pretraining", ("TinyStories",)),
    "Cosmopedia": ("knowledge-pretraining", ("Cosmopedia",)),
    "Public-domain literature": ("continued-pretraining", ("Public-domain literature",)),
    "Public-domain + TinyStories": ("continued-pretraining", ("Public-domain literature", "TinyStories")),
    "FineWeb-Edu + literature + TinyStories": ("pretraining", ("FineWeb-Edu", "Public-domain literature", "TinyStories")),
    "Arithmetic instruction pilot": ("instruction-tuning", ()),
    INSTRUCT_MIX: ("instruction-tuning", ()),
}
# Open instruction sources mixed with Jaron. Caps are unique examples after
# conversion, license/PII filters, and the 512-token chat-template fit check.
# Weights are relative sampling shares; Jaron is upweighted, then repeated up
# to INSTRUCT_MIX_MAX_REPEAT so a 3,000–5,000 step 50M run stays near 1–2 hours on a T4.
INSTRUCT_SOURCES = {
    "Dolly-15k": {
        "repo": "databricks/databricks-dolly-15k",
        "subset": None,
        "split": "train",
        "license_note": "CC BY-SA 3.0 (dataset card); attribute Databricks and Wikipedia; share-alike may apply to derived artifacts",
        "cap": 6000,
        "weight": 2,
    },
    "OpenAssistant OASST1": {
        "repo": "OpenAssistant/oasst1",
        "subset": None,
        "split": "train",
        "license_note": "Apache-2.0 (dataset card); English first-turn human replies; rows labelled PII or disallowed content are dropped",
        "cap": 5000,
        "weight": 2,
    },
    "GSM8K": {
        "repo": "openai/gsm8k",
        "subset": "main",
        "split": "train",
        "license_note": "MIT (dataset card); official test split is not used",
        "cap": 4000,
        "weight": 1,
    },
}
JARON_MIX_WEIGHT = 4
JARON_MIX_CAP = 8000
INSTRUCT_MIX_TRAIN_CAP = 24000
INSTRUCT_MIX_MAX_REPEAT = 8

def needs_jaron(recipe, stage="sft", rft_train=False):
    """Jaron is not required for a yield-only RFT pilot."""
    if stage == "rft" and not rft_train:
        return False
    if stage in ("rft", "rft-baseline"):
        return recipe in ("Jaron", INSTRUCT_MIX)
    return recipe in ("Jaron", INSTRUCT_MIX)

def instruct_mix_weights():
    return {"Jaron": JARON_MIX_WEIGHT, **{name: spec["weight"] for name, spec in INSTRUCT_SOURCES.items()}}
# Optional per-source train token budgets. Unlisted sources use SOURCES defaults
# or, for public-domain literature, the full verified corpus.
RECIPE_BUDGETS = {
    "FineWeb-Edu + literature + TinyStories": {
        "FineWeb-Edu": 20_000_000,
        "TinyStories": 500_000,
    },
}
# Larger unique-token budgets for continued 50M pretraining. The from-scratch
# recipe above is unchanged. FineWeb-Edu sample-10BT is about 10B tokens.
PRETRAIN_CONTINUE_BUDGETS = {
    "FineWeb-Edu + literature + TinyStories": {
        "FineWeb-Edu": 100_000_000,
        "TinyStories": 2_500_000,
    },
    "FineWeb-Edu + TinyStories": {
        "FineWeb-Edu": 100_000_000,
        "TinyStories": 2_500_000,
    },
    "FineWeb-Edu": {"FineWeb-Edu": 100_000_000},
}
# stream_source stops after this many shuffled rows. 100M FineWeb tokens at the
# measured ~1,216 tokens/document from Text-0.2 needs ~86k documents; keep
# headroom if a later shuffle draws shorter pages.
STREAM_ROW_LIMIT = 400_000
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
    for row in islice(data, STREAM_ROW_LIMIT):
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

def fits_instruction(q, a, tok, context, eot):
    if not isinstance(q, str) or not isinstance(a, str) or not q.strip() or not a.strip():
        return False
    prefix = tok.encode("User: " + q.strip() + "\nAssistant:").ids
    response = tok.encode(" " + a.strip()).ids + [eot]
    return bool(prefix) and len(prefix) + len(response) <= context + 1

def filter_pairs(rows, tok, context, eot):
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        q, a = row.get("prompt"), row.get("response")
        if fits_instruction(q, a, tok, context, eot):
            out.append({"prompt": q.strip(), "response": a.strip()})
    return out

def jaron_paths(jaron_path):
    """Accept a path, comma-separated paths, or a list of paths."""
    if not jaron_path:
        return []
    if isinstance(jaron_path, (list, tuple)):
        items = [str(p) for p in jaron_path]
    else:
        text = str(jaron_path)
        items = [part.strip() for part in text.split(",")] if "," in text else [text]
    paths = [Path(item).expanduser() for item in items if item and str(item).strip() not in ("None",)]
    if not paths:
        raise ValueError("No Jaron JSON paths were provided.")
    return paths

def load_jaron_rows(path):
    path = Path(path)
    digest = sha256(path)
    if digest not in KNOWN_JARON_SHAS:
        raise ValueError("Jaron JSON hash is not a reviewed contributor file: " + path.name)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON array.")
    return rows

def load_jaron_sources(jaron_path):
    """Load one or more reviewed Jaron JSON files and record per-file hashes."""
    files, rows, sources, seen = jaron_paths(jaron_path), [], [], set()
    for path in files:
        digest = sha256(path)
        if digest in seen:
            continue
        seen.add(digest)
        part = load_jaron_rows(path)
        rows.extend(part)
        sources.append({"path": path.name, "source_sha256": digest,
                        "label": KNOWN_JARON_SHAS[digest], "source_count": len(part)})
    if not rows:
        raise ValueError("Jaron JSON files were empty.")
    return rows, sources

def download_jaron_repo(repo, token, revision="main"):
    """Fetch reviewed Jaron JSON files from a private Hugging Face dataset repo."""
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(repo_id=repo, repo_type="dataset", revision=revision or "main",
                                  token=token, allow_patterns=["*.json"]))
    matches, seen = [], set()
    for path in sorted(root.rglob("*.json")):
        digest = sha256(path)
        if digest in KNOWN_JARON_SHAS and digest not in seen:
            seen.add(digest)
            matches.append(path)
    if not matches:
        raise ValueError(
            "No reviewed Jaron JSON was found in %s. Upload bloom_instruction_expanded.json to that private dataset, or set JARON_FILE."
            % repo
        )
    return matches

def dolly_pair(row):
    inst, resp = row.get("instruction"), row.get("response")
    ctx = row.get("context") or ""
    if not isinstance(inst, str) or not isinstance(resp, str):
        return None
    inst, resp = inst.strip(), resp.strip()
    if not inst or not resp:
        return None
    prompt = (ctx.strip() + "\n\n" + inst) if isinstance(ctx, str) and ctx.strip() else inst
    return {"prompt": prompt, "response": resp}

def gsm8k_pair(row):
    q, a = row.get("question"), row.get("answer")
    if not isinstance(q, str) or not isinstance(a, str):
        return None
    q, a = q.strip(), a.strip()
    if not q or not a:
        return None
    return {"prompt": q, "response": a}

def _label_names_values(labels):
    if not labels:
        return [], []
    if isinstance(labels, dict) and "name" in labels:
        return list(labels.get("name") or []), list(labels.get("value") or [])
    if isinstance(labels, dict):
        names, values = [], []
        for name, item in labels.items():
            names.append(name)
            if isinstance(item, dict):
                values.append(item.get("value", 0))
            else:
                values.append(item)
        return names, values
    return [], []

def _has_label(row, name):
    names, _ = _label_names_values(row.get("labels"))
    return name in names

def _label_value(row, name):
    names, values = _label_names_values(row.get("labels"))
    if name not in names:
        return 0.0
    try:
        return float(values[names.index(name)])
    except (TypeError, ValueError, IndexError):
        return 0.0

def oasst_usable(row):
    if row.get("deleted") or row.get("review_result") is False:
        return False
    if row.get("lang") != "en":
        return False
    if _label_value(row, "pii") > 0:
        return False
    for flag in ("spam", "not_appropriate", "hate_speech", "sexual_content"):
        if _label_value(row, flag) >= 0.5:
            return False
    return True

def oasst_pairs(rows):
    messages = {}
    for row in rows:
        mid = row.get("message_id")
        if mid:
            messages[mid] = row
    pairs = []
    for row in rows:
        if row.get("role") != "assistant":
            continue
        if row.get("rank") not in (0, None):
            continue
        parent = messages.get(row.get("parent_id"))
        if not parent or parent.get("role") != "prompter" or parent.get("parent_id"):
            continue
        if not oasst_usable(row) or not oasst_usable(parent):
            continue
        if _has_label(row, "quality") and _label_value(row, "quality") < 0.5:
            continue
        q, a = parent.get("text"), row.get("text")
        if not isinstance(q, str) or not isinstance(a, str) or not q.strip() or not a.strip():
            continue
        pairs.append({"prompt": q.strip(), "response": a.strip()})
    return pairs

def weighted_mix(trains, weights, total_cap, seed, max_repeat=INSTRUCT_MIX_MAX_REPEAT, prefer="Jaron"):
    """Build a capped mix by share. Upsample short sources; subsample long ones."""
    names = [name for name, rows in trains.items() if rows]
    if not names:
        raise ValueError("No instruction examples remained after filtering.")
    wsum = sum(weights.get(name, 1) for name in names)
    targets = {}
    for name in names:
        count = len(trains[name])
        ideal = max(1, int(total_cap * weights.get(name, 1) / wsum))
        targets[name] = min(ideal, count * max_repeat)
    # Give leftover cap to the preferred source first, then others, without exceeding max repeats.
    leftover = total_cap - sum(targets.values())
    if leftover > 0:
        order = [n for n in names if n == prefer] + [n for n in names if n != prefer]
        for name in order:
            room = len(trains[name]) * max_repeat - targets[name]
            add = min(room, leftover)
            targets[name] += add
            leftover -= add
            if leftover <= 0:
                break
    elif leftover < 0:
        # Rounding or tiny caps: reduce non-preferred sources first, keeping at least one example.
        overflow = -leftover
        for name in sorted((n for n in names if n != prefer), key=lambda n: targets[n], reverse=True):
            cut = min(overflow, targets[name] - 1)
            targets[name] -= cut
            overflow -= cut
            if overflow <= 0:
                break
        if overflow > 0 and prefer in targets:
            targets[prefer] = max(1, targets[prefer] - overflow)
    rng = random.Random(seed)
    mixed, used = [], {}
    for name in names:
        items = list(trains[name])
        target = targets[name]
        if target <= len(items):
            rng.shuffle(items)
            chosen = items[:target]
        else:
            reps, rem = divmod(target, len(items))
            chosen = items * reps + (rng.sample(items, rem) if rem else [])
        used[name] = len(chosen)
        mixed.extend(chosen)
    rng.shuffle(mixed)
    return mixed, used

def convert_instruct_source(name, rows):
    if name == "Dolly-15k":
        return [pair for row in rows if (pair := dolly_pair(row))]
    if name == "GSM8K":
        return [pair for row in rows if (pair := gsm8k_pair(row))]
    if name == "OpenAssistant OASST1":
        return oasst_pairs(rows)
    raise ValueError("Unknown instruction source: " + name)

def stream_instruct_source(name, tok, context, eot):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    spec = INSTRUCT_SOURCES[name]
    repo = spec["repo"]
    revision = HfApi().dataset_info(repo).sha
    if not revision:
        raise RuntimeError("Dataset revision unavailable: " + repo)
    print("Loading:", repo, spec["subset"], revision, flush=True)
    data = load_dataset(repo, spec["subset"], split=spec["split"], revision=revision)
    converted = convert_instruct_source(name, data)
    filtered = filter_pairs(converted, tok, context, eot)
    meta = {"name": name, "dataset": repo, "subset": spec["subset"], "split": spec["split"],
            "revision": revision, "license_note": spec["license_note"],
            "dataset_card": "https://huggingface.co/datasets/" + repo,
            "converted_count": len(converted), "filtered_count": len(filtered),
            "cap": spec["cap"], "weight": spec["weight"]}
    print(name, "converted", len(converted), "fit", len(filtered), flush=True)
    return filtered, meta

def _cap_split(rows, cap, seed):
    if len(rows) <= cap:
        return list(rows)
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:cap]

def prepare_instruct_mix(tokenizer, context, eot, seed, jaron_path, source_loader=stream_instruct_source):
    if not jaron_path:
        raise ValueError("Instruct mix requires the original Jaron JSON.")
    raw_jaron, file_meta = load_jaron_sources(jaron_path)
    jaron_rows = filter_pairs(raw_jaron, tokenizer, context, eot)
    source_rows = {"Jaron": jaron_rows}
    metadata = [{"name": "Jaron", "dataset": "Jaron instruction expanded",
                 "source_sha256": file_meta[0]["source_sha256"] if len(file_meta)==1 else [f["source_sha256"] for f in file_meta],
                 "files": file_meta, "source_count": len(raw_jaron), "filtered_count": len(jaron_rows),
                 "cap": JARON_MIX_CAP, "weight": JARON_MIX_WEIGHT,
                 "note": "Private contributor input; raw examples are not uploaded. Confirm permission and review accuracy. No public license is recorded in this repository. Multiple reviewed files are combined and de-duplicated."}]
    for name in INSTRUCT_SOURCES:
        rows, meta = source_loader(name, tokenizer, context, eot)
        source_rows[name] = rows
        metadata.append(meta)
    trains, vals = {}, {}
    for name, rows in source_rows.items():
        cap = JARON_MIX_CAP if name == "Jaron" else INSTRUCT_SOURCES[name]["cap"]
        train, val = instruction_pairs(rows)
        trains[name] = _cap_split(train, cap, seed)
        vals[name] = _cap_split(val, max(32, cap // 10), seed)
    mixed_train, train_used = weighted_mix(trains, instruct_mix_weights(), INSTRUCT_MIX_TRAIN_CAP, seed)
    mixed_val, val_used = [], {}
    for name, rows in vals.items():
        val_used[name] = len(rows)
        mixed_val.extend(rows)
    random.Random(seed).shuffle(mixed_val)
    if not mixed_train or not mixed_val:
        raise ValueError("Instruct mix produced an empty train or validation split.")
    t = encode_pairs(mixed_train, tokenizer, context, eot)
    v = encode_pairs(mixed_val, tokenizer, context, eot)
    source_sets = {name: encode_pairs(rows, tokenizer, context, eot) for name, rows in vals.items() if rows}
    for src in metadata:
        name = src["name"]
        src.update(unique_train=len(trains.get(name, [])), unique_validation=len(vals.get(name, [])),
                   mixed_train_count=train_used.get(name, 0), mixed_validation_count=val_used.get(name, 0),
                   split="normalized prompt hash; 10% held out; unique caps then weighted upsample of train")
    meta = {"recipe": INSTRUCT_MIX, "training_track": "instruction-tuning",
            "prompt_format": "User: {prompt}\\nAssistant:", "loss": "assistant tokens only",
            "train_cap": INSTRUCT_MIX_TRAIN_CAP, "max_repeat": INSTRUCT_MIX_MAX_REPEAT,
            "train_count": len(t), "validation_count": len(v), "sources": metadata}
    return t, v, source_sets, meta

def prepare(recipe, repo, tokenizer, context, eot, seed, jaron_path=None, budgets=None):
    """Return train/validation datasets, per-source validation sets, and provenance."""
    import torch
    from torch.utils.data import TensorDataset
    track,names = RECIPES[recipe]
    if track=="instruction-tuning":
        if recipe==INSTRUCT_MIX:
            return prepare_instruct_mix(tokenizer, context, eot, seed, jaron_path)
        if recipe=="Jaron":
            rows, file_meta = load_jaron_sources(jaron_path)
            source = {"dataset":"Jaron instruction expanded",
                      "source_sha256": file_meta[0]["source_sha256"] if len(file_meta)==1 else [f["source_sha256"] for f in file_meta],
                      "files": file_meta,
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
    budgets = RECIPE_BUDGETS.get(recipe, {}) if budgets is None else budgets
    for name in names:
        if name=="Public-domain literature":
            tr,va,meta = public_domain(repo,tokenizer)
        else:
            tr,va,meta = stream_source(name,tokenizer,context,seed,seen,budget=budgets.get(name))
        random.Random(seed).shuffle(tr)
        random.Random(seed).shuffle(va)
        train_sets.append(encode_windows(tr,context,eot))
        val_sets[name] = encode_windows(va,context,eot)
        metadata.append(meta)
    def combine(sets):
        return TensorDataset(*(torch.cat([s.tensors[i] for s in sets],dim=0) for i in (0,1)))
    return combine(train_sets),combine(list(val_sets.values())),val_sets,{"recipe":recipe,"training_track":track,"sources":metadata}
