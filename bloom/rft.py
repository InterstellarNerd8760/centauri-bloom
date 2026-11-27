"""Rejection-sampling fine-tune helpers for Bloom-Chat-0.2.

This module does not implement DPO, GRPO, TRL, or playground feedback.
Pretraining recipes and their defaults are unchanged. RFT starts from a
Chat checkpoint and writes a new run folder only.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from pathlib import Path

LOCKED_EVAL_VERSION = "rft-eval-v1"
LOCKED_EVAL_SEED = 47
LOCKED_EVAL_ARITHMETIC = 200
PILOT_SIZE = 500
PILOT_PER_SOURCE = 250
FULL_ARITHMETIC = 2000
FULL_GSM8K = 2000
SAMPLES_PER_PROMPT = 8
MAX_NEW_TOKENS = 128
RFT_TEMPERATURE = 0.8
RFT_LR = 2e-5
REPLAY_FRACTION = 0.25
CHAT_TEMPLATE = "User: {prompt}\nAssistant:"
STAGES = ("sft", "pretrain-continue", "rft", "rft-baseline")
CHAT_PARENT_REPOS = {
    "14M": "Centauri-Bloom/Bloom-Chat-0.1",
    "50M": "Centauri-Bloom/Bloom-Chat-0.2",
}
DEFAULT_CHAT_REVISION = "refs/pr/1"

# SHA-256 of the canonical locked-eval payload (see canonical_eval_payload).
# Regenerating the committed JSON must keep this digest.
LOCKED_EVAL_SHA256 = "9dbda985c231b1d2b01047a7ee00542c8bf38924d365f0992e6939762e08ffb7"

_NUM = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

ARITH_TEMPLATES = (
    ("add", "What is {a} plus {b}?", lambda a, b: a + b),
    ("sub_from", "Subtract {b} from {a}.", lambda a, b: a - b),
    ("mul", "Multiply {a} by {b}.", lambda a, b: a * b),
    ("times", "What is {a} times {b}?", lambda a, b: a * b),
    ("minus", "What is {a} minus {b}?", lambda a, b: a - b),
)

# Fixed qualitative and exact-match probes. "What is 12 times 5?" is required.
PROBES = (
    {"id": "probe-00", "prompt": "What is 12 times 5?", "answer": "60", "task": "exact-number"},
    {"id": "probe-01", "prompt": "What is 7 plus 8?", "answer": "15", "task": "exact-number"},
    {"id": "probe-02", "prompt": "Multiply 9 by 6.", "answer": "54", "task": "exact-number"},
    {"id": "probe-03", "prompt": "Subtract 15 from 40.", "answer": "25", "task": "exact-number"},
    {"id": "probe-04", "prompt": "What is 100 minus 37?", "answer": "63", "task": "exact-number"},
    {"id": "probe-05", "prompt": "What is Earth?", "answer": None, "task": "open"},
    {"id": "probe-06", "prompt": "Write a Python function that adds two numbers.", "answer": None, "task": "open"},
    {"id": "probe-07", "prompt": "Say hello politely.", "answer": None, "task": "open"},
    {"id": "probe-08", "prompt": "What is the capital of France?", "answer": None, "task": "open"},
    {"id": "probe-09", "prompt": "Can you be certain about every answer?", "answer": None, "task": "open"},
)


def normalize_prompt(text):
    if not isinstance(text, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(" .!?")


def locked_eval_path(repo_dir=None):
    root = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[1]
    return root / "datasets" / "rft-eval-v1" / "locked-eval.json"


def canonical_eval_payload(items, version=LOCKED_EVAL_VERSION, seed=LOCKED_EVAL_SEED):
    body = {"version": version, "seed": seed, "prompt_format": CHAT_TEMPLATE, "items": items}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def eval_digest(items, version=LOCKED_EVAL_VERSION, seed=LOCKED_EVAL_SEED):
    return hashlib.sha256(canonical_eval_payload(items, version, seed).encode("utf-8")).hexdigest()


def parse_final_number(text):
    """Return a canonical number from model or gold text, or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    source = text
    if "####" in text:
        source = text.split("####")[-1]
    matches = list(_NUM.finditer(source))
    if not matches and source is not text:
        matches = list(_NUM.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(0).replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if value.is_integer():
        return str(int(value))
    return format(value, ".10g")


def numbers_match(predicted, gold):
    left, right = parse_final_number(predicted), parse_final_number(gold if isinstance(gold, str) else str(gold) if gold is not None else "")
    return left is not None and left == right


def gsm8k_gold(response):
    return parse_final_number(response or "")


def repetition_flags(text, max_new=MAX_NEW_TOKENS, ended=True):
    flags = []
    if not isinstance(text, str) or not text.strip():
        flags.append("empty")
        return flags
    words = re.findall(r"\S+", text)
    if len(words) >= 10:
        grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
        if len(grams) != len(set(grams)):
            flags.append("repeated-5gram")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2 and len(lines) != len(set(lines)):
        flags.append("repeated-line")
    if re.search(r"(.{8,}?)\1{2,}", text):
        flags.append("repeated-span")
    if not ended and len(words) >= max(8, max_new // 8):
        flags.append("no-eos")
    return flags


def format_ok(text, prompt=""):
    if not isinstance(text, str) or not text.strip():
        return False
    if re.search(r"\bUser:", text):
        return False
    if "\nAssistant:" in text:
        return False
    stripped = text.strip()
    if prompt and normalize_prompt(stripped).startswith(normalize_prompt(prompt)) and len(stripped) > 20:
        return False
    return True


def score_completion(prompt, text, gold=None, ended=True, max_new=MAX_NEW_TOKENS):
    flags = repetition_flags(text, max_new=max_new, ended=ended)
    fmt = format_ok(text, prompt)
    parsed = parse_final_number(text)
    exact = numbers_match(text, gold) if gold is not None and str(gold).strip() != "" else None
    keep = bool(fmt and not flags and exact)
    return {
        "text": text,
        "ended": ended,
        "parsed": parsed,
        "exact": exact,
        "format_ok": fmt,
        "repetition": flags,
        "keep": keep,
    }


def choose_kept(scored):
    """Keep the shortest clean exact match; otherwise None."""
    good = [s for s in scored if s.get("keep")]
    if not good:
        return None
    good.sort(key=lambda s: (len(s["text"]), s.get("index", 0)))
    return good[0]


def yield_report(records):
    """records: iterable of dicts with source and optional kept completion."""
    by_source = {}
    for row in records:
        source = row.get("source") or "unknown"
        bucket = by_source.setdefault(source, {"kept": 0, "total": 0, "correct_any": 0})
        bucket["total"] += 1
        if row.get("kept"):
            bucket["kept"] += 1
        if row.get("any_exact"):
            bucket["correct_any"] += 1
    total = sum(v["total"] for v in by_source.values())
    kept = sum(v["kept"] for v in by_source.values())
    return {
        "kept": kept,
        "total": total,
        "kept_rate": (kept / total) if total else 0.0,
        "by_source": by_source,
    }


def arithmetic_pool():
    """Deterministic arithmetic prompts. Answers may be negative."""
    items = []
    for a in range(1, 81):
        for b in range(1, 41):
            for kind, template, fn in ARITH_TEMPLATES:
                prompt = template.format(a=a, b=b)
                items.append({
                    "prompt": prompt,
                    "answer": str(fn(a, b)),
                    "task": "exact-number",
                    "source": "arithmetic",
                    "kind": kind,
                    "a": a,
                    "b": b,
                })
    return items


def _unique_pool(rows, excluded):
    seen = set(excluded)
    out = []
    for row in rows:
        norm = normalize_prompt(row["prompt"])
        if not norm or norm in seen:
            continue
        seen.add(norm)
        item = dict(row)
        item["norm"] = norm
        out.append(item)
    out.sort(key=lambda r: (r["norm"], r["prompt"]))
    return out


def probe_items():
    items = []
    for row in PROBES:
        item = {
            "id": row["id"],
            "source": "probe",
            "prompt": row["prompt"],
            "answer": row["answer"],
            "task": row["task"],
        }
        item["norm"] = normalize_prompt(item["prompt"])
        items.append(item)
    return items


def build_locked_eval_items(n_arithmetic=LOCKED_EVAL_ARITHMETIC, seed=LOCKED_EVAL_SEED):
    probes = probe_items()
    excluded = {p["norm"] for p in probes}
    pool = _unique_pool(arithmetic_pool(), excluded)
    if len(pool) < n_arithmetic:
        raise RuntimeError("Arithmetic pool is smaller than the locked eval set.")
    chosen = random.Random(seed).sample(pool, n_arithmetic)
    chosen.sort(key=lambda r: r["prompt"])
    items = []
    for probe in probes:
        items.append({
            "id": probe["id"],
            "source": "probe",
            "prompt": probe["prompt"],
            "answer": probe["answer"],
            "task": probe["task"],
        })
    for i, row in enumerate(chosen):
        items.append({
            "id": "arith-%03d" % i,
            "source": "arithmetic",
            "prompt": row["prompt"],
            "answer": row["answer"],
            "task": "exact-number",
        })
    return items


def public_eval_item(item):
    return {k: item[k] for k in ("id", "source", "prompt", "answer", "task")}


def write_locked_eval(path=None, repo_dir=None):
    path = Path(path) if path else locked_eval_path(repo_dir)
    items = [public_eval_item(x) for x in build_locked_eval_items()]
    digest = eval_digest(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": LOCKED_EVAL_VERSION,
        "seed": LOCKED_EVAL_SEED,
        "prompt_format": CHAT_TEMPLATE,
        "sha256": digest,
        "items": items,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path, digest


def load_locked_eval(path=None, repo_dir=None, expected_sha256=LOCKED_EVAL_SHA256):
    path = Path(path) if path else locked_eval_path(repo_dir)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != LOCKED_EVAL_VERSION:
        raise ValueError("Locked eval version mismatch: %s" % raw.get("version"))
    if raw.get("seed") != LOCKED_EVAL_SEED:
        raise ValueError("Locked eval seed mismatch.")
    items = raw.get("items")
    if not isinstance(items, list) or len(items) != LOCKED_EVAL_ARITHMETIC + len(PROBES):
        raise ValueError("Locked eval item count is not the committed set.")
    digest = eval_digest(items, raw["version"], raw["seed"])
    if raw.get("sha256") != digest:
        raise ValueError("Locked eval file hash does not match its items; the set may have drifted.")
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Locked eval digest does not match the digest pinned in bloom.rft.")
    # Rebuild must reproduce the same 200 arithmetic prompts.
    rebuilt = [public_eval_item(x) for x in build_locked_eval_items()]
    if rebuilt != items:
        raise ValueError("Locked eval items no longer match the deterministic builder.")
    return {"path": str(path), "sha256": digest, "version": raw["version"], "seed": raw["seed"],
            "prompt_format": raw.get("prompt_format", CHAT_TEMPLATE), "items": items}


def locked_eval_norms(eval_set=None, repo_dir=None):
    eval_set = eval_set or load_locked_eval(repo_dir=repo_dir)
    return {normalize_prompt(item["prompt"]) for item in eval_set["items"]}


# Exact (operation, a, b) facts used by numbered probes. Training must not
# see the same calculation under a different wording.
PROBE_FACTS = {
    ("times", 12, 5), ("mul", 12, 5), ("times", 5, 12), ("mul", 5, 12),
    ("add", 7, 8), ("add", 8, 7),
    ("mul", 9, 6), ("times", 9, 6), ("mul", 6, 9), ("times", 6, 9),
    ("sub_from", 40, 15), ("minus", 40, 15),
    ("minus", 100, 37), ("sub_from", 100, 37),
}


def training_arithmetic_prompts(n, eval_norms=None, seed=LOCKED_EVAL_SEED, repo_dir=None):
    eval_norms = eval_norms if eval_norms is not None else locked_eval_norms(repo_dir=repo_dir)
    pool = [row for row in _unique_pool(arithmetic_pool(), eval_norms)
            if (row.get("kind"), row.get("a"), row.get("b")) not in PROBE_FACTS]
    if n > len(pool):
        raise ValueError("Requested %s arithmetic prompts but only %s remain outside the locked eval." % (n, len(pool)))
    chosen = random.Random(seed).sample(pool, n)
    return [{"source": "arithmetic", "prompt": row["prompt"], "answer": row["answer"]} for row in chosen]


def iter_gsm8k_prompts(rows, eval_norms=None):
    eval_norms = eval_norms or set()
    seen = set(eval_norms)
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = row.get("prompt") or row.get("question")
        response = row.get("response") or row.get("answer")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        gold = gsm8k_gold(response)
        if gold is None:
            continue
        norm = normalize_prompt(prompt)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        yield {"source": "gsm8k", "prompt": prompt.strip(), "answer": gold}


def select_gsm8k_prompts(rows, n, eval_norms=None, seed=LOCKED_EVAL_SEED):
    pool = list(iter_gsm8k_prompts(rows, eval_norms=eval_norms))
    pool.sort(key=lambda r: normalize_prompt(r["prompt"]))
    if not pool:
        return []
    if n >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n)


def build_rft_prompts(mode, gsm8k_rows=None, eval_set=None, seed=LOCKED_EVAL_SEED, repo_dir=None):
    """Return a disjoint training prompt list for the pilot or full RFT run."""
    if mode not in ("pilot", "full"):
        raise ValueError("RFT mode must be 'pilot' or 'full'.")
    eval_set = eval_set or load_locked_eval(repo_dir=repo_dir)
    norms = locked_eval_norms(eval_set)
    arith_n = PILOT_PER_SOURCE if mode == "pilot" else FULL_ARITHMETIC
    gsm_n = PILOT_PER_SOURCE if mode == "pilot" else FULL_GSM8K
    arithmetic = training_arithmetic_prompts(arith_n, eval_norms=norms, seed=seed, repo_dir=repo_dir)
    gsm8k = select_gsm8k_prompts(gsm8k_rows or [], gsm_n, eval_norms=norms, seed=seed)
    # If GSM8K is short, fill the requested total with extra arithmetic.
    target = PILOT_SIZE if mode == "pilot" else (FULL_ARITHMETIC + FULL_GSM8K)
    if len(arithmetic) + len(gsm8k) < target:
        need = target - len(arithmetic) - len(gsm8k)
        used = norms | {normalize_prompt(r["prompt"]) for r in arithmetic}
        extra = training_arithmetic_prompts(min(need + len(arithmetic), 10000), eval_norms=used, seed=seed + 1, repo_dir=repo_dir)
        arithmetic = arithmetic + extra[:need]
    prompts = arithmetic + gsm8k
    random.Random(seed).shuffle(prompts)
    if mode == "pilot":
        prompts = prompts[:PILOT_SIZE]
    overlap = [p for p in prompts if normalize_prompt(p["prompt"]) in norms]
    if overlap:
        raise RuntimeError("RFT training prompts overlap the locked eval set.")
    return prompts


def encode_chat_prefix(prompt, tok):
    return tok.encode(CHAT_TEMPLATE.format(prompt=prompt)).ids


def _decode_new(tok, token_ids):
    if not token_ids:
        return ""
    if hasattr(tok, "decode"):
        return tok.decode(token_ids)
    return ""


def generate_completions(model, tok, prompts, context, eot, device, n=SAMPLES_PER_PROMPT,
                         max_new=MAX_NEW_TOKENS, temperature=RFT_TEMPERATURE, seed=LOCKED_EVAL_SEED):
    """Sample n completions per prompt. Prompts are decoded independently; the n samples share a batch."""
    import torch
    model.eval()
    torch.manual_seed(seed)
    results = []
    with torch.no_grad():
        for prompt in prompts:
            prefix = encode_chat_prefix(prompt, tok) or [eot]
            if len(prefix) >= context:
                prefix = prefix[-(context - 1):]
            ids = torch.tensor([prefix] * n, device=device, dtype=torch.long)
            ended = [False] * n
            new_tokens = [[] for _ in range(n)]
            for _ in range(max_new):
                if all(ended):
                    break
                x = ids[:, -context:]
                logits = model(x)[:, -1].float()
                if temperature is None or temperature <= 0:
                    nxt = logits.argmax(dim=-1)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    nxt = torch.multinomial(probs, 1).squeeze(-1)
                for i in range(n):
                    if ended[i]:
                        continue
                    token = int(nxt[i].item())
                    if token == eot:
                        ended[i] = True
                        continue
                    new_tokens[i].append(token)
                if ids.shape[1] >= context:
                    ids = torch.cat([ids[:, 1:], nxt.unsqueeze(1)], dim=1)
                else:
                    ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            completions = []
            for i in range(n):
                text = _decode_new(tok, new_tokens[i]).strip()
                completions.append({"index": i, "text": text, "ended": ended[i], "tokens": len(new_tokens[i])})
            results.append(completions)
    return results


def score_prompt_completions(prompt_row, completions, max_new=MAX_NEW_TOKENS):
    scored = []
    for comp in completions:
        item = score_completion(prompt_row["prompt"], comp["text"], gold=prompt_row.get("answer"),
                                ended=comp.get("ended", True), max_new=max_new)
        item["index"] = comp.get("index", 0)
        item["tokens"] = comp.get("tokens")
        scored.append(item)
    kept = choose_kept(scored)
    return {
        "source": prompt_row.get("source"),
        "prompt": prompt_row["prompt"],
        "answer": prompt_row.get("answer"),
        "completions": scored,
        "kept": None if kept is None else kept["text"],
        "kept_index": None if kept is None else kept["index"],
        "any_exact": any(s.get("exact") for s in scored),
    }


def summarize_locked(results, eval_set):
    items = eval_set["items"]
    numbered = [r for r, item in zip(results, items) if item.get("task") == "exact-number"]
    arithmetic = [r for r, item in zip(results, items) if item.get("source") == "arithmetic"]
    probes = [r for r, item in zip(results, items) if item.get("source") == "probe"]
    probe_numbered = [r for r, item in zip(results, items)
                      if item.get("source") == "probe" and item.get("task") == "exact-number"]

    def rate(rows, key):
        if not rows:
            return None
        return sum(1 for r in rows if r.get(key)) / len(rows)

    times = next((r for r, item in zip(results, items) if item["prompt"] == "What is 12 times 5?"), None)
    return {
        "eval_sha256": eval_set["sha256"],
        "n_items": len(items),
        "arithmetic_exact": rate(arithmetic, "exact"),
        "arithmetic_n": len(arithmetic),
        "numbered_exact": rate(numbered, "exact"),
        "probe_numbered_exact": rate(probe_numbered, "exact"),
        "format_rate": rate(results, "format_ok"),
        "repetition_rate": sum(1 for r in results if r.get("repetition")) / len(results) if results else 0.0,
        "times_12_5": None if times is None else {
            "exact": times.get("exact"),
            "text": times.get("text"),
            "parsed": times.get("parsed"),
        },
    }


def evaluate_locked(model, tok, context, eot, device, eval_set=None, repo_dir=None, max_new=MAX_NEW_TOKENS):
    eval_set = eval_set or load_locked_eval(repo_dir=repo_dir)
    prompts = [item["prompt"] for item in eval_set["items"]]
    generations = generate_completions(model, tok, prompts, context, eot, device, n=1,
                                       max_new=max_new, temperature=0, seed=LOCKED_EVAL_SEED)
    results = []
    for item, comps in zip(eval_set["items"], generations):
        scored = score_completion(item["prompt"], comps[0]["text"], gold=item.get("answer"),
                                  ended=comps[0]["ended"], max_new=max_new)
        results.append({
            "id": item["id"],
            "source": item["source"],
            "prompt": item["prompt"],
            "answer": item.get("answer"),
            "task": item["task"],
            "text": scored["text"],
            "parsed": scored["parsed"],
            "exact": scored["exact"],
            "format_ok": scored["format_ok"],
            "repetition": scored["repetition"],
            "ended": scored["ended"],
        })
    return {"summary": summarize_locked(results, eval_set), "results": results}


def is_chat_parent(metrics, model_size="50M"):
    target = (metrics or {}).get("target_repo")
    track = (metrics or {}).get("training_track") or (metrics or {}).get("dataset")
    expected = CHAT_PARENT_REPOS[model_size]
    if target == expected:
        return True
    if target and target != expected:
        return False
    if track in ("instruction-tuning", "Instruct mix (Jaron + open)"):
        return True
    return False


def chat_parent_revisions(requested=None):
    """Try the unmerged Chat PR first, then main once it is merged."""
    order = []
    for rev in (requested or DEFAULT_CHAT_REVISION, "main"):
        if rev and rev not in order:
            order.append(rev)
    return order
