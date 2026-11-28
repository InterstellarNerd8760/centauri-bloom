"""Fast greedy evaluation for post-training: locked eval plus held-out rule-checked tasks."""
from __future__ import annotations

from bloom.fastgen import generate_batch
from bloom.rewards import RewardConfig, chat_prompt, score_response
from bloom.rft import MAX_NEW_TOKENS, load_locked_eval, score_completion, summarize_locked


def _decode(tok, ids, eot):
    return tok.decode([i for i in ids if i != eot]).strip()


def greedy(model, tok, prompts, eot, device, max_new=MAX_NEW_TOKENS):
    prefixes = [tok.encode(chat_prompt(p)).ids for p in prompts]
    outs = generate_batch(model, prefixes, n=1, max_new=max_new, temperature=0, eot=eot, device=device)
    return [(_decode(tok, g[0]["ids"], eot), g[0]["ended"], len(g[0]["ids"])) for g in outs]


def evaluate_locked_fast(model, tok, eot, device, eval_set=None, repo_dir=None, max_new=MAX_NEW_TOKENS):
    """Same prompts, greedy decoding and scoring as bloom.rft.evaluate_locked, but batched and KV-cached."""
    eval_set = eval_set or load_locked_eval(repo_dir=repo_dir)
    items = eval_set["items"]
    outs = greedy(model, tok, [i["prompt"] for i in items], eot, device, max_new=max_new)
    results = []
    for item, (text, ended, _) in zip(items, outs):
        scored = score_completion(item["prompt"], text, gold=item.get("answer"), ended=ended, max_new=max_new)
        results.append({"id": item["id"], "source": item["source"], "prompt": item["prompt"],
                        "answer": item.get("answer"), "task": item["task"], "text": scored["text"],
                        "parsed": scored["parsed"], "exact": scored["exact"], "format_ok": scored["format_ok"],
                        "repetition": scored["repetition"], "ended": scored["ended"]})
    return {"summary": summarize_locked(results, eval_set), "results": results}


def evaluate_holdout(model, tok, eot, device, tasks, cfg=RewardConfig(), judge=None, max_new=MAX_NEW_TOKENS):
    """Greedy pass over fixed tasks. Reports accuracy by type and format/repetition/length rates.

    judge: optional callable list[(prompt, text)] -> list[float in 0..1] (see bloom.judge).
    """
    outs = greedy(model, tok, [t.prompt for t in tasks], eot, device, max_new=max_new)
    ratings = judge([(t.prompt, o[0]) for t, o in zip(tasks, outs)]) if judge is not None else [None] * len(tasks)
    by_type, rows = {}, []
    for task, (text, ended, n_tokens), rating in zip(tasks, outs, ratings):
        scored = score_response(task, text, ended, n_tokens, cfg, rating)
        bucket = by_type.setdefault(task.type, {"n": 0, "correct": 0, "reward": 0.0})
        bucket["n"] += 1
        bucket["correct"] += 1 if scored["correct"] else 0
        bucket["reward"] += scored["reward"]
        rows.append((task, text, scored))
    verifiable = [(t, s) for t, _, s in rows if t.verifiable]
    n = max(1, len(rows))
    summary = {
        "n_tasks": len(rows),
        "verifiable_accuracy": (sum(1 for _, s in verifiable if s["correct"]) / len(verifiable)) if verifiable else None,
        "mean_reward": sum(s["reward"] for _, _, s in rows) / n,
        "format_rate": sum(1 for _, _, s in rows if s["format_ok"]) / n,
        "repetition_rate": sum(1 for _, _, s in rows if s["repetition"]) / n,
        "ended_rate": sum(1 for _, _, s in rows if s["ended"]) / n,
        "mean_tokens": sum(s["tokens"] for _, _, s in rows) / n,
        "by_type": {k: {"n": v["n"], "accuracy": (v["correct"] / v["n"]) if k != "open" else None,
                        "mean_reward": v["reward"] / v["n"]} for k, v in sorted(by_type.items())},
    }
    samples = [{"type": t.type, "prompt": t.prompt, "gold": t.gold, "text": text, "reward": round(s["reward"], 3)}
               for t, text, s in rows[:: max(1, len(rows) // 24)]]
    return {"summary": summary, "samples": samples}


def composite(locked_summary, holdout_summary):
    """One number for picking the best RL checkpoint. Accuracy dominates; failure rates are guard terms."""
    acc = holdout_summary.get("verifiable_accuracy") or 0.0
    arith = locked_summary.get("arithmetic_exact") or 0.0
    rep = max(locked_summary.get("repetition_rate") or 0.0, holdout_summary.get("repetition_rate") or 0.0)
    ended = holdout_summary.get("ended_rate") or 0.0
    return 0.6 * acc + 0.2 * arith + 0.1 * (1.0 - rep) + 0.1 * ended
