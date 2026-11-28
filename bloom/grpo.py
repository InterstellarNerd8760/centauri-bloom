"""Response-level RL for Bloom chat checkpoints: GRPO-style and best-of-n, with rule-based rewards.

Both algorithms share one loop: sample a group of answers per prompt from the current policy (KV-cached,
batched), score every whole answer with `bloom.rewards`, then update.

  grpo         group-relative advantage (reward minus the group mean, divided by the group std), one
               on-policy policy-gradient step per batch, token-level normalisation, optional KL penalty
               to the frozen starting model (k3 estimator). No value network, no PPO ratio or clipping:
               every batch is used for exactly one update, so the ratio is 1.
  best-of-n    expert iteration: for each prompt keep the single highest-reward answer if it clears a bar
               and supervised-fine-tune on the kept answers. More conservative, lower variance, less sample
               efficient. Same rewards, same rollout, same checkpoint rules.

Outputs go to a new run folder in the layout `bloom.notebook_workflow.submit_candidate` expects, so the
candidate can only reach Hugging Face as a new PR. Nothing here reads contributor data: prompts are
generated from templates (plus optional public GSM8K *train* rows).
"""
from __future__ import annotations

import copy
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from bloom.rewards import (
    DEFAULT_MIX, RewardConfig, TaskSampler, build_holdout, chat_prompt, holdout_norms, score_response,
)

ALGOS = ("grpo", "best-of-n")


@dataclass(frozen=True)
class RLSettings:
    algo: str = "grpo"
    steps: int = 200
    prompts_per_step: int = 16
    group_size: int = 8
    max_new: int = 96
    temperature: float = 1.0
    lr: float = 1e-5
    warmup_steps: int = 10
    beta: float = 0.02            # KL penalty weight (grpo; best-of-n uses it too if > 0)
    std_normalize: bool = True
    micro_batch: int = 32
    grad_clip: float = 1.0
    bon_min_reward: float = 0.9   # best-of-n: lowest reward worth imitating for a checkable task
    bon_min_reward_open: float = 0.3
    eval_every: int = 25
    holdout_per_type: int = 6
    kl_stop: float = 1.0          # stop early if the recent mean KL per token to the start exceeds this
    min_gain: float = 0.01        # composite must beat the parent by this much to be uploadable
    seed: int = 20261128


def group_advantages(rewards, std_normalize=True, eps=1e-4):
    """Per-group centred (and optionally std-scaled) advantages. Constant groups give all zeros."""
    mean = sum(rewards) / len(rewards)
    centred = [r - mean for r in rewards]
    if not std_normalize:
        return centred
    std = math.sqrt(sum(c * c for c in centred) / len(rewards))
    if std < eps:
        return [0.0] * len(rewards)
    return [c / (std + eps) for c in centred]


def best_of_n_weights(rewards, min_reward):
    """1.0 on the single best sample when it clears the bar and beats the group mean, else all zeros."""
    best = max(range(len(rewards)), key=lambda i: (rewards[i], -i))
    mean = sum(rewards) / len(rewards)
    out = [0.0] * len(rewards)
    if rewards[best] >= min_reward and rewards[best] > mean + 1e-6:
        out[best] = 1.0
    return out


def collate(samples, eot, device):
    """Right-padded x, y, completion-token mask and per-row weight. y at prompt positions is masked out."""
    import torch
    seqs = [s["prefix"] + s["ids"] for s in samples]
    width = max(len(q) for q in seqs) - 1
    x = torch.full((len(seqs), width), eot, dtype=torch.long)
    y = torch.full((len(seqs), width), eot, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.float32)
    for r, (s, q) in enumerate(zip(samples, seqs)):
        n = len(q) - 1
        x[r, :n] = torch.tensor(q[:-1])
        y[r, :n] = torch.tensor(q[1:])
        mask[r, len(s["prefix"]) - 1:n] = 1.0
    weight = torch.tensor([s["weight"] for s in samples], dtype=torch.float32)
    return x.to(device), y.to(device), mask.to(device), weight.to(device)


def token_logprobs(model, x, y, temperature, amp):
    import torch
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        logits = model(x)
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    return logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)


def policy_loss_step(model, ref, samples, settings, eot, device, scaler):
    """Accumulate gradients for one update. Returns mean per-token KL to the reference and token count."""
    import torch
    amp = device == "cuda"
    total_tokens = sum(len(s["ids"]) for s in samples)
    kl_sum = 0.0
    for lo in range(0, len(samples), settings.micro_batch):
        chunk = samples[lo:lo + settings.micro_batch]
        x, y, mask, weight = collate(chunk, eot, device)
        logp = token_logprobs(model, x, y, settings.temperature, amp)
        pg = -(weight.unsqueeze(1) * logp * mask).sum()
        loss = pg
        if ref is not None and settings.beta > 0:
            with torch.no_grad():
                ref_logp = token_logprobs(ref, x, y, settings.temperature, amp)
            delta = ref_logp - logp
            kl = ((torch.exp(delta) - delta - 1.0) * mask).sum()
            loss = loss + settings.beta * kl
            kl_sum += float(kl.detach())
        scaler.scale(loss / total_tokens).backward()
    return kl_sum / max(1, total_tokens), total_tokens


def rollout(model, tok, tasks, settings, eot, device, cfg, judge, seed):
    """Sample a group per task and score every whole answer. Returns a list of per-task groups."""
    import torch
    from bloom.fastgen import generate_batch
    prefixes = [tok.encode(chat_prompt(t.prompt)).ids for t in tasks]
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        gens = generate_batch(model, prefixes, settings.group_size, settings.max_new, settings.temperature,
                              eot, device, seed=seed)
    texts = [[tok.decode([i for i in s["ids"] if i != eot]).strip() for s in group] for group in gens]
    ratings = None
    if judge is not None:
        open_idx = [(g, k) for g, t in enumerate(tasks) if not t.verifiable or cfg.judge_on_verifiable
                    for k in range(settings.group_size)]
        scores = judge([(tasks[g].prompt, texts[g][k]) for g, k in open_idx]) if open_idx else []
        ratings = dict(zip(open_idx, scores))
    groups = []
    for g, (task, prefix) in enumerate(zip(tasks, prefixes)):
        samples = []
        for k, sample in enumerate(gens[g]):
            scored = score_response(task, texts[g][k], sample["ended"], len(sample["ids"]), cfg,
                                    None if ratings is None else ratings.get((g, k)))
            samples.append({"task": task, "prefix": prefix, "ids": sample["ids"], "ended": sample["ended"],
                            "text": texts[g][k], "scored": scored, "weight": 0.0})
        groups.append(samples)
    return groups


def assign_weights(groups, settings):
    """Fill each sample's training weight. Returns the number of groups that carry signal."""
    active = 0
    for samples in groups:
        rewards = [s["scored"]["reward"] for s in samples]
        if settings.algo == "grpo":
            weights = group_advantages(rewards, settings.std_normalize)
        else:
            bar = settings.bon_min_reward if samples[0]["task"].verifiable else settings.bon_min_reward_open
            weights = best_of_n_weights(rewards, bar)
        for s, w in zip(samples, weights):
            s["weight"] = w
        active += 1 if any(abs(w) > 0 for w in weights) else 0
    return active


def step_stats(groups, active, kl, tokens):
    flat = [s for g in groups for s in g]
    verifiable = [s for s in flat if s["task"].verifiable]
    n = max(1, len(flat))
    return {
        "reward": sum(s["scored"]["reward"] for s in flat) / n,
        "accuracy": (sum(1 for s in verifiable if s["scored"]["correct"]) / len(verifiable)) if verifiable else None,
        "format_rate": sum(1 for s in flat if s["scored"]["format_ok"]) / n,
        "repetition_rate": sum(1 for s in flat if s["scored"]["repetition"]) / n,
        "ended_rate": sum(1 for s in flat if s["ended"]) / n,
        "mean_tokens": sum(len(s["ids"]) for s in flat) / n,
        "active_groups": active, "groups": len(groups), "kl": kl, "update_tokens": tokens,
    }


def lr_at(step, settings):
    return settings.lr * min(1.0, step / max(1, settings.warmup_steps))


def train_policy(model, tok, eot, device, settings, cfg, sampler, holdout, locked_eval, judge,
                 log=print, on_eval=None, log_path=None):
    """Run the RL loop in place. Calls on_eval(step, evaluation) at each evaluation. Returns the history."""
    import torch
    from bloom.policy_eval import composite, evaluate_holdout, evaluate_locked_fast
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    ref = None
    if settings.beta > 0:
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    history, recent_kl, recent_active, t0 = [], [], [], time.monotonic()

    def evaluate(step):
        locked = evaluate_locked_fast(model, tok, eot, device, eval_set=locked_eval)
        held = evaluate_holdout(model, tok, eot, device, holdout, cfg, judge)
        ev = {"step": step, "locked": locked["summary"], "holdout": held["summary"], "samples": held["samples"],
              "composite": composite(locked["summary"], held["summary"])}
        log("eval step %d | composite %.4f | holdout acc %.3f | locked arithmetic %.3f | repetition %.3f" % (
            step, ev["composite"], held["summary"]["verifiable_accuracy"] or 0.0,
            locked["summary"]["arithmetic_exact"] or 0.0, held["summary"]["repetition_rate"]))
        if on_eval:
            on_eval(step, ev)
        return ev

    evals = [evaluate(0)]
    stopped = None
    for step in range(1, settings.steps + 1):
        tasks = sampler.batch(settings.prompts_per_step)
        model.eval()
        groups = rollout(model, tok, tasks, settings, eot, device, cfg, judge, seed=settings.seed * 100003 + step)
        active = assign_weights(groups, settings)
        train_samples = [s for g in groups for s in g if abs(s["weight"]) > 0 or
                         (settings.algo == "grpo" and any(abs(x["weight"]) > 0 for x in g))]
        kl, tokens = 0.0, 0
        if train_samples:
            model.train()
            opt.zero_grad(set_to_none=True)
            kl, tokens = policy_loss_step(model, ref, train_samples, settings, eot, device, scaler)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, settings)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            scaler.step(opt)
            scaler.update()
        stats = {"step": step, **step_stats(groups, active, kl, tokens), "elapsed": time.monotonic() - t0}
        history.append(stats)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")
        if step == 1 or step % 5 == 0:
            log("step %d/%d | reward %.3f | acc %s | active %d/%d | kl %.4f | tokens %.1f | %.0fs" % (
                step, settings.steps, stats["reward"],
                "n/a" if stats["accuracy"] is None else "%.3f" % stats["accuracy"],
                active, len(groups), kl, stats["mean_tokens"], stats["elapsed"]))
        recent_active = (recent_active + [active / max(1, len(groups))])[-10:]
        if step % 20 == 0 and len(recent_active) == 10 and sum(recent_active) / 10 < 0.2:
            log("Warning: under 20%% of groups carry signal (the policy is near-deterministic or the tasks are too "
                "hard). Raise the sampling temperature to 1.1-1.3 or add easier tasks.")
        recent_kl = (recent_kl + [kl])[-10:]
        if len(recent_kl) == 10 and sum(recent_kl) / 10 > settings.kl_stop:
            stopped = "kl_stop at step %d (mean KL %.3f per token)" % (step, sum(recent_kl) / 10)
            log("Stopping early:", stopped)
            evals.append(evaluate(step))
            break
        if step % settings.eval_every == 0 or step == settings.steps:
            evals.append(evaluate(step))
    return {"history": history, "evals": evals, "stopped": stopped, "steps_run": len(history)}


def settings_from(**overrides):
    s = RLSettings(**{k: v for k, v in overrides.items() if v is not None})
    if s.algo not in ALGOS:
        raise ValueError("algo must be one of %s" % (ALGOS,))
    if not 1 <= s.steps <= 5000 or s.group_size < 2 or s.prompts_per_step < 1:
        raise ValueError("Invalid RL step, group, or prompt counts.")
    if s.max_new < 8 or not 0.2 <= s.temperature <= 2.0:
        raise ValueError("Invalid max_new or temperature.")
    return s
