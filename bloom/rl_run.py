"""Run response-level RL on a Bloom chat checkpoint and write a candidate in the standard run layout.

New weights go to a new run folder and, later, only to a new Hugging Face PR through
`bloom.notebook_workflow.submit_candidate`. The parent checkpoint is never written to.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from bloom.grpo import RLSettings, settings_from, train_policy
from bloom.notebook_workflow import pick_device, sha256
from bloom.posttrain import newest_chat
from bloom.rewards import DEFAULT_MIX, RewardConfig, TaskSampler, build_holdout, holdout_norms
from bloom.rft import is_chat_parent, load_locked_eval

STAGE = "rl"


def run_rl(api, repo_dir, out_dir, settings=None, parent_run_dir=None, parent_hf_repo=None,
           parent_hf_revision=None, device=None, judge=None, judge_info=None, gsm8k_rows=None,
           extra_open_prompts=(), reward_cfg=None, log=print, use_gsm8k=False):
    import torch
    from tokenizers import Tokenizer
    from bloom.checkpoint import config_hash, load_weights, save_weights
    from bloom.model import Bloom
    from bloom.trainer import CHAT_REPOS, PARAM_COUNT_50M, load_hub_parent, load_local_parent

    settings = settings or RLSettings()
    cfg_rw = reward_cfg or RewardConfig()
    if parent_run_dir and (parent_hf_repo or parent_hf_revision):
        raise ValueError("Set PARENT_RUN_DIR or PARENT_HF_REPO/PARENT_HF_REVISION, not both.")
    device = device or pick_device()
    if device == "cpu" and settings.steps > 5:
        log("Warning: RL on CPU is very slow; use a GPU for real runs.")
    target_repo_id = CHAT_REPOS["50M"]
    out = Path(out_dir)
    if (out / "metrics.json").exists():
        raise RuntimeError("This run already completed. Use the notebook's upload cell, or choose a new folder.")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Run folder is not empty. RL runs are not resumable; start a new folder.")
    out.mkdir(parents=True, exist_ok=True)

    if parent_run_dir:
        parent, cfg, pm, weight_path, revision = load_local_parent(parent_run_dir, "50M", repo_dir)
    else:
        repo = parent_hf_repo or target_repo_id
        if repo != target_repo_id:
            raise ValueError("RL starts from %s, not %s." % (target_repo_id, repo))
        rev = parent_hf_revision or None
        if rev in (None, "", "auto"):
            rev, found = newest_chat(api, repo, token=api.token)
            log("Newest SFT candidate:", json.dumps(found))
        parent, cfg, pm, weight_path, revision = load_hub_parent(api, repo, token=api.token, revision=rev)
    if not is_chat_parent(pm, "50M"):
        raise ValueError("RL must start from a Bloom-Chat checkpoint (after SFT), not a text/pretrain checkpoint.")
    tok = Tokenizer.from_file(str(parent / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None or tok.get_vocab_size() != cfg.vocab_size or config_hash(cfg) != pm["config_sha256"]:
        raise ValueError("Parent tokenizer/config verification failed.")
    model = Bloom(cfg)
    model.load_state_dict(load_weights(weight_path, pm["weights_sha256"], model.state_dict()), strict=True)
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    if params != PARAM_COUNT_50M:
        raise RuntimeError("Expected %d parameters, got %d." % (PARAM_COUNT_50M, params))
    try:
        target_revision = api.model_info(target_repo_id).sha
    except Exception as exc:
        raise RuntimeError("Target repo %s is not reachable; this trainer will not create it." % target_repo_id) from exc

    if use_gsm8k and gsm8k_rows is None:
        from bloom.trainer import _load_gsm8k_rows
        gsm8k_rows, gsm_meta = _load_gsm8k_rows(tok, cfg.context, eot)
        log("GSM8K train rows:", len(gsm8k_rows), "|", json.dumps(gsm_meta, default=str)[:200])
    locked = load_locked_eval(repo_dir=repo_dir)
    holdout = build_holdout(settings.holdout_per_type, repo_dir=repo_dir, gsm8k_rows=gsm8k_rows,
                            gsm8k_n=min(24, settings.holdout_per_type * 4) if gsm8k_rows else 0)
    sampler = TaskSampler(settings.seed, mix=DEFAULT_MIX, excluded_norms=holdout_norms(holdout),
                          gsm8k_rows=gsm8k_rows, extra_open_prompts=extra_open_prompts, repo_dir=repo_dir)
    run_settings = {"rl": settings.__dict__, "reward": cfg_rw.__dict__, "parent_revision": revision,
                    "target_repo": target_repo_id, "target_revision": target_revision,
                    "parent_run_dir": str(parent_run_dir or ""), "task_mix": dict(DEFAULT_MIX),
                    "judge": judge_info or None, "tokenizer_sha256": sha256(parent / "tokenizer.json")}
    (out / "run-settings.json").write_text(json.dumps(run_settings, indent=2))
    shutil.copyfile(parent / "tokenizer.json", out / "tokenizer.json")
    (out / "config.json").write_text(json.dumps(cfg.__dict__, indent=2, sort_keys=True))

    best = {"step": 0, "composite": None, "eval": None}

    def on_eval(step, ev):
        if best["composite"] is None or ev["composite"] > best["composite"] + 1e-9:
            best.update(step=step, composite=ev["composite"], eval=ev)
            if step > 0:
                save_weights(out / "best.safetensors", model.state_dict(), model.state_dict())
                log("New best checkpoint at step %d (composite %.4f)" % (step, ev["composite"]))

    result = train_policy(model, tok, eot, device, settings, cfg_rw, sampler, holdout, locked, judge, log=log,
                          on_eval=on_eval, log_path=str(out / "rl-log.jsonl"))
    parent_eval = result["evals"][0]
    gain = best["composite"] - parent_eval["composite"]
    uploadable = best["step"] > 0 and gain >= settings.min_gain
    steps_run = result["steps_run"]
    evaluation = {
        "stage": STAGE, "algo": settings.algo, "selected_step": best["step"], "composite_gain": gain,
        "parent": {"locked": parent_eval["locked"], "holdout": parent_eval["holdout"],
                   "composite": parent_eval["composite"]},
        "candidate": {"locked": best["eval"]["locked"], "holdout": best["eval"]["holdout"],
                      "composite": best["eval"]["composite"]},
        "candidate_samples": best["eval"]["samples"], "parent_samples": parent_eval["samples"],
        "eval_curve": [{"step": e["step"], "composite": e["composite"],
                        "holdout_accuracy": e["holdout"]["verifiable_accuracy"],
                        "locked_arithmetic": e["locked"]["arithmetic_exact"],
                        "repetition_rate": e["holdout"]["repetition_rate"]} for e in result["evals"]],
        "stopped": result["stopped"],
        "note": "Greedy decoding on a locked eval (210 prompts) and 72+ fixed rule-checked synthetic tasks that "
                "training prompts never overlap. Composite = 0.6*holdout accuracy + 0.2*locked arithmetic + "
                "0.1*(1-repetition) + 0.1*ended rate.",
    }
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False))
    provenance = {
        "stage": STAGE, "algo": settings.algo, "private_data": "none",
        "prompt_sources": ["generated templates (bloom.rewards)"] + (["openai/gsm8k train split (MIT)"] if gsm8k_rows else [])
                          + (["caller-supplied open prompts"] if extra_open_prompts else []),
        "task_mix": dict(DEFAULT_MIX), "holdout_tasks": len(holdout), "locked_eval_sha256": locked["sha256"],
        "parent_revision": revision, "judge": judge_info or None,
        "note": "No contributor (Jaron) data is read, used as prompts, or uploaded by the RL stage."}
    (out / "data-provenance.json").write_text(json.dumps(provenance, indent=2))
    metrics = {
        "additional_steps": steps_run, "stage": STAGE, "algo": settings.algo, "run_name": out.name,
        "candidate_uploadable": bool(uploadable), "selected_step": best["step"],
        "step": int(pm.get("step") or 0) + steps_run, "step_scope": "RL update steps on top of the parent",
        "parent_revision": revision, "parent_weights_sha256": pm["weights_sha256"],
        "parent_reported_train_tokens": pm.get("train_tokens"),
        "cumulative_train_tokens": pm.get("cumulative_train_tokens"),
        "parameters": params, "config_sha256": config_hash(cfg), "training_track": "response-level-rl",
        "dataset": "generated verifiable tasks + format/repetition/length rewards", "device": device,
        "model_size": "50M", "target_repo": target_repo_id, "target_revision": target_revision,
        "locked_eval_sha256": locked["sha256"], "parent_locked_eval": parent_eval["locked"],
        "candidate_locked_eval": best["eval"]["locked"], "parent_composite": parent_eval["composite"],
        "candidate_composite": best["eval"]["composite"], "composite_gain": gain,
        "rl_settings": settings.__dict__, "learning_rate": settings.lr, "seed": settings.seed,
        "prompt_format": "User: {prompt}\\nAssistant:",
        "data_provenance_sha256": sha256(out / "data-provenance.json"),
        "evaluation_sha256": sha256(out / "evaluation.json"),
    }
    if uploadable:
        final = out / ("step-%06d.safetensors" % steps_run)
        (out / "best.safetensors").replace(final)
        metrics["weights_sha256"] = sha256(final)
        metrics["note"] = "RL candidate: best held-out checkpoint (step %d of %d). Parent was not modified." % (best["step"], steps_run)
    else:
        (out / "best.safetensors").unlink(missing_ok=True)
        metrics["note"] = ("RL did not beat the parent by min_gain=%.3f on the held-out composite; no candidate weights "
                           "were written and nothing should be uploaded." % settings.min_gain)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    log("COMPLETED RL run:", out, "| uploadable:", uploadable, "| gain: %.4f" % gain)
    return metrics


def run_from_payload(payload_json):
    """Entry point for the notebook subprocess. The HF token comes from the environment, never from the payload."""
    import os
    from huggingface_hub import HfApi
    payload = json.loads(payload_json)
    api = HfApi(token=os.environ["HF_TOKEN"])
    device = payload.get("device") or pick_device()
    judge, judge_info = None, None
    if payload.get("judge_model"):
        from bloom.judge import make_hf_judge
        judge = make_hf_judge(payload["judge_model"], device=device, revision=payload.get("judge_revision") or None)
        judge_info = {"model_id": payload["judge_model"], "revision": payload.get("judge_revision") or None,
                      "weight": RewardConfig().w_judge, "scope": "open-ended prompts only"}
    settings = settings_from(**payload["settings"])
    return run_rl(api, payload["repo_dir"], payload["out_dir"], settings,
                  parent_run_dir=payload.get("parent_run_dir") or None,
                  parent_hf_repo=payload.get("parent_hf_repo") or None,
                  parent_hf_revision=payload.get("parent_hf_revision") or None,
                  device=device, judge=judge, judge_info=judge_info, use_gsm8k=bool(payload.get("use_gsm8k")))
