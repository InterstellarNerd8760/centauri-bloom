"""Helpers for the post-training notebooks: pick the newest parent, keep private data private."""
from __future__ import annotations

import json
from pathlib import Path

from bloom.notebook_workflow import parent_total

TEXT_REPO = "Centauri-Bloom/Bloom-Text-0.2"
CHAT_REPO = "Centauri-Bloom/Bloom-Chat-0.2"
# The only files a candidate PR may carry. Raw or tokenized training data must never be among them.
UPLOADABLE = frozenset({"model.safetensors", "config.json", "tokenizer.json", "metrics.json",
                        "data-provenance.json", "evaluation.json", "bloom-f32.gguf"})
# Local files that hold tokenized (hence decodable) training windows or resumable state.
PRIVATE_LOCAL = ("processed-corpus.npz", "partial-optimizer.pt", "partial.safetensors", "partial.json", "partial.tmp")


def _pr_numbers(api, repo):
    found = api.get_repo_discussions(repo, repo_type="model", discussion_type="pull_request",
                                     discussion_status="open")
    return sorted(int(d.num) for d in found)


def _read_metrics(api, repo, revision, token=None):
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo, "metrics.json", revision=revision, token=token)
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def candidate_revisions(api, repo, token=None):
    """Every revision of `repo` that has weights and metrics: open PRs, then main. Newest PR first."""
    out = []
    for num in reversed(_pr_numbers(api, repo)):
        rev = "refs/pr/%d" % num
        try:
            files = set(api.list_repo_files(repo, revision=rev))
        except Exception:
            continue
        metrics = _read_metrics(api, repo, rev, token) if {"model.safetensors", "metrics.json"} <= files else None
        if metrics:
            out.append({"revision": rev, "pr": num, "metrics": metrics})
    try:
        files = set(api.list_repo_files(repo))
        metrics = _read_metrics(api, repo, "main", token) if {"model.safetensors", "metrics.json"} <= files else None
        if metrics:
            out.append({"revision": "main", "pr": None, "metrics": metrics})
    except Exception:
        pass
    return out


def pretrain_tokens(metrics):
    """Cumulative non-padding training tokens reported by a Text-0.2 pretraining checkpoint, or None."""
    return parent_total(metrics)


def newest_pretrain(api, repo=TEXT_REPO, token=None, min_tokens=0):
    """Pick the pretraining checkpoint with the most cumulative tokens (ties: newest PR, then main).

    Returns (pinned_commit_sha, info). Raises if the best candidate has fewer than `min_tokens`,
    so a post-training run cannot silently start from a pretraining session that has not landed yet.
    """
    best = None
    for cand in candidate_revisions(api, repo, token):
        m = cand["metrics"]
        if m.get("training_track") in ("instruction-tuning", "rejection-sampling", "rejection-sampling-pilot"):
            continue
        tokens = pretrain_tokens(m)
        if tokens is None:
            continue
        rank = (tokens, cand["pr"] if cand["pr"] is not None else -1)
        if best is None or rank > best[0]:
            best = (rank, cand)
    if best is None:
        raise RuntimeError("No pretraining checkpoint with weights and token counts was found in %s." % repo)
    (tokens, _), cand = best
    if tokens < min_tokens:
        raise RuntimeError(
            "Newest pretrain checkpoint is %s with %s tokens, below the required %s. The latest pretraining session "
            "has probably not landed as a Hugging Face PR yet. Wait for it, or lower MIN_PARENT_TOKENS on purpose."
            % (cand["revision"], format(tokens, ","), format(min_tokens, ",")))
    sha = api.model_info(repo, revision=cand["revision"]).sha
    return sha, {"revision": cand["revision"], "tokens": tokens, "val_loss": cand["metrics"].get("val_loss"),
                 "pinned_sha": sha, "pr": cand["pr"]}


def newest_chat(api, repo=CHAT_REPO, token=None, dataset_prefix="Instruct mix v2", stage="sft"):
    """Newest open PR (else main) on the Chat repo from a given post-training stage. Returns (sha, info)."""
    for cand in candidate_revisions(api, repo, token):
        m = cand["metrics"]
        if m.get("stage", "sft") != stage or not str(m.get("dataset", "")).startswith(dataset_prefix):
            continue
        if m.get("candidate_uploadable") is False:
            continue
        sha = api.model_info(repo, revision=cand["revision"]).sha
        return sha, {"revision": cand["revision"], "pr": cand["pr"], "dataset": m.get("dataset"),
                     "val_loss": m.get("val_loss"), "pinned_sha": sha}
    raise RuntimeError("No %s checkpoint trained on '%s' was found in %s. Run the SFT notebook first, or set "
                       "PARENT_HF_REVISION / PARENT_RUN_DIR explicitly." % (stage, dataset_prefix, repo))


def scrub_private_artifacts(out_dir):
    """Delete tokenized-data caches and resume state from a finished run folder. Returns removed names.

    Kaggle saves everything under /kaggle/working with the notebook version. The processed corpus holds the
    token windows of the private Jaron set, so it must not stay in an output that could ever be shared.
    """
    removed = []
    for name in PRIVATE_LOCAL:
        path = Path(out_dir) / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def assert_upload_clean(names):
    """Raise unless every file name is one a candidate PR is allowed to carry."""
    extra = sorted(set(names) - UPLOADABLE)
    if extra:
        raise RuntimeError("Refusing to upload unexpected files: %s" % ", ".join(extra))
    return True


def provenance_is_private_safe(provenance):
    """True if a data-provenance dict has no raw example text (only counts, hashes and file names)."""
    text = json.dumps(provenance)
    return '"prompt"' not in text and '"response"' not in text
