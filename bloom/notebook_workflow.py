"""Shared continuation notebook utilities. No credentials or raw datasets are published."""
import hashlib
import json
import os
from pathlib import Path

def sha256(path):
    with Path(path).open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()

def pick_device():
    """Prefer CUDA, then MPS, then CPU. fp16 autocast is applied only on CUDA."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"

def detect_environment():
    """Return 'kaggle', 'colab', or 'local' without importing Bloom training code."""
    if Path("/kaggle/working").is_dir():
        return "kaggle"
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return "local"
    return "colab"

def preflight(device="cpu"):
    # Import dependencies before the first lazy Dynamo import.
    import sympy
    import sympy.logic.boolalg
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA runtime before training.")
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS is not available.")
    try:
        p = torch.nn.Parameter(torch.ones(2, device=device))
        opt = torch.optim.AdamW([p], lr=1e-4)
        (p.square().sum()).backward()
        opt.step()
        if not torch.isfinite(p).all():
            raise RuntimeError("Nonfinite optimizer smoke test")
    except Exception as exc:
        raise RuntimeError("Optimizer preflight failed. Restart the Python session before processing data; do not reinstall packages in a live training session.") from exc
    print("Optimizer ready:", torch.__version__, "| SymPy:", sympy.__version__, "|", device)

def parent_total(metrics):
    n = metrics.get("cumulative_train_tokens")
    if type(n) is int and n >= 0:
        return n
    scope = str(metrics.get("train_tokens_scope", ""))
    n = metrics.get("train_tokens")
    if type(n) is not int or n < 0:
        return None
    if metrics.get("step") == 3000 and not scope.startswith("this continuation"):
        return n
    prev = metrics.get("parent_reported_train_tokens")
    if metrics.get("step") == 4500 and scope.startswith("this continuation") and prev == 49152000 and n == 24576000:
        return prev + n
    return None

def auth_client(repo, token, require_repo=True):
    from huggingface_hub import HfApi
    if not token or not token.strip():
        raise ValueError("An HF token is required for submitting a PR.")
    client = HfApi(token=token.strip())
    client.whoami()
    try:
        client.model_info(repo)
    except Exception:
        if require_repo:
            raise
        print("Target repo not found yet:", repo,
              "| Training can proceed. Create the repo on Hugging Face before the upload cell. This trainer will not create it.")
    return client

def save_corpus(path, train_ds, val_ds, revision, source_loaders=None):
    import numpy as np
    path = Path(path)
    arrays = {"train_x": train_ds.tensors[0].numpy(), "train_y": train_ds.tensors[1].numpy(),
              "val_x": val_ds.tensors[0].numpy(), "val_y": val_ds.tensors[1].numpy(),
              "parent_revision": np.array(revision)}
    for i, (name, loader) in enumerate((source_loaders or {}).items()):
        arrays[f"source_{i}_name"] = np.array(name)
        arrays[f"source_{i}_x"] = loader.dataset.tensors[0].numpy()
        arrays[f"source_{i}_y"] = loader.dataset.tensors[1].numpy()
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)
    print("Processed corpus saved:", path)

def cached_revision(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return str(data["parent_revision"].item())

def restore_corpus(path, revision, context, vocab_size):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    with np.load(path, allow_pickle=False) as data:
        if str(data["parent_revision"].item()) != revision:
            raise ValueError("Cached corpus belongs to a different parent.")
        def ds(prefix):
            x, y = data[prefix + "_x"], data[prefix + "_y"]
            if x.shape != y.shape or x.ndim != 2 or x.shape[1] != context or not len(x):
                raise ValueError("Invalid cached window shape")
            if x.dtype.kind not in "iu" or y.dtype.kind not in "iu":
                raise ValueError("Invalid cached token type")
            if x.min() < 0 or x.max() >= vocab_size or ((y != -100) & ((y < 0) | (y >= vocab_size))).any():
                raise ValueError("Cached token IDs outside vocabulary")
            return TensorDataset(torch.from_numpy(x.astype("int64", copy=False)), torch.from_numpy(y.astype("int64", copy=False)))
        sources = {}
        i = 0
        while f"source_{i}_name" in data:
            sources[str(data[f"source_{i}_name"].item())] = ds(f"source_{i}")
            i += 1
        return ds("train"), ds("val"), sources

def upload_base_revision(metrics):
    """Return the target-repo revision a candidate PR should be based on.

    For instruction fine-tunes the text parent and the Chat upload repo differ, so
    `target_revision` (Chat/main at train time) is the PR base. `parent_revision`
    remains the pretrained checkpoint SHA.
    """
    target = metrics.get("target_revision")
    if target:
        return target
    return metrics.get("parent_revision")

def _is_run_dir(path):
    path = Path(path)
    if not path.is_dir() or not (path / "metrics.json").is_file():
        return False
    if not (path / "tokenizer.json").is_file() or not (path / "config.json").is_file():
        return False
    return bool(list(path.glob("step-*.safetensors")) or (path / "model.safetensors").is_file()
                or (path / "partial.safetensors").is_file())

def _collect_run_dirs(root, seen, matches):
    if not root.is_dir():
        return
    for metrics_path in root.rglob("metrics.json"):
        cand = metrics_path.parent
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen or not _is_run_dir(cand):
            continue
        seen.add(key)
        matches.append(cand)

def _pick_run_dir(path, matches):
    if path.name:
        named = [m for m in matches if m.name == path.name or path.name in m.parts]
        if len(named) == 1:
            return named[0]
        if named:
            matches = named
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    raise ValueError("Several completed runs were found; set RECOVER_RUN_DIR to one of: "
                     + ", ".join(str(m) for m in matches))

def find_run_dir(path):
    """Resolve a completed run folder, searching nested Kaggle input mounts if needed."""
    path = Path(path).expanduser()
    if _is_run_dir(path):
        return path
    seen, matches = set(), []
    if path.exists():
        _collect_run_dirs(path, seen, matches)
        picked = _pick_run_dir(path, matches)
        if picked is not None:
            return picked
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        _collect_run_dirs(kaggle_input, seen, matches)
        picked = _pick_run_dir(path, matches)
        if picked is not None:
            return picked
    raise FileNotFoundError(
        "No completed Bloom run was found at %s. Point RECOVER_RUN_DIR at the run folder or its Kaggle input parent."
        % path
    )

def _is_writable(path):
    path = Path(path)
    try:
        resolved = path.resolve()
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.is_dir() and (kaggle_input in resolved.parents or resolved == kaggle_input.resolve()):
            return False
        if not os.access(path, os.W_OK):
            return False
        probe = path / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False

def prepare_recovered_run(path, writable_root):
    """Find a completed run and copy it to a writable folder when the source is read-only."""
    import shutil
    src = find_run_dir(path)
    writable_root = Path(writable_root)
    writable_root.mkdir(parents=True, exist_ok=True)
    if _is_writable(src):
        return src
    dest = writable_root / src.name
    if dest.resolve() != src.resolve():
        if not dest.exists():
            shutil.copytree(src, dest)
        elif not _is_run_dir(dest):
            raise RuntimeError("Writable copy %s exists but is not a complete run folder." % dest)
    return dest

def repair_completed_metrics(out_dir, steps=None):
    """Fix additional_steps when a continuation recorded the cumulative step count."""
    out = Path(out_dir)
    metrics = json.loads((out / "metrics.json").read_text())
    settings = {}
    if (out / "run-settings.json").is_file():
        settings = json.loads((out / "run-settings.json").read_text())
    requested = int(steps) if steps is not None else (int(settings["steps"]) if settings.get("steps") is not None else None)

    def has_weights(n):
        return (out / ("step-%06d.safetensors" % int(n))).is_file()

    additional = metrics.get("additional_steps")
    repaired = False
    if requested is not None and additional != requested and has_weights(requested):
        metrics = dict(metrics)
        metrics["additional_steps"] = requested
        repaired = True
    elif (requested is None and settings.get("steps") is not None
          and additional != settings["steps"] and has_weights(settings["steps"])):
        metrics = dict(metrics)
        metrics["additional_steps"] = int(settings["steps"])
        repaired = True
    if repaired:
        tmp = out / "metrics.json.tmp"
        tmp.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
        tmp.replace(out / "metrics.json")
        print("Repaired additional_steps to", metrics["additional_steps"], "in", out, flush=True)
    return metrics

def submit_candidate(api, repo, out_dir, steps, revision, run_name):
    from huggingface_hub import CommitOperationAdd, constants
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    constants.HF_HUB_DISABLE_XET = True
    out = Path(out_dir)
    if not (out / "metrics.json").is_file():
        out = find_run_dir(out)
    if not _is_writable(out):
        working = Path("/kaggle/working/centauri-bloom-runs") if Path("/kaggle/working").is_dir() else out.parent
        out = prepare_recovered_run(out, working)
    metrics = repair_completed_metrics(out, steps=steps)
    expected = upload_base_revision(metrics)
    if metrics.get("additional_steps") != steps or expected != revision:
        raise ValueError("Incomplete or mismatched training run")
    weights = out / f"step-{steps:06d}.safetensors"
    paths = {"model.safetensors": weights, **{n: out / n for n in
             ("config.json", "tokenizer.json", "metrics.json", "data-provenance.json", "evaluation.json")}}
    for name, key in (("model.safetensors", "weights_sha256"),
                      ("data-provenance.json", "data_provenance_sha256"),
                      ("evaluation.json", "evaluation_sha256")):
        if sha256(paths[name]) != metrics.get(key):
            raise ValueError("Hash mismatch: " + name)
    receipt = out / "upload-result.json"
    if receipt.exists():
        print("Already submitted:", receipt.read_text())
        return json.loads(receipt.read_text())
    try:
        current = api.model_info(repo).sha
    except Exception as exc:
        raise RuntimeError(
            "Target repo %s does not exist. Create it on Hugging Face (an empty model repo is enough), then retry this upload. This trainer will not create or overwrite repos." % repo
        ) from exc
    if revision:
        if current != revision:
            raise RuntimeError("Main changed during training. Preserve the candidate; do not overwrite the newer checkpoint.")
    else:
        # First from-scratch candidate; the repo existed only at upload time.
        revision = current
    # Export the exact candidate, not a stale GGUF from the parent revision.
    # Do this only after all checkpoint/provenance checks; no training is rerun.
    from bloom.gguf_export import GGUF_NAME, export_candidate
    paths[GGUF_NAME] = export_candidate(out, steps, metrics)
    operations = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p)) for n, p in paths.items()]
    try:
        result = api.create_commit(repo_id=repo, operations=operations, revision="main",
                                   parent_commit=revision, create_pr=True,
                                   commit_message=f"Bloom continuation {run_name}")
    except Exception:
        print("Upload failed; local candidate is safe:", out)
        print("Check token permissions and existing HF discussions before retrying. Do not retrain.")
        raise
    info = {"commit_url": str(result.commit_url), "pr_url": str(result.pr_url), "published_to_main": False}
    receipt.write_text(json.dumps(info, indent=2))
    print("UPLOAD COMPLETE:", info["pr_url"])
    return info
