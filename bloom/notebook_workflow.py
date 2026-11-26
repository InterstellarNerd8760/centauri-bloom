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

def preflight(device="cpu"):
    # Import dependencies before the first lazy Dynamo import.
    import sympy
    import sympy.logic.boolalg
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA runtime before training.")
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

def auth_client(repo, token):
    from huggingface_hub import HfApi
    if not token or not token.strip():
        raise ValueError("An HF token is required for submitting a PR.")
    client = HfApi(token=token.strip())
    client.whoami()
    client.model_info(repo)
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

def submit_candidate(api, repo, out_dir, steps, revision, run_name):
    from huggingface_hub import CommitOperationAdd, constants
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    constants.HF_HUB_DISABLE_XET = True
    out = Path(out_dir)
    metrics = json.loads((out / "metrics.json").read_text())
    if metrics.get("additional_steps") != steps or metrics.get("parent_revision") != revision:
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
    if api.model_info(repo).sha != revision:
        raise RuntimeError("Main changed during training. Preserve the candidate; do not overwrite the newer checkpoint.")
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
