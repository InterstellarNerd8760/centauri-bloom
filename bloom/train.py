"""Bounded, single-worker text training with saved SafeTensors artifacts."""
import argparse
from dataclasses import asdict
from itertools import islice
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import torch
from torch.nn import functional as F

from .checkpoint import config_hash, load_weights, save_weights
from .data import TextDataset, make_dataloader
from .model import Bloom, Config


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def evaluate(model, loader, device, batches):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in islice(loader, batches):
            x, y = x.to(device), y.to(device)
            value = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum")
            if not torch.isfinite(value):
                raise ValueError("non-finite validation loss")
            total += value.item()
            count += y.numel()
    if not count:
        raise ValueError("empty evaluation")
    return total / count, count


def run(directory, output, *, steps=100, batch_size=8, context_length=32,
        max_tokens=1_000_000, eval_batches=16, seed=42, device="cpu",
        learning_rate=0.001, cpu_threads=1, parent=None, parent_sha256=None,
        expected_manifest_sha256=None, shard_id=None):
    config = Config()
    for value in (steps, batch_size, context_length, max_tokens, eval_batches, cpu_threads):
        if type(value) is not int or value < 1:
            raise ValueError("budgets and dimensions must be positive integers")
    if context_length > config.context or batch_size > 256 or eval_batches > 256:
        raise ValueError("context, batch or evaluation budget exceeds baseline limits")
    if steps > 10000 or max_tokens > 10_000_000 or steps * batch_size * context_length > max_tokens:
        raise ValueError("requested training exceeds token or step budget")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.01:
        raise ValueError("invalid learning rate")
    if (parent is None) != (parent_sha256 is None):
        raise ValueError("parent path and trusted SHA-256 must be provided together")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested device is not available")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(seed)
    train = TextDataset(directory, "train", context_length,
                        expected_manifest_sha256=expected_manifest_sha256, shard_id=shard_id)
    valid = TextDataset(directory, "validation", context_length,
                        expected_manifest_sha256=train.manifest_sha256)
    train_loader = make_dataloader(train, batch_size, shuffle=True, seed=seed)
    val_loader = make_dataloader(valid, batch_size)
    model = Bloom(config)
    if parent is not None:
        model.load_state_dict(load_weights(parent, parent_sha256, model.state_dict()))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start = time.perf_counter()
    before, validation_tokens = evaluate(model, val_loader, device, eval_batches)
    model.train()
    iterator = iter(train_loader)
    seen = 0
    for step in range(1, steps + 1):
        batch = next(iterator, None)
        if batch is None:
            iterator = iter(train_loader)
            batch = next(iterator)
        x, y = (tensor.to(device) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        seen += y.numel()
        if step == 1 or step % 25 == 0 or step == steps:
            print(f"step {step}/{steps} | loss {loss.item():.4f} | bytes {seen}", flush=True)
    after, _ = evaluate(model, val_loader, device, eval_batches)
    elapsed = time.perf_counter() - start
    metrics = {
        "validation_loss_before": before, "validation_loss_after": after,
        "validation_target_tokens": validation_tokens, "train_target_tokens": seen,
        "steps": steps, "elapsed_seconds": elapsed,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    settings = {
        "steps": steps, "batch_size": batch_size, "context_length": context_length,
        "max_tokens": max_tokens, "eval_batches": eval_batches, "seed": seed,
        "learning_rate": learning_rate, "cpu_threads": cpu_threads,
        "optimizer": "AdamW", "gradient_clip_norm": 1.0, "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.machine(),
        "torch_version": str(torch.__version__), "python_version": platform.python_version(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        artifact = Path(tmp) / "artifact"
        artifact.mkdir()
        model.cpu()
        weights = artifact / "model.safetensors"
        digest = save_weights(weights, model.state_dict(), model.state_dict())
        load_weights(weights, digest, model.state_dict())
        manifest = {
            "version": 1, "kind": "experimental_text_baseline",
            "dataset_id": train.dataset_id, "dataset_version": train.dataset_version,
            "dataset_sha256": train.manifest_sha256, "train_shard_ids": list(train.shard_ids),
            "validation_shard_ids": list(valid.shard_ids), "config_sha256": config_hash(config),
            "weights_sha256": digest, "weights_bytes": weights.stat().st_size,
            "parent_sha256": parent_sha256, "training": settings,
        }
        write_json(artifact / "config.json", asdict(config))
        write_json(artifact / "tokenizer.json", {"type": "utf8-byte", "vocab_size": 256,
                                                "special_tokens": []})
        write_json(artifact / "manifest.json", manifest)
        write_json(artifact / "metrics.json", metrics)
        (artifact / "README.md").write_text(
            "# Experimental Bloom text baseline\n\n"
            "Small next-byte model; not a chatbot or an official release. "
            "See manifest.json, metrics.json and config.json for run details. "
            "Weights-only continuation is supported; optimizer state is not saved. "
            "Verify rights and provenance before redistribution.\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        artifact.rename(output)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--shard-id")
    args = parser.parse_args()
    try:
        print(json.dumps(run(**vars(args)), indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
