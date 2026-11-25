"""Run a small single-worker CPU training demo on the built text dataset."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import TextDataset, make_dataloader
from .model import Bloom, Config


def loss(
    model: Bloom, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())


def run(
    directory: str | Path,
    *,
    context_length: int = 32,
    steps: int = 4,
    batch_size: int = 4,
    seed: int = 42,
) -> dict:
    config = Config()
    if not 1 <= context_length <= config.context:
        raise ValueError(f"context_length must be between 1 and {config.context}")
    if steps < 1:
        raise ValueError("steps must be positive")

    train = TextDataset(directory, "train", context_length)
    validation = TextDataset(
        directory, "validation", context_length,
        expected_manifest_sha256=train.manifest_sha256,
    )
    train_loader = make_dataloader(train, batch_size, shuffle=True, seed=seed)
    validation_batch = next(iter(make_dataloader(validation, batch_size)))

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = Bloom(config).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.eval()
    with torch.no_grad():
        before = loss(model, *validation_batch).item()

    model.train()
    batches = iter(train_loader)
    train_target_tokens = 0
    for _ in range(steps):
        batch = next(batches, None)
        if batch is None:
            batches = iter(train_loader)
            batch = next(batches)
        inputs, targets = batch
        optimizer.zero_grad()
        loss(model, inputs, targets).backward()
        optimizer.step()
        train_target_tokens += targets.numel()

    model.eval()
    with torch.no_grad():
        after = loss(model, *validation_batch).item()

    return {
        "dataset_id": train.dataset_id,
        "dataset_version": train.dataset_version,
        "dataset_manifest_sha256": train.manifest_sha256,
        "train_shard_ids": train.shard_ids,
        "validation_shard_ids": validation.shard_ids,
        "seed": seed,
        "context_length": context_length,
        "batch_size": batch_size,
        "steps": steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_target_tokens": train_target_tokens,
        "validation_target_tokens": validation_batch[1].numel(),
        "validation_loss_before": before,
        "validation_loss_after": after,
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing manifest.json")
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = run(**vars(args))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
