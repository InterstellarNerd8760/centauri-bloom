"""Local, bounded CPU-only two-worker smoke test; no network access."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F
from .model import Bloom, Config
from .aggregate import average_updates
from .checkpoint import (RoundAssignment, config_hash, sha256, save_weights,
                         load_weights, save_checkpoint, validate_checkpoint)


def loss(model, batches):
    return F.cross_entropy(model(batches[:, :-1]).flatten(0, 1),
                           batches[:, 1:].reshape(-1))


def run():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = Config()
    base = Bloom(config)
    reference = base.state_dict()
    rng = torch.Generator().manual_seed(123)
    data = torch.randint(0, 16, (12, 17), generator=rng)
    held_out = data[8:]
    # Synthetic dataset identity, independent of PyTorch serialization.
    dataset_digest = sha256(json.dumps(data.tolist(), separators=(",", ":")).encode())
    updates = []
    before = float(loss(base, held_out).detach())
    with TemporaryDirectory() as directory:
        root = Path(directory)
        parent_path = root / "parent.safetensors"
        digest = save_weights(parent_path, reference, reference)
        parent = load_weights(parent_path, digest, reference)
        for i, shard in enumerate((data[:4], data[4:8])):
            assignment = RoundAssignment("smoke-001", digest, config_hash(config),
                                         dataset_digest, f"shard-{i}", 3)
            worker = copy.deepcopy(base)
            worker.load_state_dict(load_weights(parent_path, digest, reference))
            optimizer = torch.optim.AdamW(worker.parameters(), lr=0.001)
            for _ in range(3):
                optimizer.zero_grad()
                loss(worker, shard).backward()
                optimizer.step()
            weights = root / f"worker-{i}.safetensors"
            manifest = root / f"worker-{i}.json"
            save_checkpoint(weights, manifest, worker.state_dict(), reference,
                            assignment, 3)
            validated = validate_checkpoint(weights, manifest, assignment, reference)
            updates.append({k: v - parent[k] for k, v in validated.items()})
        base.load_state_dict(average_updates(parent, updates))
    return {"workers": 2, "steps_per_worker": 3,
            "parameters": sum(p.numel() for p in base.parameters()),
            "validated_checkpoints": len(updates),
            "held_out_loss_before": before,
            "held_out_loss_after": float(loss(base, held_out).detach())}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
