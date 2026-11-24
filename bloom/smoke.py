"""Local, bounded CPU-only two-worker smoke test; no network access."""
import copy
import json
import torch
from torch.nn import functional as F
from .model import Bloom, Config
from .aggregate import average_updates

def loss(model, batches):
    return F.cross_entropy(model(batches[:, :-1]).flatten(0, 1),
                           batches[:, 1:].reshape(-1))

def run():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = Config()
    base = Bloom(config)
    parent = {k: v.detach().clone() for k, v in base.state_dict().items()}
    rng = torch.Generator().manual_seed(123)
    data = torch.randint(0, 16, (12, 17), generator=rng)
    held_out = data[8:]
    updates = []
    before = float(loss(base, held_out).detach())
    for shard in (data[:4], data[4:8]):
        worker = copy.deepcopy(base)
        optimizer = torch.optim.AdamW(worker.parameters(), lr=0.001)
        for _ in range(3):
            optimizer.zero_grad()
            loss(worker, shard).backward()
            optimizer.step()
        updates.append({k: v.detach() - parent[k] for k, v in worker.state_dict().items()})
    base.load_state_dict(average_updates(parent, updates))
    return {"workers": 2, "steps_per_worker": 3,
            "parameters": sum(p.numel() for p in base.parameters()),
            "held_out_loss_before": before,
            "held_out_loss_after": float(loss(base, held_out).detach())}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
