"""In-memory same-parent averaging baseline, not a public upload service."""
import torch

def average_updates(parent, updates):
    if not updates:
        raise ValueError("no updates")
    result = {}
    for update in updates:
        if set(update) != set(parent):
            raise ValueError("incompatible keys")
    for key, original in parent.items():
        if not original.is_floating_point() or not torch.isfinite(original).all():
            raise ValueError("parent must be finite floating point")
        deltas = []
        for update in updates:
            delta = update[key]
            if delta.shape != original.shape or delta.dtype != original.dtype:
                raise ValueError("incompatible tensor")
            if delta.device != original.device or not torch.isfinite(delta).all():
                raise ValueError("invalid update")
            deltas.append(delta.to(torch.float64))
        combined = (original.to(torch.float64) + torch.stack(deltas).mean(0)).to(original.dtype)
        if not torch.isfinite(combined).all():
            raise ValueError("nonfinite aggregate")
        result[key] = combined
    return result
