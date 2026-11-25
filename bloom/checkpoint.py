"""Version 1 checkpoint exchange. Local validation only; no automatic acceptance."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import torch
from safetensors.torch import load, save

VERSION = 1
MAX_MANIFEST_BYTES = 16384
HEX = re.compile(r"[0-9a-f]{64}")
IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,128}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_hash(config) -> str:
    """Hash a dataclass config using sorted compact UTF-8 JSON."""
    return sha256(json.dumps(asdict(config), sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _digest(value):
    return isinstance(value, str) and HEX.fullmatch(value) is not None


def _positive_int(value):
    return type(value) is int and value > 0


@dataclass(frozen=True)
class RoundAssignment:
    round_id: str
    parent_sha256: str
    config_sha256: str
    dataset_sha256: str
    shard_id: str
    step_budget: int

    def __post_init__(self):
        for value in (self.round_id, self.shard_id):
            if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
                raise ValueError("invalid round or shard identifier")
        for value in (self.parent_sha256, self.config_sha256, self.dataset_sha256):
            if not _digest(value):
                raise ValueError("invalid SHA-256")
        if not _positive_int(self.step_budget):
            raise ValueError("invalid step budget")


def _read_bounded(path, limit):
    with Path(path).open("rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ValueError("artifact exceeds size limit")
    return data


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key")
        result[key] = value
    return result


def _check_state(state, reference):
    if set(state) != set(reference):
        raise ValueError("incompatible tensor keys")
    aliases = {}
    for key, expected in reference.items():
        actual = state[key]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"incompatible tensor schema: {key}")
        if not actual.is_floating_point() or not torch.isfinite(actual).all():
            raise ValueError(f"invalid tensor values: {key}")
        # A live model state_dict preserves tied parameters (e.g. tokens/head).
        # Reject conflicting values for exactly aliased tensors.
        alias = (expected.device, expected.data_ptr(), expected.shape,
                 expected.stride(), expected.dtype)
        if expected.numel() and alias in aliases:
            if not torch.equal(actual, state[aliases[alias]]):
                raise ValueError(f"inconsistent tied tensor: {key}")
        aliases[alias] = key


def _size_limit(reference):
    # Derive a local upper bound from trusted tensor sizes, not worker metadata.
    payload = sum(t.numel() * t.element_size() for t in reference.values())
    header = 4096 + sum(256 + 6 * len(k) + 32 * t.ndim
                        for k, t in reference.items())
    return payload + header


def save_weights(path, state, reference):
    """Save a full CPU snapshot; clone tied tensors for SafeTensors."""
    snapshot = {k: v.detach().cpu().contiguous().clone() for k, v in state.items()}
    _check_state(snapshot, reference)
    data = save(snapshot)
    if len(data) > _size_limit(reference):
        raise ValueError("checkpoint exceeds size limit")
    Path(path).write_bytes(data)
    return sha256(data)


def _decode_weights(data, reference):
    from safetensors import SafetensorError
    try:
        state = load(data)
    except (SafetensorError, ValueError, RuntimeError) as exc:
        raise ValueError("invalid SafeTensors checkpoint") from exc
    _check_state(state, reference)
    return state


def load_weights(path, expected_sha256, reference):
    """Read exactly the bytes verified against a trusted digest."""
    if not _digest(expected_sha256):
        raise ValueError("invalid expected SHA-256")
    data = _read_bounded(path, _size_limit(reference))
    if sha256(data) != expected_sha256:
        raise ValueError("checkpoint hash mismatch")
    return _decode_weights(data, reference)


def save_checkpoint(weights_path, manifest_path, state, reference,
                    assignment: RoundAssignment, steps: int):
    """Write worker full weights and a JSON sidecar. No optimizer state."""
    if not _positive_int(steps) or steps > assignment.step_budget:
        raise ValueError("steps exceed assignment budget or are invalid")
    if Path(weights_path).resolve() == Path(manifest_path).resolve():
        raise ValueError("weights and manifest paths must differ")
    digest = save_weights(weights_path, state, reference)
    manifest = {
        "version": VERSION, "kind": "full_weights",
        "assignment": asdict(assignment), "steps": steps,
        "weights_sha256": digest,
        "weights_bytes": Path(weights_path).stat().st_size,
    }
    Path(manifest_path).write_text(json.dumps(manifest, sort_keys=True, indent=2)
                                  + "\n", encoding="utf-8")
    return manifest


def validate_checkpoint(weights_path, manifest_path, expected: RoundAssignment,
                        reference):
    """Validate a submission against coordinator-owned assignment and schema."""
    manifest = json.loads(_read_bounded(manifest_path, MAX_MANIFEST_BYTES),
                          object_pairs_hook=_unique_pairs)
    fields = {"version", "kind", "assignment", "steps", "weights_sha256",
              "weights_bytes"}
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError("invalid manifest fields")
    if (type(manifest["version"]) is not int or manifest["version"] != VERSION
            or manifest["kind"] != "full_weights"):
        raise ValueError("unsupported checkpoint version or kind")
    raw = manifest["assignment"]
    if not isinstance(raw, dict) or set(raw) != set(asdict(expected)):
        raise ValueError("invalid assignment fields")
    if RoundAssignment(**raw) != expected:
        raise ValueError("assignment mismatch")
    steps, size = manifest["steps"], manifest["weights_bytes"]
    if not _positive_int(steps) or steps > expected.step_budget:
        raise ValueError("invalid reported steps")
    if not _positive_int(size) or size > _size_limit(reference):
        raise ValueError("invalid reported size")
    if not _digest(manifest["weights_sha256"]):
        raise ValueError("invalid checkpoint digest")
    data = _read_bounded(weights_path, size)
    if len(data) != size or sha256(data) != manifest["weights_sha256"]:
        raise ValueError("checkpoint size or hash mismatch")
    return _decode_weights(data, reference)
