import hashlib
import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save

from bloom.checkpoint import (RoundAssignment, config_hash, load_weights,
                              save_checkpoint, save_weights, validate_checkpoint)
from bloom.model import Bloom, Config


@pytest.fixture
def submission(tmp_path):
    config = Config()
    model = Bloom(config)
    reference = model.state_dict()
    parent = save_weights(tmp_path / "parent.safetensors", reference, reference)
    assignment = RoundAssignment("round-001", parent, config_hash(config),
                                 "a" * 64, "shard-01", 3)
    weights, manifest = tmp_path / "worker.safetensors", tmp_path / "worker.json"
    save_checkpoint(weights, manifest, reference, reference, assignment, 3)
    return weights, manifest, assignment, reference


def test_round_trip(submission):
    weights, manifest, assignment, reference = submission
    restored = validate_checkpoint(weights, manifest, assignment, reference)
    for key in reference:
        torch.testing.assert_close(restored[key], reference[key])
    target = Bloom(Config())
    target.load_state_dict(restored)
    assert target.head.weight is target.tokens.weight


def test_parent_hash(submission):
    weights, _, assignment, reference = submission
    parent = weights.with_name("parent.safetensors")
    load_weights(parent, assignment.parent_sha256, reference)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_weights(parent, "b" * 64, reference)


@pytest.mark.parametrize("field,value", [
    ("round_id", "round-002"), ("parent_sha256", "b" * 64),
    ("config_sha256", "b" * 64), ("dataset_sha256", "b" * 64),
    ("shard_id", "other"), ("step_budget", 4),
])
def test_assignment_mismatch(submission, field, value):
    weights, manifest, assignment, reference = submission
    with pytest.raises(ValueError, match="assignment mismatch"):
        validate_checkpoint(weights, manifest, replace(assignment, **{field: value}), reference)


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("version", True), ("steps", 4), ("steps", True),
    ("steps", 0), ("weights_bytes", -1), ("weights_bytes", 10**12),
    ("weights_sha256", "invalid"), ("kind", "pickle"),
])
def test_bad_manifest(submission, field, value):
    weights, manifest, assignment, reference = submission
    record = json.loads(manifest.read_text())
    record[field] = value
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        validate_checkpoint(weights, manifest, assignment, reference)


def test_duplicate_manifest_key(submission):
    weights, manifest, assignment, reference = submission
    manifest.write_text('{"version":1,"version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        validate_checkpoint(weights, manifest, assignment, reference)


@pytest.mark.parametrize("change", ["truncate", "append", "flip"])
def test_corrupt_artifact(submission, change):
    weights, manifest, assignment, reference = submission
    data = weights.read_bytes()
    weights.write_bytes({"truncate": data[:-1], "append": data + b"x",
                         "flip": data[:-1] + bytes([data[-1] ^ 1])}[change])
    with pytest.raises(ValueError):
        validate_checkpoint(weights, manifest, assignment, reference)


def rewrite(weights, manifest, data):
    weights.write_bytes(data)
    record = json.loads(manifest.read_text())
    record.update(weights_bytes=len(data), weights_sha256=hashlib.sha256(data).hexdigest())
    manifest.write_text(json.dumps(record))


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype", "nan", "inf", "tied"])
def test_reject_invalid_tensors_even_with_valid_hash(submission, change):
    weights, manifest, assignment, reference = submission
    state = {k: v.detach().clone() for k, v in reference.items()}
    key = "positions.weight"
    if change == "missing":
        del state[key]
    elif change == "extra":
        state["extra"] = torch.zeros(1)
    elif change == "shape":
        state[key] = state[key][:-1].contiguous()
    elif change == "dtype":
        state[key] = state[key].to(torch.int32)
    elif change in ("nan", "inf"):
        state[key][0, 0] = float(change)
    else:
        state["head.weight"][0, 0] += 1
    rewrite(weights, manifest, save(state))
    with pytest.raises(ValueError):
        validate_checkpoint(weights, manifest, assignment, reference)


def test_reject_invalid_safetensors(submission):
    weights, manifest, assignment, reference = submission
    rewrite(weights, manifest, b"not a safetensors file")
    with pytest.raises(ValueError):
        validate_checkpoint(weights, manifest, assignment, reference)


def test_reject_oversized_manifest(submission):
    weights, manifest, assignment, reference = submission
    manifest.write_bytes(b" " * 20000)
    with pytest.raises(ValueError, match="size limit"):
        validate_checkpoint(weights, manifest, assignment, reference)


def test_budget_and_config(submission):
    weights, manifest, assignment, reference = submission
    with pytest.raises(ValueError):
        save_checkpoint(weights, manifest, reference, reference, assignment, 4)
    assert config_hash(Config()) != config_hash(Config(context=64))
