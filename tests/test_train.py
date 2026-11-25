import hashlib
import json
from pathlib import Path

import pytest

from bloom.checkpoint import load_weights
from bloom.dataset import build_dataset
from bloom.model import Bloom, Config
from bloom.train import run


def test_saved_baseline(tmp_path):
    from bloom import dataset
    source = Path(__file__).parent / "fixtures/text/sources.json"
    data = tmp_path / "data"
    dataset.build_dataset(source, data)
    out = tmp_path / "model"
    metrics = run(data, out, steps=2, batch_size=2, eval_batches=2, max_tokens=128)
    manifest = json.loads((out / "manifest.json").read_text())
    assert metrics["steps"] == 2
    assert 0 < metrics["train_target_tokens"] <= 128
    assert metrics["validation_target_tokens"] > 0
    assert hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest() == manifest["weights_sha256"]
    model = Bloom(Config())
    model.load_state_dict(load_weights(out / "model.safetensors", manifest["weights_sha256"], model.state_dict()))
    with pytest.raises(FileExistsError):
        run(data, out, steps=1)
    with pytest.raises(ValueError, match="budget"):
        run(data, tmp_path / "too-big", steps=100, max_tokens=1)
    resumed = run(data, tmp_path / "resumed", steps=1, batch_size=2, max_tokens=64,
                  parent=out / "model.safetensors", parent_sha256=manifest["weights_sha256"])
    assert resumed["steps"] == 1
