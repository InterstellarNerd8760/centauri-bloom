import ast
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from bloom.notebook_workflow import cached_revision, detect_environment, parent_total, pick_device, preflight, restore_corpus, save_corpus

def test_notebooks_compile():
    for path in Path("notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{path}:cell{i}")

def test_optimizer_preflight():
    preflight("cpu")

def test_corpus_roundtrip(tmp_path):
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([[-100, 3], [4, 5]])
    train = TensorDataset(x, y)
    path = tmp_path / "processed-corpus.npz"
    save_corpus(path, train, train, "parent", {"source": DataLoader(train)})
    assert cached_revision(path) == "parent"
    t, v, s = restore_corpus(path, "parent", 2, 8)
    assert torch.equal(t.tensors[0], x)
    assert torch.equal(v.tensors[1], y)
    assert torch.equal(s["source"].tensors[1], y)
    with pytest.raises(ValueError, match="different parent"):
        restore_corpus(path, "other", 2, 8)
    with pytest.raises(ValueError, match="outside vocabulary"):
        restore_corpus(path, "parent", 2, 4)

def test_unknown_cumulative_not_invented():
    assert parent_total({"step": 3000, "train_tokens": 49152000}) == 49152000
    assert parent_total({"step": 4500, "train_tokens": 24576000, "parent_reported_train_tokens": 49152000, "train_tokens_scope": "this continuation only"}) == 73728000
    assert parent_total({"step": 7500, "train_tokens": 1000, "train_tokens_scope": "this continuation only; non-padding input tokens"}) is None
    assert parent_total({"step": 7500, "cumulative_train_tokens": 98000000}) == 98000000

def test_pick_device_and_environment():
    device = pick_device()
    assert device in {"cuda", "mps", "cpu"}
    # CI and the local checkout are not Colab or Kaggle.
    assert detect_environment() == "local"
    preflight(device if device != "cuda" else "cpu")
