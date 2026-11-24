import math
import pytest
import torch
from bloom.model import Bloom, Config
from bloom.aggregate import average_updates
from bloom.smoke import run

def test_causal_mask():
    torch.manual_seed(1)
    model = Bloom(Config()).eval()
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[1, 2, 9, 8]])
    with torch.no_grad():
        torch.testing.assert_close(model(x)[:, :2], model(y)[:, :2])

def test_average():
    p = {"a": torch.tensor([1.0, 2.0])}
    a = {"a": torch.tensor([2.0, 4.0])}
    b = {"a": torch.tensor([0.0, 2.0])}
    torch.testing.assert_close(average_updates(p, [a, b])["a"], torch.tensor([2.0, 5.0]))

def test_reject_nonfinite():
    with pytest.raises(ValueError):
        average_updates({"a": torch.ones(1)}, [{"a": torch.tensor([float("nan")])}])

def test_smoke():
    result = run()
    assert result["workers"] == 2
    assert math.isfinite(result["held_out_loss_after"])
