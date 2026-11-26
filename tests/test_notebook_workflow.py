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

def _write_run(path, additional_steps, parent_revision, target_revision, steps=None, step=None,
               dataset="Instruct mix (Jaron + open)", model_size="50M",
               target_repo="Centauri-Bloom/Bloom-Chat-0.2"):
    path.mkdir(parents=True, exist_ok=True)
    if steps is None:
        steps = additional_steps
    if step is None:
        step = additional_steps
    weights = path / ("step-%06d.safetensors" % steps)
    weights.write_bytes(b"weights")
    (path / "config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    provenance = path / "data-provenance.json"
    provenance.write_text("{}")
    evaluation = path / "evaluation.json"
    evaluation.write_text("{}")
    (path / "run-settings.json").write_text(json.dumps({
        "recipe": dataset, "steps": steps, "model_size": model_size,
        "parent_revision": parent_revision, "target_revision": target_revision,
    }))
    from bloom.notebook_workflow import sha256
    metrics = {
        "additional_steps": additional_steps,
        "step": step,
        "dataset": dataset,
        "model_size": model_size,
        "parent_revision": parent_revision,
        "target_revision": target_revision,
        "target_repo": target_repo,
        "run_name": path.name,
        "weights_sha256": sha256(weights),
        "data_provenance_sha256": sha256(provenance),
        "evaluation_sha256": sha256(evaluation),
    }
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics

def test_upload_base_revision_uses_chat_target_not_text_parent():
    from bloom.notebook_workflow import upload_base_revision
    parent = "66f77363cc174c895533c3be337437ba3a837866"
    target = "3f424e59516bc6aed588717f532f404c1a6e4b2a"
    assert upload_base_revision({"parent_revision": parent, "target_revision": target}) == target
    assert upload_base_revision({"parent_revision": parent, "target_revision": None}) == parent
    assert upload_base_revision({"parent_revision": parent}) == parent

def test_submit_candidate_accepts_hf_parent_chat_finetune(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from bloom.notebook_workflow import submit_candidate, upload_base_revision
    parent = "66f77363cc174c895533c3be337437ba3a837866"
    target = "3f424e59516bc6aed588717f532f404c1a6e4b2a"
    out = tmp_path / "bloom-50m-instruct-mix-(jaron-open)-20261126-213854"
    _write_run(out, additional_steps=3000, parent_revision=parent, target_revision=target, step=13000)
    # The old check compared parent_revision to the upload/target revision and failed.
    assert out.joinpath("metrics.json").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["parent_revision"] != metrics["target_revision"]
    assert metrics["additional_steps"] == 3000 and metrics["step"] == 13000
    assert upload_base_revision(metrics) == target

    class Api:
        def model_info(self, repo):
            return SimpleNamespace(sha=target)
        def create_commit(self, **kwargs):
            assert kwargs["parent_commit"] == target
            assert kwargs["create_pr"] is True
            return SimpleNamespace(commit_url="https://example/commit", pr_url="https://example/pr")

    monkeypatch.setattr("bloom.gguf_export.export_candidate", lambda *args, **kwargs: out / "bloom-f32.gguf")
    (out / "bloom-f32.gguf").write_bytes(b"gguf")
    info = submit_candidate(Api(), "Centauri-Bloom/Bloom-Chat-0.2", out, 3000, target, out.name)
    assert info["pr_url"] == "https://example/pr"
    assert (out / "upload-result.json").is_file()

def test_submit_candidate_rejects_step_mismatch(tmp_path):
    from bloom.notebook_workflow import submit_candidate
    parent = "66f77363cc174c895533c3be337437ba3a837866"
    target = "3f424e59516bc6aed588717f532f404c1a6e4b2a"
    out = tmp_path / "run"
    _write_run(out, additional_steps=3000, parent_revision=parent, target_revision=target, step=13000)
    try:
        submit_candidate(object(), "Centauri-Bloom/Bloom-Chat-0.2", out, 5000, target, "run")
    except ValueError as exc:
        assert "mismatched" in str(exc)
    else:
        raise AssertionError("expected mismatch")

def test_repair_additional_steps_from_settings_and_weights(tmp_path):
    from bloom.notebook_workflow import repair_completed_metrics
    parent = "66f77363cc174c895533c3be337437ba3a837866"
    target = "3f424e59516bc6aed588717f532f404c1a6e4b2a"
    out = tmp_path / "run"
    _write_run(out, additional_steps=13000, parent_revision=parent, target_revision=target,
               steps=3000, step=13000)
    # Weight file is named after the requested 3000 steps, not the cumulative 13000.
    assert (out / "step-003000.safetensors").is_file()
    assert not (out / "step-013000.safetensors").is_file()
    metrics = repair_completed_metrics(out, steps=3000)
    assert metrics["additional_steps"] == 3000
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["additional_steps"] == 3000
    assert saved["parent_revision"] == parent
    assert saved["step"] == 13000

def test_find_run_dir_nested_kaggle_style(tmp_path):
    from bloom.notebook_workflow import find_run_dir, prepare_recovered_run
    parent = "66f77363cc174c895533c3be337437ba3a837866"
    target = "3f424e59516bc6aed588717f532f404c1a6e4b2a"
    mount = tmp_path / "input" / "bloom-chat-0-2-sft-run1" / "centauri-bloom-runs"
    run = mount / "bloom-50m-instruct-mix-(jaron-open)-20261126-213854"
    _write_run(run, additional_steps=3000, parent_revision=parent, target_revision=target, step=13000)
    assert find_run_dir(tmp_path / "input" / "bloom-chat-0-2-sft-run1") == run
    assert find_run_dir(run) == run
    writable = tmp_path / "working"
    # Simulate a read-only input by removing write permission on the source.
    mode = run.stat().st_mode
    run.chmod(0o555)
    try:
        copied = prepare_recovered_run(tmp_path / "input" / "bloom-chat-0-2-sft-run1", writable)
    finally:
        run.chmod(mode)
    assert copied == writable / run.name
    assert (copied / "metrics.json").is_file()
    assert copied != run

def test_run_training_skips_completed_recover(tmp_path):
    from bloom.trainer import run_training
    parent = "66f77363cc174c895533c3be337437ba3a837866"
    target = "3f424e59516bc6aed588717f532f404c1a6e4b2a"
    out = tmp_path / "run"
    _write_run(out, additional_steps=3000, parent_revision=parent, target_revision=target, step=13000)
    class Api:
        token = "x"
    metrics = run_training(Api(), "Instruct mix (Jaron + open)", 3000, str(tmp_path), str(out),
                           recover=True, model_size="50M", device="cpu")
    assert metrics["additional_steps"] == 3000
    assert metrics["step"] == 13000
