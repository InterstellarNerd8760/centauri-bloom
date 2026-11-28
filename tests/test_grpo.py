import json
import string

import pytest
import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

from bloom.checkpoint import config_hash, save_weights
from bloom.grpo import (
    RLSettings, assign_weights, best_of_n_weights, collate, group_advantages, policy_loss_step, rollout,
    settings_from, train_policy,
)
from bloom.model import Bloom, Config
from bloom.notebook_workflow import sha256
from bloom.rewards import RewardConfig, Task, TaskSampler, build_holdout, holdout_norms
from bloom.rft import load_locked_eval

EOT = "<|endoftext|>"


def char_tokenizer():
    chars = list(string.printable) + [EOT]
    vocab = {c: i for i, c in enumerate(dict.fromkeys(chars))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token=" "))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    tok.decoder = decoders.Fuse()
    return tok


def tiny_model(tok, context=200, seed=0):
    torch.manual_seed(seed)
    return Bloom(Config(vocab_size=tok.get_vocab_size(), context=context, width=32, heads=4, layers=2))


def test_group_advantages():
    adv = group_advantages([1.0, 0.0, 0.0, 0.0])
    assert abs(sum(adv)) < 1e-6 and adv[0] > 0 > adv[1]
    assert group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]
    raw = group_advantages([2.0, 0.0], std_normalize=False)
    assert raw == [1.0, -1.0]


def test_best_of_n_keeps_only_clear_winners():
    assert best_of_n_weights([0.2, 1.3, 0.1], 0.9) == [0.0, 1.0, 0.0]
    assert best_of_n_weights([0.2, 0.5, 0.1], 0.9) == [0.0, 0.0, 0.0]
    assert best_of_n_weights([1.0, 1.0, 1.0], 0.9) == [0.0, 0.0, 0.0]


def test_collate_masks_prompt_and_pads_right():
    samples = [{"prefix": [1, 2, 3], "ids": [4, 5, 0], "weight": 1.0},
               {"prefix": [1, 2], "ids": [6, 0], "weight": -1.0}]
    x, y, mask, w = collate(samples, eot=0, device="cpu")
    assert x.shape == (2, 5)
    assert x[0].tolist() == [1, 2, 3, 4, 5] and y[0].tolist() == [2, 3, 4, 5, 0]
    assert mask[0].tolist() == [0, 0, 1, 1, 1]
    assert x[1].tolist()[:3] == [1, 2, 6] and mask[1].tolist() == [0, 1, 1, 0, 0]
    assert w.tolist() == [1.0, -1.0]


def test_policy_gradient_raises_good_and_lowers_bad_logprob():
    tok = char_tokenizer()
    model = tiny_model(tok)
    eot = tok.token_to_id(EOT)
    good = {"prefix": tok.encode("User: hi\nAssistant:").ids, "ids": tok.encode(" ok").ids + [eot], "weight": 1.0}
    bad = {"prefix": good["prefix"], "ids": tok.encode(" no").ids + [eot], "weight": -1.0}
    settings = RLSettings(beta=0.0, micro_batch=2)
    from bloom.grpo import token_logprobs

    def lp(sample):
        x, y, mask, _ = collate([sample], eot, "cpu")
        with torch.no_grad():
            return float((token_logprobs(model, x, y, 1.0, False) * mask).sum())

    before = (lp(good), lp(bad))
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for _ in range(5):
        opt.zero_grad()
        policy_loss_step(model, None, [good, bad], settings, eot, "cpu", scaler)
        opt.step()
    assert lp(good) > before[0] and lp(bad) < before[1]


def test_rollout_scores_every_sample_and_assigns_weights():
    tok = char_tokenizer()
    model = tiny_model(tok)
    eot = tok.token_to_id(EOT)
    tasks = [Task("arithmetic", "What is 2 plus 2?", "4"), Task("open", "Say hi."), Task("even_odd", "Is 3 even or odd?", "odd")]
    settings = RLSettings(group_size=4, max_new=10)
    groups = rollout(model, tok, tasks, settings, eot, "cpu", RewardConfig(), None, seed=1)
    assert len(groups) == 3 and all(len(g) == 4 for g in groups)
    for g in groups:
        for s in g:
            assert len(s["ids"]) <= 10 and "reward" in s["scored"]
    active = assign_weights(groups, settings)
    assert 0 <= active <= 3
    for g in groups:
        ws = [s["weight"] for s in g]
        assert abs(sum(ws)) < 1e-4  # grpo advantages are centred
    assign_weights(groups, RLSettings(algo="best-of-n"))
    for g in groups:
        assert sum(1 for s in g if s["weight"]) <= 1


def test_settings_validation():
    assert settings_from(steps=5, algo="best-of-n").algo == "best-of-n"
    for bad in ({"algo": "ppo"}, {"steps": 0}, {"group_size": 1}, {"temperature": 5.0}):
        with pytest.raises(ValueError):
            settings_from(**bad)


@pytest.mark.parametrize("algo", ["grpo", "best-of-n"])
def test_train_policy_runs_end_to_end_on_cpu(algo, tmp_path):
    tok = char_tokenizer()
    model = tiny_model(tok, context=200)
    eot = tok.token_to_id(EOT)
    start = {k: v.clone() for k, v in model.state_dict().items()}
    settings = RLSettings(algo=algo, steps=3, prompts_per_step=4, group_size=4, max_new=12, eval_every=3,
                          holdout_per_type=1, lr=1e-3, warmup_steps=1, bon_min_reward=-10, bon_min_reward_open=-10)
    holdout = build_holdout(1)
    sampler = TaskSampler(1, excluded_norms=holdout_norms(holdout))
    logs, seen = [], []
    result = train_policy(model, tok, eot, "cpu", settings, RewardConfig(), sampler, holdout, load_locked_eval(),
                          None, log=lambda *a: logs.append(" ".join(map(str, a))), on_eval=lambda s, e: seen.append(s),
                          log_path=str(tmp_path / "log.jsonl"))
    assert result["steps_run"] == 3 and seen == [0, 3]
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 3
    assert any(not torch.equal(start[k], v) for k, v in model.state_dict().items())
    for row in result["history"]:
        assert row["groups"] == 4 and row["mean_tokens"] > 0
    ev = result["evals"][0]
    assert 0.0 <= ev["composite"] <= 1.0 and ev["locked"]["n_items"] == 210


def test_early_stop_on_kl(tmp_path):
    tok = char_tokenizer()
    model = tiny_model(tok)
    eot = tok.token_to_id(EOT)
    settings = RLSettings(steps=40, prompts_per_step=4, group_size=4, max_new=8, eval_every=100, holdout_per_type=1,
                          lr=0.05, warmup_steps=1, kl_stop=1e-6, beta=0.5)
    holdout = build_holdout(1)
    result = train_policy(model, tok, eot, "cpu", settings, RewardConfig(), TaskSampler(2, excluded_norms=holdout_norms(holdout)),
                          holdout, load_locked_eval(), None, log=lambda *a: None)
    assert result["stopped"] and result["steps_run"] < 40


def _patch_parent(monkeypatch, parent, pm, cfg):
    """The real loader pins the 50M shape; swap in one that checks hashes only, for the tiny test model."""
    from bloom import trainer
    monkeypatch.setattr(trainer, "PARAM_COUNT_50M", pm["parameters"])
    monkeypatch.setattr(trainer, "load_local_parent",
                        lambda *a, **k: (parent, cfg, json.loads((parent / "metrics.json").read_text()),
                                         parent / "step-000010.safetensors", "local:parent"))


def _write_parent(tmp_path, tok, model):
    parent = tmp_path / "sft-run"
    parent.mkdir()
    cfg = model.config
    digest = save_weights(parent / "step-000010.safetensors", model.state_dict(), model.state_dict())
    (parent / "config.json").write_text(json.dumps(cfg.__dict__, sort_keys=True))
    tok.save(str(parent / "tokenizer.json"))
    metrics = {"additional_steps": 10, "step": 10, "weights_sha256": digest, "config_sha256": config_hash(cfg),
               "target_repo": "Centauri-Bloom/Bloom-Chat-0.2", "dataset": "Instruct mix v2 (Jaron v2 + open)",
               "stage": "sft", "parameters": sum(p.numel() for p in model.parameters())}
    (parent / "metrics.json").write_text(json.dumps(metrics))
    return parent, metrics


class FakeApi:
    token = "x"

    class _Info:
        sha = "a" * 40

    def model_info(self, *a, **k):
        return self._Info()


def test_run_rl_writes_a_candidate_in_the_upload_layout(tmp_path, monkeypatch):
    from bloom import rl_run, trainer
    tok = char_tokenizer()
    model = tiny_model(tok)
    parent, pm = _write_parent(tmp_path, tok, model)
    _patch_parent(monkeypatch, parent, pm, model.config)
    calls = iter(range(100))
    monkeypatch.setattr("bloom.policy_eval.composite", lambda a, b: next(calls) * 0.1)  # force "improvement"
    settings = RLSettings(steps=2, prompts_per_step=2, group_size=4, max_new=8, eval_every=1, holdout_per_type=1,
                          warmup_steps=1, min_gain=0.05)
    out = tmp_path / "rl-run"
    metrics = rl_run.run_rl(FakeApi(), ".", out, settings, parent_run_dir=parent, device="cpu", log=lambda *a: None)
    assert metrics["candidate_uploadable"] and metrics["additional_steps"] == 2 and metrics["stage"] == "rl"
    weights = out / "step-000002.safetensors"
    assert weights.is_file() and sha256(weights) == metrics["weights_sha256"]
    assert not (out / "best.safetensors").exists()
    for name, key in (("data-provenance.json", "data_provenance_sha256"), ("evaluation.json", "evaluation_sha256")):
        assert sha256(out / name) == metrics[key]
    assert metrics["target_revision"] == "a" * 40 and metrics["target_repo"] == "Centauri-Bloom/Bloom-Chat-0.2"
    prov = json.loads((out / "data-provenance.json").read_text())
    assert prov["private_data"] == "none"
    from bloom.posttrain import provenance_is_private_safe
    assert provenance_is_private_safe(prov)
    # The parent run folder is untouched.
    assert sorted(p.name for p in parent.iterdir()) == ["config.json", "metrics.json", "step-000010.safetensors", "tokenizer.json"]


def test_run_rl_without_gain_writes_no_weights(tmp_path, monkeypatch):
    from bloom import rl_run, trainer
    tok = char_tokenizer()
    model = tiny_model(tok)
    parent, pm = _write_parent(tmp_path, tok, model)
    _patch_parent(monkeypatch, parent, pm, model.config)
    monkeypatch.setattr("bloom.policy_eval.composite", lambda a, b: 0.5)
    settings = RLSettings(steps=1, prompts_per_step=2, group_size=2, max_new=8, eval_every=1, holdout_per_type=1)
    out = tmp_path / "rl-run"
    metrics = rl_run.run_rl(FakeApi(), ".", out, settings, parent_run_dir=parent, device="cpu", log=lambda *a: None)
    assert metrics["candidate_uploadable"] is False
    assert not list(out.glob("*.safetensors"))


def test_run_rl_refuses_text_parent_and_nonempty_out(tmp_path, monkeypatch):
    from bloom import rl_run, trainer
    tok = char_tokenizer()
    model = tiny_model(tok)
    parent, pm = _write_parent(tmp_path, tok, model)
    _patch_parent(monkeypatch, parent, pm, model.config)
    pm["target_repo"], pm["dataset"] = "Centauri-Bloom/Bloom-Text-0.2", "FineWeb-Edu"
    (parent / "metrics.json").write_text(json.dumps(pm))
    with pytest.raises(ValueError, match="Chat"):
        rl_run.run_rl(FakeApi(), ".", tmp_path / "o1", RLSettings(steps=1), parent_run_dir=parent, device="cpu", log=lambda *a: None)
    busy = tmp_path / "o2"
    busy.mkdir()
    (busy / "x").write_text("1")
    with pytest.raises(RuntimeError, match="not empty"):
        rl_run.run_rl(FakeApi(), ".", busy, RLSettings(steps=1), parent_run_dir=parent, device="cpu", log=lambda *a: None)
