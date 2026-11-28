import json
from types import SimpleNamespace

import pytest

from bloom import posttrain
from bloom.posttrain import (
    UPLOADABLE, assert_upload_clean, newest_chat, newest_pretrain, provenance_is_private_safe,
    scrub_private_artifacts,
)


class FakeApi:
    """Just enough of HfApi: PRs with metrics, plus main."""

    def __init__(self, prs, main=None):
        self.prs = prs      # {num: metrics}
        self.main = main    # metrics or None

    def get_repo_discussions(self, repo, **kw):
        return [SimpleNamespace(num=n) for n in self.prs]

    def list_repo_files(self, repo, revision="main"):
        if revision == "main":
            return ["model.safetensors", "metrics.json"] if self.main else ["README.md"]
        return ["model.safetensors", "metrics.json"]

    def model_info(self, repo, revision="main"):
        return SimpleNamespace(sha="sha-" + revision.replace("/", "-"))


@pytest.fixture
def api_with(monkeypatch):
    def make(prs, main=None):
        api = FakeApi(prs, main)

        def fake_read(_api, repo, revision, token=None):
            return api.main if revision == "main" else api.prs[int(revision.rsplit("/", 1)[1])]
        monkeypatch.setattr(posttrain, "_read_metrics", fake_read)
        return api
    return make


def pre(tokens, **extra):
    return {"cumulative_train_tokens": tokens, "training_track": "pretraining", "val_loss": 3.0, **extra}


def test_newest_pretrain_picks_most_tokens_and_pins_the_sha(api_with):
    api = api_with({5: pre(819_200_000), 6: pre(983_040_000), 7: pre(1_146_880_000)})
    sha, info = newest_pretrain(api, min_tokens=1_100_000_000)
    assert info["revision"] == "refs/pr/7" and sha == "sha-refs-pr-7" and info["tokens"] == 1_146_880_000


def test_newest_pretrain_refuses_until_the_latest_session_has_landed(api_with):
    api = api_with({5: pre(819_200_000), 6: pre(983_040_000)})
    with pytest.raises(RuntimeError, match="has probably not landed"):
        newest_pretrain(api, min_tokens=1_100_000_000)
    assert newest_pretrain(api)[1]["revision"] == "refs/pr/6"  # explicit opt-out still works


def test_newest_pretrain_ignores_instruction_checkpoints_and_uses_main_if_larger(api_with):
    chat = {"cumulative_train_tokens": 9_999_999_999, "training_track": "instruction-tuning"}
    api = api_with({3: pre(100), 4: chat}, main=pre(5000))
    assert newest_pretrain(api)[1]["revision"] == "main"


def test_newest_chat_prefers_newest_matching_sft_and_skips_non_uploadable(api_with):
    sft = {"stage": "sft", "dataset": "Instruct mix v2 (Jaron v2 + open)"}
    rl = {"stage": "rl", "dataset": "generated"}
    old_mix = {"stage": "sft", "dataset": "Instruct mix (Jaron + open)"}
    api = api_with({1: sft, 2: old_mix, 3: rl, 4: {**sft, "candidate_uploadable": False}})
    sha, info = newest_chat(api)
    assert info["pr"] == 1 and sha == "sha-refs-pr-1"
    with pytest.raises(RuntimeError, match="No sft checkpoint"):
        newest_chat(api_with({3: rl}))


def test_scrub_removes_tokenized_corpus_and_resume_state(tmp_path):
    for name in ("processed-corpus.npz", "partial-optimizer.pt", "partial.safetensors", "metrics.json",
                 "step-003000.safetensors"):
        (tmp_path / name).write_bytes(b"x")
    removed = scrub_private_artifacts(tmp_path)
    assert set(removed) == {"processed-corpus.npz", "partial-optimizer.pt", "partial.safetensors"}
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "step-003000.safetensors").exists()


def test_upload_guard_allows_only_the_candidate_files():
    assert assert_upload_clean(sorted(UPLOADABLE))
    for bad in ("processed-corpus.npz", "bloom_instruction_expanded_v2.json", "rft-generations.json", "run-settings.json"):
        with pytest.raises(RuntimeError):
            assert_upload_clean(["model.safetensors", bad])


def test_submit_candidate_file_list_is_within_the_upload_guard():
    import inspect
    from bloom import notebook_workflow
    src = inspect.getsource(notebook_workflow.submit_candidate)
    names = {"model.safetensors", "config.json", "tokenizer.json", "metrics.json", "data-provenance.json",
             "evaluation.json"}
    for name in names:
        assert '"%s"' % name in src
    assert "create_pr=True" in src and 'revision="main"' in src
    assert names | {"bloom-f32.gguf"} == set(UPLOADABLE)
    assert "upload_file(" not in src and "upload_folder(" not in src


def test_provenance_privacy_check():
    assert provenance_is_private_safe({"sources": [{"name": "Jaron", "source_sha256": "ab", "source_count": 3}]})
    assert not provenance_is_private_safe({"rows": [{"prompt": "hi", "response": "yo"}]})


def test_sft_run_records_locked_eval_and_targets_a_chat_pr(tmp_path, monkeypatch):
    """run_training(stage='sft', locked_eval=True) on a tiny model: parent+candidate locked eval are recorded."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_grpo import FakeApi as _Api, char_tokenizer, tiny_model, _write_parent
    from bloom import trainer
    from bloom.trainer_data import INSTRUCT_MIX_V2, encode_pairs

    tok = char_tokenizer()
    eot = tok.token_to_id("<|endoftext|>")
    model = tiny_model(tok, context=200)
    parent, pm = _write_parent(tmp_path, tok, model)
    pm.update(step=60000, cumulative_train_tokens=1_146_880_000, train_tokens=163_840_000, training_track="pretraining",
              target_repo="Centauri-Bloom/Bloom-Text-0.2", dataset="FineWeb-Edu + literature + TinyStories")
    pm.pop("stage")
    (parent / "metrics.json").write_text(json.dumps(pm))
    (parent / "model.safetensors").write_bytes((parent / "step-000010.safetensors").read_bytes())

    class Api(_Api):
        def model_info(self, repo, revision=None):
            return SimpleNamespace(sha=("c" * 40) if "Chat" in repo else ("d" * 40))

    monkeypatch.setattr(trainer, "PARAM_COUNT_50M", pm["parameters"])
    monkeypatch.setattr(trainer, "load_hub_parent",
                        lambda api, repo, token=None, revision=None: (parent, model.config, pm, parent / "model.safetensors", "p" * 40))
    snap = tmp_path / "tokenizer-root"
    snap.mkdir()
    tok.save(str(snap / "tokenizer.json"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(snap))

    rows = [("What is %d plus 1?" % i, str(i + 1)) for i in range(40)]
    def fake_prepare(recipe, repo, tokenizer, context, eot_id, seed, jaron_path=None, budgets=None):
        data = encode_pairs(rows, tokenizer, context, eot_id)
        return data, encode_pairs(rows[:8], tokenizer, context, eot_id), {}, {"recipe": recipe, "sources": [{"name": "Jaron", "source_count": 40}]}
    monkeypatch.setattr(trainer, "prepare", fake_prepare)

    out = tmp_path / "post-sft-out"
    metrics = trainer.run_training(Api(), INSTRUCT_MIX_V2, 6, ".", out, jaron_path=["unused.json"], model_size="50M",
                                   device="cpu", parent_hf_repo="Centauri-Bloom/Bloom-Text-0.2",
                                   parent_hf_revision="p" * 40, stage="sft", locked_eval=True)
    assert metrics["target_repo"] == "Centauri-Bloom/Bloom-Chat-0.2" and metrics["target_revision"] == "c" * 40
    assert metrics["parent_revision"] == "p" * 40
    assert metrics["parent_locked_eval"]["n_items"] == 210 and metrics["candidate_locked_eval"]["n_items"] == 210
    assert metrics["locked_eval_sha256"] == metrics["parent_locked_eval"]["eval_sha256"]
    ev = json.loads((out / "evaluation.json").read_text())
    assert ev["parent_locked_eval"] and ev["candidate_locked_eval"]
    assert (out / "step-000006.safetensors").is_file()
    # Default behaviour is unchanged: no locked eval unless asked.
    out2 = tmp_path / "post-sft-out2"
    plain = trainer.run_training(Api(), INSTRUCT_MIX_V2, 2, ".", out2, jaron_path=["unused.json"], model_size="50M",
                                 device="cpu", parent_hf_repo="Centauri-Bloom/Bloom-Text-0.2",
                                 parent_hf_revision="p" * 40, stage="sft")
    assert "candidate_locked_eval" not in plain


def test_instruct_mix_v2_shares_caps_and_repeats(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from bloom.trainer_data import (
        INSTRUCT_MIX, INSTRUCT_MIX_V2, INSTRUCT_SOURCES, KNOWN_JARON_SHAS, JARON_V2_SHA, MIX_SETTINGS,
        instruct_mix_weights, prepare_instruct_mix,
    )
    assert len(JARON_V2_SHA) == 64
    assert instruct_mix_weights(INSTRUCT_MIX) == {"Jaron": 4, "Dolly-15k": 2, "OpenAssistant OASST1": 2, "GSM8K": 1}
    assert instruct_mix_weights(INSTRUCT_MIX_V2) == {"Jaron": 4, "Dolly-15k": 2, "OpenAssistant OASST1": 1, "GSM8K": 1}
    assert MIX_SETTINGS[INSTRUCT_MIX_V2]["max_repeat"] < MIX_SETTINGS[INSTRUCT_MIX]["max_repeat"]

    class Tok:
        def encode(self, text):
            return NS(ids=[1 + ord(c) % 14 for c in text])

    def make(prefix, n):
        return [{"prompt": "%s question number %d?" % (prefix, i), "response": "Answer %d." % i} for i in range(n)]
    pool = {"Dolly-15k": make("d", 300), "OpenAssistant OASST1": make("o", 60), "GSM8K": make("g", 200)}

    def fake_source(name, tok, context, eot):
        rows = pool[name]
        return rows, {"name": name, "dataset": "stub/" + name, "license_note": "test", "filtered_count": len(rows),
                      "cap": INSTRUCT_SOURCES[name]["cap"], "weight": INSTRUCT_SOURCES[name]["weight"]}
    monkeypatch.setattr("bloom.trainer_data.load_jaron_sources",
                        lambda _p: (make("j", 1000), [{"path": "j.json", "source_sha256": "ab", "label": "t", "source_count": 1000}]))
    monkeypatch.setitem(MIX_SETTINGS, INSTRUCT_MIX_V2, {**MIX_SETTINGS[INSTRUCT_MIX_V2], "train_cap": 800, "jaron_cap": 500,
                                                       "caps": {"Dolly-15k": 200, "OpenAssistant OASST1": 100, "GSM8K": 150}})
    t, v, _, meta = prepare_instruct_mix(Tok(), 128, 0, 47, "x.json", source_loader=fake_source, recipe=INSTRUCT_MIX_V2)
    assert meta["recipe"] == INSTRUCT_MIX_V2 and meta["train_cap"] == 800 and meta["max_repeat"] == 4
    by = {s["name"]: s for s in meta["sources"]}
    assert by["Jaron"]["weight"] == 4 and by["OpenAssistant OASST1"]["weight"] == 1
    assert by["Jaron"]["cap"] == 500 and by["Dolly-15k"]["cap"] == 200
    assert sum(s["mixed_train_count"] for s in by.values()) == len(t) <= 800
    assert by["OpenAssistant OASST1"]["mixed_train_count"] <= 4 * by["OpenAssistant OASST1"]["unique_train"]
    assert provenance_is_private_safe(meta)
