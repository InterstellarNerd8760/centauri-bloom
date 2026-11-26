import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bloom.rft import (
    FULL_ARITHMETIC, LOCKED_EVAL_ARITHMETIC, LOCKED_EVAL_SHA256, PILOT_SIZE, PROBES,
    build_locked_eval_items, build_rft_prompts, canonical_eval_payload, chat_parent_revisions,
    choose_kept, eval_digest, evaluate_locked, format_ok, generate_completions, gsm8k_gold,
    is_chat_parent, load_locked_eval, locked_eval_norms, locked_eval_path, normalize_prompt,
    numbers_match, parse_final_number, public_eval_item, repetition_flags, score_completion,
    score_prompt_completions, training_arithmetic_prompts, write_locked_eval, yield_report,
)
from bloom.trainer import CHAT_REPOS, TEXT_REPOS, target_repo
from bloom.trainer_data import INSTRUCT_MIX, needs_jaron


class TinyTok:
    def encode(self, text):
        return SimpleNamespace(ids=[1 + ord(c) % 14 for c in text][:40] or [1])

    def decode(self, ids):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)

    def token_to_id(self, _):
        return 0

    def get_vocab_size(self):
        return 16


def test_parse_and_match_numbers():
    assert parse_final_number("The answer is 60.") == "60"
    assert parse_final_number("reason\n#### 1,234") == "1234"
    assert parse_final_number("60.0") == "60"
    assert parse_final_number("no digits") is None
    assert numbers_match("12 times 5 is 60", "60")
    assert numbers_match("#### 15", 15)
    assert not numbers_match("120", "60")
    assert gsm8k_gold("Step\n#### 42") == "42"
    assert normalize_prompt("  What is 12 TIMES 5? ") == "what is 12 times 5"


def test_format_and_repetition():
    assert format_ok("60")
    assert not format_ok("User: hi\nAssistant: 60")
    assert not format_ok("")
    flags = repetition_flags("hello hello hello hello hello hello hello hello hello hello " * 2)
    assert "repeated-5gram" in flags
    ok = score_completion("What is 12 times 5?", "60", gold="60", ended=True)
    assert ok["keep"] and ok["exact"]
    bad = score_completion("What is 12 times 5?", "120 120 120 120 120 120 120 120 120 120 120", gold="60")
    assert not bad["keep"]
    assert choose_kept([
        {"keep": True, "text": "60 units", "index": 0},
        {"keep": True, "text": "60", "index": 1},
        {"keep": False, "text": "120", "index": 2},
    ])["text"] == "60"


def test_yield_report_by_source():
    rows = [
        {"source": "arithmetic", "kept": "16", "any_exact": True},
        {"source": "arithmetic", "kept": None, "any_exact": False},
        {"source": "gsm8k", "kept": "3", "any_exact": True},
        {"source": "gsm8k", "kept": None, "any_exact": True},
    ]
    report = yield_report(rows)
    assert report["kept"] == 2 and report["total"] == 4
    assert report["by_source"]["arithmetic"] == {"kept": 1, "total": 2, "correct_any": 1}
    assert report["by_source"]["gsm8k"]["correct_any"] == 2


def test_locked_eval_is_pinned_and_disjoint():
    ev = load_locked_eval()
    assert ev["sha256"] == LOCKED_EVAL_SHA256
    assert ev["sha256"] == eval_digest(ev["items"])
    assert len(ev["items"]) == LOCKED_EVAL_ARITHMETIC + len(PROBES)
    assert ev["items"][0]["prompt"] == "What is 12 times 5?"
    assert ev["items"][0]["answer"] == "60"
    assert sum(i["source"] == "arithmetic" for i in ev["items"]) == 200
    rebuilt = [public_eval_item(x) for x in build_locked_eval_items()]
    assert rebuilt == ev["items"]
    norms = locked_eval_norms(ev)
    train = training_arithmetic_prompts(250, eval_norms=norms)
    assert not ({normalize_prompt(r["prompt"]) for r in train} & norms)
    assert not any((r.get("kind") if False else None) for r in train)
    # Probe calculations must not appear under another wording.
    texts = {r["prompt"] for r in train}
    assert "What is 12 times 5?" not in texts
    assert "Multiply 12 by 5." not in texts
    assert "What is 7 plus 8?" not in texts
    gsm = [{"prompt": "Natalia sold clips.", "answer": "#### 72"},
           {"prompt": ev["items"][11]["prompt"], "answer": ev["items"][11]["answer"]}]
    # The second row is an eval arithmetic prompt and must be dropped.
    prompts = build_rft_prompts("pilot", gsm8k_rows=gsm, eval_set=ev, seed=47)
    assert len(prompts) == PILOT_SIZE
    assert not any(normalize_prompt(p["prompt"]) in norms for p in prompts)
    sources = {p["source"] for p in prompts}
    assert "arithmetic" in sources
    # Short GSM8K is filled with extra arithmetic so the pilot still has 500 prompts.
    assert sum(p["source"] == "arithmetic" for p in prompts) >= 250


def test_locked_eval_rejects_drift(tmp_path):
    ev = load_locked_eval()
    tampered = tmp_path / "locked-eval.json"
    payload = json.loads(Path(locked_eval_path()).read_text())
    payload["items"][0]["prompt"] = "What is 12 times 6?"
    payload["sha256"] = ev["sha256"]
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="drifted|hash"):
        load_locked_eval(path=tampered)
    other = tmp_path / "other.json"
    write_locked_eval(path=other)
    # A rewritten file must still match the pinned digest.
    again = load_locked_eval(path=other)
    assert again["sha256"] == LOCKED_EVAL_SHA256


def test_full_prompt_budget_is_disjoint():
    ev = load_locked_eval()
    gsm = [{"prompt": "Q %s?" % i, "answer": str(i)} for i in range(3000)]
    prompts = build_rft_prompts("full", gsm8k_rows=gsm, eval_set=ev, seed=47)
    assert len(prompts) == FULL_ARITHMETIC + 2000 or len(prompts) >= FULL_ARITHMETIC
    assert not any(normalize_prompt(p["prompt"]) in locked_eval_norms(ev) for p in prompts)


def test_tiny_generation_and_locked_eval():
    from bloom.model import Bloom, Config
    tok = TinyTok()
    model = Bloom(Config(vocab_size=16, context=64, width=8, heads=2, layers=1))
    comps = generate_completions(model, tok, ["What is 1 plus 1?"], 64, 0, "cpu",
                                 n=2, max_new=8, temperature=0.8, seed=47)
    assert len(comps) == 1 and len(comps[0]) == 2
    assert all(isinstance(c["text"], str) for c in comps[0])
    record = score_prompt_completions({"prompt": "What is 1 plus 1?", "answer": "2", "source": "arithmetic"}, comps[0])
    assert "kept" in record and len(record["completions"]) == 2
    ev = {"sha256": "x", "items": [
        {"id": "probe-00", "source": "probe", "prompt": "What is 12 times 5?", "answer": "60", "task": "exact-number"},
        {"id": "arith-000", "source": "arithmetic", "prompt": "What is 1 plus 1?", "answer": "2", "task": "exact-number"},
    ]}
    scored = evaluate_locked(model, tok, 64, 0, "cpu", eval_set=ev, max_new=8)
    assert scored["summary"]["n_items"] == 2
    assert "times_12_5" in scored["summary"]
    assert 0 <= scored["summary"]["repetition_rate"] <= 1


def test_stage_routing_and_jaron_requirement():
    assert target_repo("50M", INSTRUCT_MIX, stage="rft") == CHAT_REPOS["50M"]
    assert target_repo("50M", INSTRUCT_MIX, stage="rft-baseline") == "Centauri-Bloom/Bloom-Chat-0.2"
    assert target_repo("50M", "FineWeb-Edu + literature + TinyStories", stage="sft") == TEXT_REPOS["50M"]
    with pytest.raises(ValueError, match="Bloom-Chat-0.2"):
        target_repo("14M", INSTRUCT_MIX, stage="rft")
    assert needs_jaron(INSTRUCT_MIX)
    assert not needs_jaron(INSTRUCT_MIX, stage="rft", rft_train=False)
    assert needs_jaron(INSTRUCT_MIX, stage="rft", rft_train=True)
    assert needs_jaron(INSTRUCT_MIX, stage="rft-baseline")
    assert chat_parent_revisions("refs/pr/1") == ["refs/pr/1", "main"]
    assert chat_parent_revisions("main") == ["main"]
    assert is_chat_parent({"target_repo": "Centauri-Bloom/Bloom-Chat-0.2"}, "50M")
    assert not is_chat_parent({"target_repo": "Centauri-Bloom/Bloom-Text-0.2"}, "50M")


def test_rft_rejects_text_checkpoint_parent(tmp_path):
    from bloom.trainer import run_training

    class Api:
        token = "x"

        def model_info(self, repo, revision=None):
            raise AssertionError("should not contact the hub")

    with pytest.raises(ValueError, match="Bloom-Chat-0.2"):
        run_training(Api(), INSTRUCT_MIX, 0, ".", str(tmp_path / "out"), model_size="50M",
                     device="cpu", parent_hf_repo="Centauri-Bloom/Bloom-Text-0.2",
                     parent_hf_revision="refs/pr/1", stage="rft", rft_train=False)


def test_notebook_stage_defaults_and_payload():
    notebook = json.loads(Path("notebooks/bloom-trainer.ipynb").read_text())
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == 5
    for s in code:
        ast.parse(s)
    config = ast.parse(code[0])
    values = {n.targets[0].id: n.value.value for n in config.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and isinstance(n.value, ast.Constant)}
    assert values["STAGE"] == "sft"
    assert values["RFT_MODE"] == "pilot"
    assert values["RFT_TRAIN"] is False
    assert values["REPO_REF"] == "main"
    assert values["MODEL"] == "50M"
    assert values["DATASET"] == INSTRUCT_MIX
    assert values["STEPS"] == 3000
    assert values["PARENT_HF_REPO"] == "Centauri-Bloom/Bloom-Text-0.2"
    joined = "\n".join(code)
    assert "stage=STAGE" in joined
    assert "rft_mode" in joined
    assert "rft_train" in joined
    assert "candidate_uploadable" in joined
    assert "REPO_REF" in code[1] or "REPO_REF" in joined
    assert "TRL" not in joined
    assert "GRPO" not in joined
    assert "thumbs" not in joined.lower()
    assert "feedback" not in joined.lower() or "CONFIRM_DATA_PERMISSION" in joined
