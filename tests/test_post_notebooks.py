import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = ("notebooks/bloom-post-sft.ipynb", "notebooks/bloom-post-rl.ipynb")


def cells(path):
    nb = json.loads(Path(path).read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def settings(code):
    return {n.targets[0].id: n.value.value for n in ast.parse(code).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebooks_parse_and_only_open_prs(path):
    code = cells(path)
    for src in code:
        ast.parse(src)
    joined = "\n".join(code)
    assert "submit_candidate(" in joined and "assert_upload_clean(" in joined
    for forbidden in ("create_repo(", "upload_folder(", "upload_file(", "delete_repo(", "create_commit(", "push_to_hub("):
        assert forbidden not in joined, forbidden
    assert 'os.environ.get("HF_TOKEN")' in joined and "UserSecretsClient" in joined
    assert "print(AUTH_API.token" not in joined and "print(token" not in joined
    assert "provenance_is_private_safe(" in joined
    assert settings(code[0])["REPO_REF"] == "post-training-sft-rl"


def test_sft_notebook_defaults_and_private_data_handling():
    code = cells(NOTEBOOKS[0])
    values = settings(code[0])
    assert values["STEPS"] == 3000 and values["PARENT_HF_REVISION"] == "auto"
    assert values["MIN_PARENT_TOKENS"] == 1_100_000_000
    assert values["CONFIRM_PRIVATE_NOTEBOOK"] is False  # must be ticked on purpose
    assert values["DATASET"] == "Instruct mix v2 (Jaron v2 + open)"
    assert values["TEXT_REPO"] == "Centauri-Bloom/Bloom-Text-0.2" and values["CHAT_REPO"] == "Centauri-Bloom/Bloom-Chat-0.2"
    joined = "\n".join(code)
    assert "newest_pretrain(" in joined and "scrub_private_artifacts(" in joined
    assert "locked_eval=bool(RUN_LOCKED_EVAL)" in joined
    assert '/kaggle/input' in joined and "reviewed(" in joined
    # Jaron is read from the private input only: no HF dataset download, no Drive upload of it.
    assert "download_jaron_repo" not in joined and "JARON_HF_REPO" not in joined
    assert "FORCE_UPLOAD" in joined and "parent_val_loss_same_windows" in joined


def test_rl_notebook_defaults_and_no_jaron():
    code = cells(NOTEBOOKS[1])
    values = settings(code[0])
    assert values["ALGO"] == "grpo" and values["PARENT_HF_REVISION"] == "auto"
    assert values["JUDGE_MODEL"] == ""  # judge is opt-in
    assert values["LEARNING_RATE"] == 1e-5 and values["KL_BETA"] == 0.02
    joined = "\n".join(code)
    assert "jaron" not in joined.lower()
    assert 'metrics["candidate_uploadable"]' in joined and "run_from_payload" in joined
