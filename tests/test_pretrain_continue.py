import ast
import json
from pathlib import Path

from bloom.trainer import (
    PRETRAIN_CONTINUE_LR,
    PRETRAIN_CONTINUE_STAGE,
    STAGES,
    TEXT_REPOS,
    is_pretrain_recipe,
    run_training,
    target_repo,
    text_pr_base_revision,
)
from bloom.trainer_data import (
    INSTRUCT_MIX,
    PRETRAIN_CONTINUE_BUDGETS,
    RECIPE_BUDGETS,
    STREAM_ROW_LIMIT,
)


def test_pretrain_continue_constants_and_target():
    assert PRETRAIN_CONTINUE_STAGE == "pretrain-continue"
    assert PRETRAIN_CONTINUE_STAGE in STAGES
    assert PRETRAIN_CONTINUE_LR == 1e-4
    assert STREAM_ROW_LIMIT == 400_000
    assert RECIPE_BUDGETS["FineWeb-Edu + literature + TinyStories"] == {
        "FineWeb-Edu": 20_000_000,
        "TinyStories": 500_000,
    }
    assert PRETRAIN_CONTINUE_BUDGETS["FineWeb-Edu + literature + TinyStories"] == {
        "FineWeb-Edu": 100_000_000,
        "TinyStories": 2_500_000,
    }
    recipe = "FineWeb-Edu + literature + TinyStories"
    assert is_pretrain_recipe(recipe)
    assert not is_pretrain_recipe(INSTRUCT_MIX)
    assert target_repo("50M", recipe, stage="pretrain-continue") == TEXT_REPOS["50M"]
    assert target_repo("50M", recipe, stage="sft") == TEXT_REPOS["50M"]
    assert target_repo("50M", INSTRUCT_MIX, stage="sft") == "Centauri-Bloom/Bloom-Chat-0.2"


def test_text_pr_base_uses_main_when_parent_is_unmerged():
    text = TEXT_REPOS["50M"]
    main = "b6e16cc8ce462326bf4ce4446dfc2f8941bdcb73"
    parent = "66f77363cc174c895533c3be337437ba3a837866"
    assert text_pr_base_revision(text, "50M", None, parent, main) == main
    assert text_pr_base_revision(text, "50M", None, main, main) == main
    assert text_pr_base_revision(text, "50M", "/tmp/run", parent, main) == main
    assert text_pr_base_revision(TEXT_REPOS["14M"], "14M", None, "aaa", "aaa") == "aaa"
    assert text_pr_base_revision(TEXT_REPOS["14M"], "14M", None, "aaa", None) == "aaa"


def test_from_scratch_pretraining_still_rejects_parent():
    class Api:
        token = "x"

        def model_info(self, repo, revision=None):
            raise AssertionError("should not contact the hub")

    try:
        run_training(
            Api(), "FineWeb-Edu + literature + TinyStories", 100, ".", ".",
            model_size="50M", device="cpu", parent_hf_repo="Centauri-Bloom/Bloom-Text-0.2",
            parent_hf_revision="refs/pr/1", stage="sft",
        )
    except ValueError as exc:
        assert "pretrain-continue" in str(exc)
        assert "from-scratch" in str(exc)
    else:
        raise AssertionError("expected from-scratch parent rejection")


def test_pretrain_continue_rejects_instruct_and_14m():
    class Api:
        token = "x"

        def model_info(self, repo, revision=None):
            raise AssertionError("should not contact the hub")

    try:
        run_training(
            Api(), INSTRUCT_MIX, 100, ".", ".",
            model_size="50M", device="cpu", stage="pretrain-continue",
        )
    except ValueError as exc:
        assert "pretraining dataset" in str(exc)
    else:
        raise AssertionError("expected instruct-mix rejection")
    try:
        run_training(
            Api(), "FineWeb-Edu + literature + TinyStories", 100, ".", ".",
            model_size="14M", device="cpu", stage="pretrain-continue",
        )
    except ValueError as exc:
        assert "50M" in str(exc)
    else:
        raise AssertionError("expected 14M rejection")


def test_pretrain_continue_rejects_mixed_parent_sources():
    class Api:
        token = "x"

        def model_info(self, repo, revision=None):
            raise AssertionError("should not contact the hub")

    try:
        run_training(
            Api(), "FineWeb-Edu + literature + TinyStories", 100, ".", ".",
            model_size="50M", device="cpu", parent_run_dir="/tmp/run",
            parent_hf_revision="refs/pr/1", stage="pretrain-continue",
        )
    except ValueError as exc:
        assert "not both" in str(exc)
    else:
        raise AssertionError("expected mixed-parent error")


def test_notebook_exposes_pretrain_continue_without_changing_defaults():
    notebook = json.loads(Path("notebooks/bloom-trainer.ipynb").read_text())
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == 5
    for s in code:
        ast.parse(s)
    values = {
        n.targets[0].id: n.value.value
        for n in ast.parse(code[0]).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and isinstance(n.value, ast.Constant)
    }
    assert values["STAGE"] == "sft"
    assert values["DATASET"] == INSTRUCT_MIX
    assert values["STEPS"] == 3000
    assert values["PARENT_HF_REPO"] == "Centauri-Bloom/Bloom-Text-0.2"
    assert values["PARENT_HF_REVISION"] == "refs/pr/1"
    joined = "\n".join(code)
    assert "pretrain-continue" in code[0]
    assert "pretrain-continue" in joined
    assert "1e-4" in code[0]
    assert "create_pr=True" not in joined  # upload stays in submit_candidate
    assert "submit_candidate(" in code[-1]
    assert "upload_folder(" not in joined
    markdown = "".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    assert "pretrain-continue" in markdown
