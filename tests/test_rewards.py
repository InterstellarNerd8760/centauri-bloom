import random

from bloom.rewards import (
    CAPITALS, DEFAULT_MIX, GENERATORS, NUMERIC, VERIFIABLE, RewardConfig, Task, TaskSampler, build_holdout,
    check_answer, gsm8k_tasks, holdout_norms, length_score, score_response,
)
from bloom.rft import PROBE_FACTS, locked_eval_norms, normalize_prompt


def good_answer(task):
    g = task.gold
    return {
        "sort_numbers": ", ".join(str(n) for n in g),
        "reverse_word": g,
        "capital": g,
    }.get(task.type, str(g))


def test_generators_have_checkable_gold_and_wrong_answers_fail():
    rng = random.Random(0)
    for kind in VERIFIABLE:
        if kind == "gsm8k":
            continue
        for _ in range(25):
            task = GENERATORS[kind](rng)
            assert task.verifiable and task.prompt.strip()
            ok, _ = check_answer(task, good_answer(task))
            assert ok, (kind, task.prompt, task.gold)
            bad, _ = check_answer(task, "banana banana 999999")
            assert not bad, (kind, task.prompt)


def test_natural_phrasings_are_accepted():
    cases = [
        (Task("arithmetic", "What is 12 times 5?", "60"), "12 times 5 equals 60."),
        (Task("even_odd", "Is 8 even or odd?", "even"), "8 is even."),
        (Task("capital", "What is the capital of Japan?", "Tokyo"), "The capital of Japan is Tokyo."),
        (Task("compare", "Which number is bigger, 7 or 12?", "12", {"other": "7"}), "The bigger number is 12."),
        (Task("first_letter", "What is the first letter of the word planet?", "p"), "The first letter is P."),
        (Task("reverse_word", "Spell the word stone backwards.", "enots"), "enots"),
        (Task("sort_numbers", "Put these numbers in order: 9, 3, 5.", [3, 5, 9]), "3, 5, 9"),
    ]
    for task, text in cases:
        assert check_answer(task, text)[0], task.prompt
    assert not check_answer(cases[0][0], "12 times 5 equals 120.")[0]
    assert not check_answer(cases[1][0], "8 is odd, not even.")[0]
    assert not check_answer(cases[2][0], "Tokyo and Rome")[0]


def test_numeric_closeness_gives_partial_credit_only_when_wrong():
    task = Task("arithmetic", "What is 12 times 5?", "60")
    assert check_answer(task, "60") == (True, 1.0)
    ok, close = check_answer(task, "58")
    assert not ok and 0.9 < close < 1.0
    assert check_answer(task, "no number")[1] == 0.0


def test_sampler_never_returns_locked_eval_probe_facts_or_holdout():
    holdout = build_holdout(per_type=6)
    banned = holdout_norms(holdout) | locked_eval_norms()
    sampler = TaskSampler(3, excluded_norms=holdout_norms(holdout))
    seen_types = set()
    for task in sampler.batch(600):
        seen_types.add(task.type)
        assert task.norm not in banned
        if task.type == "arithmetic":
            assert (task.meta["kind"], task.meta["a"], task.meta["b"]) not in PROBE_FACTS
    assert {"arithmetic", "capital", "open", "word_problem"} <= seen_types
    assert "gsm8k" not in seen_types  # no GSM8K rows supplied
    assert "France" not in " ".join(CAPITALS)


def test_holdout_is_fixed_and_unique():
    a, b = build_holdout(per_type=5), build_holdout(per_type=5)
    assert [t.as_dict() for t in a] == [t.as_dict() for t in b]
    norms = [t.norm for t in a]
    assert len(norms) == len(set(norms))
    assert len(a) == 5 * len(GENERATORS)


def test_sampler_is_seeded():
    one = [t.prompt for t in TaskSampler(11).batch(20)]
    two = [t.prompt for t in TaskSampler(11).batch(20)]
    three = [t.prompt for t in TaskSampler(12).batch(20)]
    assert one == two and one != three


def test_gsm8k_rows_become_tasks_and_respect_exclusions():
    rows = [{"prompt": "Tom has 3 pens and buys 4. How many now?", "response": "3+4=7\n#### 7"},
            {"question": "Q2?", "answer": "stuff #### 1,200"},
            {"prompt": "bad", "response": "no number"}]
    tasks = gsm8k_tasks(rows, excluded_norms={normalize_prompt("Q2?")})
    assert [t.gold for t in tasks] == ["7"]
    sampler = TaskSampler(1, mix={"gsm8k": 1.0}, gsm8k_rows=rows)
    assert sampler.sample().type == "gsm8k"


def test_reward_ordering_prefers_correct_clean_stopped_answers():
    task = Task("arithmetic", "What is 12 times 5?", "60")
    cfg = RewardConfig()
    best = score_response(task, "60", True, 3, cfg)
    wrong = score_response(task, "61", True, 3, cfg)
    far = score_response(task, "5000", True, 3, cfg)
    rambling = score_response(task, "60 " + "and then 60 " * 30, False, 90, cfg)
    echo = score_response(task, "User: What is 12 times 5?\nAssistant: 60", True, 12, cfg)
    assert best["correct"] and best["reward"] > wrong["reward"] > far["reward"]
    assert best["reward"] > rambling["reward"]
    assert not echo["format_ok"] and not echo["correct"] and echo["reward"] < best["reward"]
    assert rambling["repetition"]
    assert best["reward"] <= cfg.w_task + cfg.w_format + cfg.w_eos + 1e-9


def test_open_prompts_reward_format_stop_and_no_repetition_and_judge():
    task = Task("open", "Explain what gravity is in one sentence.")
    cfg = RewardConfig()
    clean = score_response(task, "Gravity is the force that pulls objects toward each other.", True, 14, cfg)
    loop = score_response(task, "gravity is gravity " * 12, False, 70, cfg)
    empty = score_response(task, "", True, 1, cfg)
    assert clean["correct"] is None and clean["reward"] > loop["reward"] and clean["reward"] > empty["reward"]
    liked = score_response(task, "Gravity pulls things together.", True, 8, cfg, judge_score=1.0)
    disliked = score_response(task, "Gravity pulls things together.", True, 8, cfg, judge_score=0.0)
    assert liked["reward"] - disliked["reward"] == 2 * cfg.w_judge
    verifiable = score_response(Task("arithmetic", "2 plus 2?", "4"), "4", True, 2, cfg, judge_score=0.0)
    assert verifiable["parts"]["judge"] == 0.0  # judge is ignored on checkable tasks by default


def test_length_score_bands():
    assert length_score(5, (3, 10)) == 0.0
    assert length_score(1, (3, 10)) == -1.0
    assert -1.0 <= length_score(40, (3, 10)) < 0.0
    assert set(DEFAULT_MIX) <= set(VERIFIABLE) | {"open"} and NUMERIC <= set(VERIFIABLE)
