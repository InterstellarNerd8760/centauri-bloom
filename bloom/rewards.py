"""Rule-based rewards and verifiable tasks for response-level RL.

Everything here is pure Python and original: tasks are generated from small
templates (never from contributor data), answers are checked by rules, and
an optional judge is only an extra term. Prompts overlapping the locked eval
(`datasets/rft-eval-v1`) are excluded, as are the numbered-probe facts.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from bloom.rft import (
    CHAT_TEMPLATE, PROBE_FACTS, format_ok, locked_eval_norms, normalize_prompt,
    parse_final_number, repetition_flags,
)

NAMES = ("Sam", "Maya", "Leo", "Nia", "Omar", "Priya", "Jon", "Ana", "Kai", "Zoe", "Ravi", "Ella")
ITEMS = ("apples", "pencils", "marbles", "stickers", "cards", "books", "coins", "shells", "balloons", "cookies")
# Widely known capitals. France is absent on purpose: it is a locked-eval probe.
CAPITALS = {
    "Japan": "Tokyo", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin", "Canada": "Ottawa",
    "Egypt": "Cairo", "Kenya": "Nairobi", "Peru": "Lima", "Chile": "Santiago", "Norway": "Oslo",
    "Greece": "Athens", "Portugal": "Lisbon", "Ireland": "Dublin", "Austria": "Vienna", "Cuba": "Havana",
    "Thailand": "Bangkok", "Argentina": "Buenos Aires", "Poland": "Warsaw", "Sweden": "Stockholm",
    "Turkey": "Ankara", "India": "New Delhi", "Australia": "Canberra", "Russia": "Moscow",
    "Hungary": "Budapest", "Finland": "Helsinki", "Vietnam": "Hanoi", "Iran": "Tehran",
}
WORDS = ("planet", "rocket", "garden", "window", "river", "castle", "button", "silver", "pencil",
         "bridge", "cloud", "forest", "magnet", "violin", "ladder", "candle", "orange", "stone")
TOPICS = ("gravity", "a volcano", "photosynthesis", "the Moon", "a prime number", "an atom", "a rainbow",
          "a magnet", "the ocean", "electricity", "a compass", "a telescope", "a glacier", "a comet",
          "a triangle", "a seed", "sound", "a satellite")
ACTIVITIES = ("studying for a test", "running a mile", "learning to code", "writing an essay",
              "staying focused", "starting a garden", "learning a new language")
UNITS = (("minutes", "seconds", 60), ("hours", "minutes", 60), ("meters", "centimeters", 100),
         ("kilometers", "meters", 1000), ("dollars", "cents", 100), ("weeks", "days", 7), ("feet", "inches", 12))

VERIFIABLE = ("arithmetic", "word_problem", "successor", "even_odd", "compare", "count_letters",
              "first_letter", "reverse_word", "sort_numbers", "unit_convert", "capital", "gsm8k")
OPEN = ("open",)
TASK_TYPES = VERIFIABLE + OPEN
NUMERIC = {"arithmetic", "word_problem", "successor", "count_letters", "unit_convert", "gsm8k"}
DEFAULT_MIX = {"arithmetic": 0.26, "word_problem": 0.10, "successor": 0.04, "even_odd": 0.04, "compare": 0.04,
               "count_letters": 0.04, "first_letter": 0.04, "reverse_word": 0.04, "sort_numbers": 0.04,
               "unit_convert": 0.04, "capital": 0.04, "gsm8k": 0.06, "open": 0.22}
HOLDOUT_SEED = 9001


@dataclass(frozen=True)
class RewardConfig:
    w_task: float = 1.0
    w_partial: float = 0.25      # share of w_task given for a close wrong number
    w_format: float = 0.15
    w_eos: float = 0.15
    w_repeat: float = 0.25       # per repetition flag, at most two flags counted
    w_length: float = 0.2
    w_judge: float = 0.5         # judge in [0, 1] is mapped to [-w_judge, +w_judge]
    short_band: tuple = (1, 40)  # answer length band (tokens) for verifiable tasks
    open_band: tuple = (5, 80)   # and for open prompts
    judge_on_verifiable: bool = False


@dataclass
class Task:
    type: str
    prompt: str
    gold: object = None
    meta: dict = field(default_factory=dict)

    @property
    def verifiable(self):
        return self.type in VERIFIABLE

    @property
    def norm(self):
        return normalize_prompt(self.prompt)

    def as_dict(self):
        return {"type": self.type, "prompt": self.prompt, "gold": self.gold, "meta": self.meta}


def chat_prompt(prompt):
    return CHAT_TEMPLATE.format(prompt=prompt)


# ---------------------------------------------------------------- generators

def _arithmetic(rng):
    a, b = rng.randint(1, 99), rng.randint(1, 49)
    kind = rng.choice(("add", "sub_from", "mul", "times", "minus", "plus"))
    if kind in ("mul", "times"):
        a, b = rng.randint(2, 20), rng.randint(2, 12)
    template, gold = {
        "add": ("What is {a} plus {b}?", a + b), "plus": ("Add {a} and {b}.", a + b),
        "sub_from": ("Subtract {b} from {a}.", a - b), "minus": ("What is {a} minus {b}?", a - b),
        "mul": ("Multiply {a} by {b}.", a * b), "times": ("What is {a} times {b}?", a * b),
    }[kind]
    return Task("arithmetic", template.format(a=a, b=b), str(gold), {"kind": kind, "a": a, "b": b})


def _word_problem(rng):
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    a, b = rng.randint(5, 40), rng.randint(2, 30)
    if rng.random() < 0.5:
        return Task("word_problem", f"{name} has {a} {item} and finds {b} more. How many {item} does {name} have now?",
                    str(a + b))
    a, b = max(a, b) + 1, min(a, b)
    return Task("word_problem", f"{name} has {a} {item} and gives away {b}. How many {item} are left?", str(a - b))


def _successor(rng):
    n = rng.randint(1, 500)
    if rng.random() < 0.5:
        return Task("successor", f"What number comes right after {n}?", str(n + 1))
    return Task("successor", f"What number comes right before {n}?", str(n - 1))


def _even_odd(rng):
    n = rng.randint(1, 999)
    return Task("even_odd", f"Is {n} even or odd?", "even" if n % 2 == 0 else "odd")


def _compare(rng):
    a, b = rng.sample(range(1, 200), 2)
    which = rng.choice(("bigger", "smaller"))
    gold = max(a, b) if which == "bigger" else min(a, b)
    return Task("compare", f"Which number is {which}, {a} or {b}?", str(gold), {"other": str(a + b - gold)})


def _count_letters(rng):
    word = rng.choice(WORDS)
    letter = rng.choice(sorted(set(word)))
    return Task("count_letters", f"How many times does the letter {letter} appear in the word {word}?",
                str(word.count(letter)))


def _first_letter(rng):
    word = rng.choice(WORDS)
    return Task("first_letter", f"What is the first letter of the word {word}?", word[0])


def _reverse_word(rng):
    word = rng.choice(WORDS)
    return Task("reverse_word", f"Spell the word {word} backwards.", word[::-1])


def _sort_numbers(rng):
    nums = rng.sample(range(1, 100), 3)
    return Task("sort_numbers", f"Put these numbers in order from smallest to largest: {nums[0]}, {nums[1]}, {nums[2]}.",
                sorted(nums))


def _unit_convert(rng):
    big, small, factor = rng.choice(UNITS)
    n = rng.randint(2, 20)
    return Task("unit_convert", f"How many {small} are in {n} {big}?", str(n * factor))


def _capital(rng):
    country = rng.choice(sorted(CAPITALS))
    return Task("capital", f"What is the capital of {country}?", CAPITALS[country])


def _open(rng):
    pick = rng.random()
    if pick < 0.5:
        return Task("open", f"Explain what {rng.choice(TOPICS)} is in one sentence.")
    if pick < 0.75:
        return Task("open", f"Give me one short tip for {rng.choice(ACTIVITIES)}.")
    if pick < 0.9:
        return Task("open", f"Tell me a short fact about {rng.choice(TOPICS)}.")
    return Task("open", rng.choice(("Say hello and ask how my day is going.", "Introduce yourself in two sentences.",
                                    "Say thank you politely.", "Wish me luck on my test.")))


GENERATORS = {
    "arithmetic": _arithmetic, "word_problem": _word_problem, "successor": _successor, "even_odd": _even_odd,
    "compare": _compare, "count_letters": _count_letters, "first_letter": _first_letter,
    "reverse_word": _reverse_word, "sort_numbers": _sort_numbers, "unit_convert": _unit_convert,
    "capital": _capital, "open": _open,
}


def _excluded(task, excluded_norms):
    if task.norm in excluded_norms:
        return True
    meta = task.meta
    return task.type == "arithmetic" and (meta.get("kind"), meta.get("a"), meta.get("b")) in PROBE_FACTS


def gsm8k_tasks(rows, excluded_norms=()):
    """Tasks from GSM8K *train* rows ({'prompt','response'} or {'question','answer'})."""
    tasks, seen = [], set(excluded_norms)
    for row in rows or []:
        prompt = row.get("prompt") or row.get("question")
        gold = parse_final_number(row.get("response") or row.get("answer") or "")
        if not isinstance(prompt, str) or gold is None:
            continue
        task = Task("gsm8k", prompt.strip(), gold)
        if task.norm and task.norm not in seen:
            seen.add(task.norm)
            tasks.append(task)
    return tasks


class TaskSampler:
    """Seeded sampler over the task mix. Never yields held-out or locked-eval prompts."""

    def __init__(self, seed, mix=None, excluded_norms=(), gsm8k_rows=None, extra_open_prompts=(),
                 repo_dir=None):
        self.rng = random.Random(seed)
        mix = dict(DEFAULT_MIX if mix is None else mix)
        self.excluded = set(excluded_norms) | locked_eval_norms(repo_dir=repo_dir)
        self.gsm8k = gsm8k_tasks(gsm8k_rows, self.excluded)
        self.extra_open = [p.strip() for p in extra_open_prompts if isinstance(p, str) and p.strip()
                           and normalize_prompt(p) not in self.excluded]
        if not self.gsm8k:
            mix.pop("gsm8k", None)
        unknown = set(mix) - set(TASK_TYPES)
        if unknown:
            raise ValueError("Unknown task types: %s" % sorted(unknown))
        total = sum(mix.values())
        if total <= 0:
            raise ValueError("Task mix is empty.")
        self.types = list(mix)
        self.weights = [mix[t] / total for t in self.types]

    def sample(self):
        for _ in range(1000):
            kind = self.rng.choices(self.types, self.weights)[0]
            if kind == "gsm8k":
                task = self.rng.choice(self.gsm8k)
            elif kind == "open" and self.extra_open and self.rng.random() < 0.5:
                task = Task("open", self.rng.choice(self.extra_open))
            else:
                task = GENERATORS[kind](self.rng)
            if not _excluded(task, self.excluded):
                return task
        raise RuntimeError("Could not draw a task outside the excluded prompts.")

    def batch(self, n):
        return [self.sample() for _ in range(n)]


def build_holdout(per_type=6, seed=HOLDOUT_SEED, repo_dir=None, gsm8k_rows=None, gsm8k_n=0):
    """Fixed, deduplicated held-out tasks: per_type of each generated type (+ optional GSM8K train rows)."""
    rng = random.Random(seed)
    locked = locked_eval_norms(repo_dir=repo_dir)
    seen, tasks = set(locked), []
    for kind in GENERATORS:
        got, tries = 0, 0
        while got < per_type and tries < 2000:
            tries += 1
            task = GENERATORS[kind](rng)
            if task.norm in seen or _excluded(task, locked):
                continue
            seen.add(task.norm)
            tasks.append(task)
            got += 1
    if gsm8k_n:
        tasks.extend(gsm8k_tasks(gsm8k_rows, seen)[:gsm8k_n])
    return tasks


def holdout_norms(tasks):
    return {t.norm for t in tasks}


# ------------------------------------------------------------------ checking

def _words(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def _numbers(text):
    return [int(m) for m in re.findall(r"-?\d+", text.replace(",", ""))]


def check_answer(task, text):
    """Return (correct: bool, closeness in [0, 1]). Closeness is 0 for non-numeric tasks."""
    gold, kind = task.gold, task.type
    if kind in NUMERIC:
        got = parse_final_number(text)
        if got is None:
            return False, 0.0
        try:
            g, p = float(gold), float(got)
        except ValueError:
            return False, 0.0
        correct = p == g
        return correct, 1.0 if correct else max(0.0, 1.0 - abs(p - g) / max(1.0, abs(g)))
    if kind == "compare":
        nums = _numbers(text)
        return (str(gold) in map(str, nums) and task.meta["other"] not in map(str, nums)), 0.0
    if kind == "sort_numbers":
        return _numbers(text) == list(gold), 0.0
    words = _words(text)
    if len(words) > 12:
        return False, 0.0
    if kind == "even_odd":
        other = "odd" if gold == "even" else "even"
        return (gold in words and other not in words), 0.0
    lowered = text.lower()
    if kind == "capital":
        found = [c for c in CAPITALS.values() if re.search(r"\b" + re.escape(c.lower()) + r"\b", lowered)]
        return (found == [gold]), 0.0
    if kind == "first_letter":
        # "a" and "i" are words, and no word in WORDS starts with either.
        singles = {w for w in re.findall(r"\b[a-z]\b", lowered) if w not in ("a", "i")}
        return singles == {gold}, 0.0
    if kind == "reverse_word":
        spelled = re.sub(r"[^a-z]", "", lowered)
        return (gold in words or spelled.endswith(gold)), 0.0
    raise ValueError("Unknown task type " + kind)


def length_score(n_tokens, band):
    """0 inside the band; negative outside. Tokens, including the end token."""
    low, high = band
    if n_tokens < low:
        return -1.0
    if n_tokens > high:
        return -min(1.0, (n_tokens - high) / max(1, high))
    return 0.0


def score_response(task, text, ended, n_tokens, cfg=RewardConfig(), judge_score=None):
    """Total reward and its components for one completion.

    judge_score is an optional precomputed judge rating in [0, 1] (see bloom.judge); None skips the term.
    """
    text = (text or "").strip()
    fmt = format_ok(text, task.prompt)
    flags = [f for f in repetition_flags(text, ended=ended) if f not in ("no-eos", "empty")]
    parts = {"task": 0.0, "partial": 0.0, "format": cfg.w_format * (1.0 if fmt else -1.0),
             "eos": cfg.w_eos * (1.0 if ended else -1.0),
             "repeat": -cfg.w_repeat * min(2, len(flags)),
             "length": cfg.w_length * length_score(n_tokens, cfg.short_band if task.verifiable else cfg.open_band),
             "judge": 0.0}
    correct, closeness = None, 0.0
    if task.verifiable:
        correct, closeness = check_answer(task, text) if fmt else (False, 0.0)
        if correct:
            parts["task"] = cfg.w_task
        else:
            parts["partial"] = cfg.w_task * cfg.w_partial * closeness
    if judge_score is not None and (not task.verifiable or cfg.judge_on_verifiable) and text:
        parts["judge"] = cfg.w_judge * (2.0 * min(1.0, max(0.0, float(judge_score))) - 1.0)
    return {"reward": sum(parts.values()), "parts": parts, "correct": correct, "format_ok": fmt,
            "repetition": flags, "ended": bool(ended), "tokens": int(n_tokens)}
