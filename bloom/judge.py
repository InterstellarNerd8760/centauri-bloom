"""Optional small LLM judge for open-ended answers. Off by default; never required.

The judge is only an extra reward term for prompts that have no checkable answer. It sees the prompt
and the model's answer and nothing else (no contributor data). `make_hf_judge` needs `transformers`,
which is not a core dependency, and downloads a small open instruct model.
"""
from __future__ import annotations

import re

RUBRIC = ("You are grading an answer from a very small language model. Rate it from 1 (nonsense, repeats itself, "
          "or ignores the question) to 5 (clear, on topic, correct, concise). Reply with one digit only.\n\n"
          "Question: {prompt}\nAnswer: {answer}")


def expected_score(digit_logits):
    """Map logits over the digits 1..5 to a rating in [0, 1] (expected value, rescaled)."""
    import torch
    probs = torch.softmax(torch.as_tensor(digit_logits, dtype=torch.float32), dim=-1)
    value = float((probs * torch.arange(1, 6, dtype=torch.float32)).sum())
    return (value - 1.0) / 4.0


def parse_digit(text):
    """Fallback: first digit 1-5 in a judge reply, rescaled to [0, 1]; None if absent."""
    m = re.search(r"[1-5]", text or "")
    return None if m is None else (int(m.group(0)) - 1) / 4.0


def make_hf_judge(model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cuda", revision=None, batch_size=32):
    """Return judge(pairs) -> list[float in 0..1]. Scores are the expected 1-5 rating from the digit logits."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]

    @torch.no_grad()
    def judge(pairs):
        scores = []
        for lo in range(0, len(pairs), batch_size):
            chunk = pairs[lo:lo + batch_size]
            texts = [tok.apply_chat_template([{"role": "user", "content": RUBRIC.format(prompt=p, answer=a)}],
                                             tokenize=False, add_generation_prompt=True) for p, a in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to(device)
            logits = model(**enc).logits[:, -1, :][:, digit_ids]
            scores.extend(expected_score(row) for row in logits.float().cpu())
        return scores

    judge.model_id = model_id
    judge.revision = revision
    return judge
