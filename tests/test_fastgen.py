import torch

from bloom.fastgen import forward_cached, generate_batch, logprob_check
from bloom.model import Bloom, Config


def tiny(seed=0, context=48):
    torch.manual_seed(seed)
    return Bloom(Config(vocab_size=40, context=context, width=32, heads=4, layers=3))


def test_cached_logits_match_full_forward():
    model = tiny()
    tokens = torch.randint(0, 40, (3, 20))
    assert logprob_check(model, tokens) < 1e-4


def test_context_limit_is_enforced():
    model = tiny(context=8)
    try:
        forward_cached(model, torch.zeros(1, 9, dtype=torch.long))
    except ValueError as exc:
        assert "context" in str(exc)
    else:
        raise AssertionError("expected context error")


def test_greedy_generation_matches_naive_decode():
    model = tiny()
    prefixes = [[1, 2, 3], [4, 5, 6, 7], [1, 2, 3]]
    got = generate_batch(model, prefixes, n=2, max_new=10, temperature=0, eot=0, device="cpu")
    for prefix, group in zip(prefixes, got):
        ids = list(prefix)
        new = []
        for _ in range(10):
            nxt = int(model(torch.tensor([ids]))[0, -1].argmax())
            new.append(nxt)
            ids.append(nxt)
            if nxt == 0:
                break
        for sample in group:
            assert sample["ids"] == new
            assert sample["ended"] == (new[-1] == 0)


def test_sampling_is_seeded_and_batched_by_length():
    model = tiny()
    prefixes = [[1, 2], [3, 4, 5], [1, 2]]
    a = generate_batch(model, prefixes, n=3, max_new=6, temperature=1.0, eot=0, device="cpu", seed=5)
    b = generate_batch(model, prefixes, n=3, max_new=6, temperature=1.0, eot=0, device="cpu", seed=5)
    assert a == b
    assert len(a) == 3 and all(len(g) == 3 for g in a)
    for group in a:
        for sample in group:
            assert len(sample["ids"]) <= 6
            if sample["ended"]:
                assert sample["ids"][-1] == 0 and 0 not in sample["ids"][:-1]
            else:
                assert len(sample["ids"]) == 6 or 0 not in sample["ids"]


def test_long_prefix_is_trimmed_to_fit_context():
    model = tiny(context=16)
    out = generate_batch(model, [list(range(1, 30))], n=1, max_new=8, temperature=0, eot=0, device="cpu")
    assert 1 <= len(out[0][0]["ids"]) <= 1
