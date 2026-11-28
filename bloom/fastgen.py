"""KV-cached batched sampling for the Bloom transformer.

`bloom.model.Bloom` has no cache, so naive generation re-runs the whole
sequence for every new token. This module walks the same weights with a
per-layer key/value cache (about 20x fewer FLOPs for a 128-token answer) and
is checked against `Bloom.forward` in tests. It never changes the model or
its checkpoints. Sampling is batched: prompts with the same prefix length are
decoded together, `n` samples per prompt.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.nn import functional as F


def _layer_forward(layer, x, past, causal):
    """One pre-norm encoder layer with an optional key/value cache."""
    attn = layer.self_attn
    batch, steps, width = x.shape
    heads = attn.num_heads
    head_dim = width // heads
    h = layer.norm1(x)
    qkv = F.linear(h, attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = (t.reshape(batch, steps, heads, head_dim).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
    if past is not None:
        k = torch.cat([past[0], k.to(past[0].dtype)], dim=2)
        v = torch.cat([past[1], v.to(past[1].dtype)], dim=2)
    out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), is_causal=causal)
    out = out.transpose(1, 2).reshape(batch, steps, width)
    x = x + attn.out_proj(out)
    x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
    return x, (k, v)


def forward_cached(model, tokens, cache=None):
    """Logits for `tokens` (B, T), continuing from `cache` (list of (k, v) per layer).

    With no cache this is a normal causal forward over the prefix; with a cache,
    `tokens` are the new positions only. Returns (logits, new_cache).
    """
    batch, steps = tokens.shape
    start = 0 if cache is None else cache[0][0].shape[2]
    if start + steps > model.config.context:
        raise ValueError("sequence exceeds context")
    positions = torch.arange(start, start + steps, device=tokens.device)
    x = model.tokens(tokens) + model.positions(positions)
    new_cache = []
    for i, layer in enumerate(model.blocks.layers):
        x, kv = _layer_forward(layer, x, None if cache is None else cache[i], causal=cache is None and steps > 1)
        new_cache.append(kv)
    return model.head(model.norm(x)), new_cache


def _sample_next(logits, temperature, top_k, generator):
    logits = logits.float()
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k and 0 < top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate_batch(model, prefixes, n, max_new, temperature, eot, device, top_k=0,
                   max_rows=256, seed=None):
    """Sample `n` completions for every prefix (a list of token-id lists).

    Returns, per prefix, a list of n dicts:
      ids    completion token ids, including the end token when the model emitted it
      ended  True if the model emitted the end token before the length limit
    Prefixes of the same length are decoded together, at most `max_rows` rows per call.
    The model is put in eval mode; the caller restores train mode.
    """
    model.eval()
    context = model.config.context
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(int(seed))
    results = [None] * len(prefixes)
    buckets = defaultdict(list)
    prefixes = [list(p) or [eot] for p in prefixes]
    for index, prefix in enumerate(prefixes):
        if len(prefix) >= context:
            prefixes[index] = prefix = prefix[-(context - 1):]
        buckets[len(prefix)].append(index)
    per_call = max(1, max_rows // max(1, n))
    for length, indices in sorted(buckets.items()):
        budget = min(max_new, context - length)
        for lo in range(0, len(indices), per_call):
            chunk = indices[lo:lo + per_call]
            rows = [list(prefixes[i]) for i in chunk for _ in range(n)]
            ids = torch.tensor(rows, device=device, dtype=torch.long)
            ended = torch.zeros(len(rows), dtype=torch.bool, device=device)
            out = torch.full((len(rows), budget), eot, dtype=torch.long, device=device)
            lengths = torch.zeros(len(rows), dtype=torch.long, device=device)
            logits, cache = forward_cached(model, ids)
            for t in range(budget):
                nxt = _sample_next(logits[:, -1], temperature, top_k, gen)
                nxt = torch.where(ended, torch.full_like(nxt, eot), nxt)
                out[:, t] = nxt
                lengths += (~ended).long()
                ended = ended | (nxt == eot)
                if bool(ended.all()) or t + 1 == budget:
                    break
                logits, cache = forward_cached(model, nxt.unsqueeze(1), cache)
            out_cpu, len_cpu, end_cpu = out.cpu().tolist(), lengths.cpu().tolist(), ended.cpu().tolist()
            for j, index in enumerate(chunk):
                group = []
                for s in range(n):
                    r = j * n + s
                    group.append({"ids": out_cpu[r][:len_cpu[r]], "ended": bool(end_cpu[r])})
                results[index] = group
    return results


def logprob_check(model, tokens):
    """Max abs difference between cached and uncached logits (used by tests)."""
    model.eval()
    with torch.no_grad():
        full = model(tokens)
        logits, cache = forward_cached(model, tokens[:, :1])
        steps = [logits]
        for t in range(1, tokens.shape[1]):
            logits, cache = forward_cached(model, tokens[:, t:t + 1], cache)
            steps.append(logits)
        stepped = torch.cat(steps, dim=1)
        prefill, _ = forward_cached(model, tokens)
    return max(float((full - stepped).abs().max()), float((full - prefill).abs().max()))
