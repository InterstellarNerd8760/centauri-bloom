"""Experimental GPT-2-compatible GGUF export of the canonical Bloom checkpoint.

Never modify the source weights, tokenizer, or config. Full F32 avoids
quantization loss; inference engines may still differ numerically.
"""
import argparse
import json
from pathlib import Path

from bloom.notebook_workflow import sha256

GGUF_NAME = "bloom-f32.gguf"


def tensor_mapping(cfg):
    mapping = {"tokens.weight": "token_embd.weight",
               "positions.weight": "pos_embd.weight",
               "norm.weight": "output_norm.weight",
               "norm.bias": "output_norm.bias",
               "head.weight": "output.weight"}
    for i in range(cfg.layers):
        prefix = f"blocks.layers.{i}."
        names = {"self_attn.in_proj_weight": "attn_qkv.weight",
                 "self_attn.in_proj_bias": "attn_qkv.bias",
                 "self_attn.out_proj.weight": "attn_output.weight",
                 "self_attn.out_proj.bias": "attn_output.bias",
                 "linear1.weight": "ffn_up.weight",
                 "linear1.bias": "ffn_up.bias",
                 "linear2.weight": "ffn_down.weight",
                 "linear2.bias": "ffn_down.bias",
                 "norm1.weight": "attn_norm.weight",
                 "norm1.bias": "attn_norm.bias",
                 "norm2.weight": "ffn_norm.weight",
                 "norm2.bias": "ffn_norm.bias"}
        mapping.update({prefix + k: f"blk.{i}." + v for k, v in names.items()})
    return mapping


def tokenizer_metadata(path, vocab_size):
    """Only accept the GPT-2 byte-level BPE pipeline, never silently relabel another tokenizer."""
    from tokenizers import Tokenizer
    from gguf import TokenType
    spec = json.loads(Path(path).read_text())
    model = spec["model"]
    pre = spec.get("pre_tokenizer") or {}
    decoder = spec.get("decoder") or {}
    post = spec.get("post_processor") or {}
    if (model.get("type") != "BPE" or pre.get("type") != "ByteLevel"
            or pre.get("add_prefix_space", False) or not pre.get("use_regex", True)
            or decoder.get("type") != "ByteLevel" or spec.get("normalizer") is not None
            or post.get("type") not in (None, "ByteLevel")
            or spec.get("truncation") is not None or spec.get("padding") is not None
            or model.get("byte_fallback", False) or model.get("dropout") is not None
            or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix")
            or model.get("ignore_merges", False)):
        raise ValueError("Tokenizer is not supported by this GPT-2 GGUF exporter.")
    tok = Tokenizer.from_file(str(path))
    if tok.get_vocab_size() != vocab_size:
        raise ValueError("Tokenizer size mismatch")
    tokens = [tok.id_to_token(i) for i in range(vocab_size)]
    if any(t is None for t in tokens):
        raise ValueError("Noncontiguous token IDs")
    special = {}
    for entry in spec.get("added_tokens", []):
        if (not entry.get("special") or entry.get("lstrip") or entry.get("rstrip")
                or entry.get("single_word")):
            raise ValueError("Unsupported added token behavior")
        special[entry["id"]] = entry["content"]
    eos = tok.token_to_id("<|endoftext|>")
    if eos is None or special.get(eos) != "<|endoftext|>":
        raise ValueError("Missing special end-of-text token")
    types = [int(TokenType.CONTROL if i in special else TokenType.NORMAL)
             for i in range(vocab_size)]
    merges = [" ".join(m) if isinstance(m, list) else m for m in model["merges"]]
    if not merges:
        raise ValueError("Missing BPE merges")
    return tokens, types, merges, eos


def export_gguf(weights, config, tokenizer, output, metrics):
    """Validate the checkpoint, write an atomic F32 GGUF, and read it back."""
    import gguf
    import numpy as np
    from bloom.model import Bloom, Config
    from bloom.checkpoint import config_hash, load_weights

    cfg = Config(**json.loads(Path(config).read_text()))
    if config_hash(cfg) != metrics["config_sha256"]:
        raise ValueError("Config hash mismatch")
    state = load_weights(weights, metrics["weights_sha256"], Bloom(cfg).state_dict())
    mapping = tensor_mapping(cfg)
    if set(mapping) != set(state):
        raise ValueError("Unsupported model tensor schema")
    tokens, types, merges, eos = tokenizer_metadata(tokenizer, cfg.vocab_size)
    output = Path(output)
    if output.resolve() in {Path(p).resolve() for p in (weights, config, tokenizer)}:
        raise ValueError("Output must not replace the source checkpoint")
    tmp = output.with_name(output.name + ".tmp")
    try:
        writer = gguf.GGUFWriter(tmp, "gpt2")
        writer.add_name("Centauri Bloom Text 0.1")
        writer.add_file_type(0)  # ALL_F32
        writer.add_context_length(cfg.context)
        writer.add_embedding_length(cfg.width)
        writer.add_block_count(cfg.layers)
        writer.add_feed_forward_length(4 * cfg.width)
        writer.add_head_count(cfg.heads)
        writer.add_layer_norm_eps(1e-5)
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("gpt-2")
        writer.add_token_list(tokens)
        writer.add_token_types(types)
        writer.add_token_merges(merges)
        writer.add_bos_token_id(eos)
        writer.add_eos_token_id(eos)
        writer.add_add_bos_token(False)
        writer.add_add_eos_token(False)
        writer.add_string("centauri.source_weights_sha256", metrics["weights_sha256"])
        writer.add_string("centauri.source_tokenizer_sha256", sha256(tokenizer))
        writer.add_string("centauri.source_config_sha256", metrics["config_sha256"])
        for old, new in mapping.items():
            writer.add_tensor(new, state[old].float().numpy())
        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        reader = gguf.GGUFReader(tmp)
        tensors = {t.name: t for t in reader.tensors}
        if set(tensors) != set(mapping.values()):
            raise ValueError("GGUF tensor names differ from source mapping")
        for old, new in mapping.items():
            if not np.array_equal(tensors[new].data, state[old].float().numpy()):
                raise ValueError("GGUF tensor round-trip mismatch: " + new)
        del tensors, reader
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    print("GGUF saved:", output, "| SHA-256:", sha256(output))
    return output


def export_candidate(out_dir, steps, metrics):
    out = Path(out_dir)
    return export_gguf(out / f"step-{steps:06d}.safetensors", out / "config.json",
                       out / "tokenizer.json", out / GGUF_NAME, metrics)


def submit_current(api, repo, out_dir):
    """Export the current merged revision without training; submit only GGUF."""
    from huggingface_hub import snapshot_download, CommitOperationAdd
    revision = api.model_info(repo).sha
    if not revision:
        raise RuntimeError("Cannot pin current model revision")
    source = Path(snapshot_download(repo, revision=revision, token=api.token,
                  allow_patterns=["model.safetensors", "config.json", "tokenizer.json", "metrics.json"]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((source / "metrics.json").read_text())
    artifact = export_gguf(source / "model.safetensors", source / "config.json",
                           source / "tokenizer.json", out / GGUF_NAME, metrics)
    if api.model_info(repo).sha != revision:
        raise RuntimeError("Main changed; export again from the new revision.")
    result = api.create_commit(repo_id=repo, revision="main", parent_commit=revision,
        create_pr=True, operations=[CommitOperationAdd(path_in_repo=GGUF_NAME,
                                                      path_or_fileobj=str(artifact))],
        commit_message=f"Add experimental F32 GGUF for {revision[:12]}")
    print("GGUF-only PR:", result.pr_url)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", action="store_true", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default="Centauri-Bloom/Bloom-Text-0.1")
    args = parser.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    api.whoami()
    submit_current(api, args.repo, args.output)
