import json
import pytest

from bloom.checkpoint import config_hash, save_weights
from bloom.gguf_export import export_gguf, GGUF_NAME
from bloom.model import Bloom, Config


def test_gguf_roundtrip(tmp_path):
    import gguf
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(["hello hello world world", "a tiny test"],
        trainers.BpeTrainer(vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=["<|endoftext|>"]))
    tok = tmp_path / "tokenizer.json"
    tokenizer.save(str(tok))
    cfg = Config(vocab_size=tokenizer.get_vocab_size(), context=16, width=16, heads=4, layers=2)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg.__dict__))
    state = Bloom(cfg).state_dict()
    weights = tmp_path / "model.safetensors"
    digest = save_weights(weights, state, state)
    metrics = {"config_sha256": config_hash(cfg), "weights_sha256": digest}
    output = export_gguf(weights, config, tok, tmp_path / GGUF_NAME, metrics)
    assert output.read_bytes()[:4] == b"GGUF"
    reader = gguf.GGUFReader(output)
    assert len(reader.tensors) == len(state)
    assert bytes(reader.fields["general.architecture"].parts[-1]) == b"gpt2"
    assert weights.is_file()
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        export_gguf(weights, config, tok, tmp_path / "bad.gguf",
                    {**metrics, "weights_sha256": "0"*64})
    assert not (tmp_path / "bad.gguf").exists()


def test_reject_nonmatching_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from bloom.gguf_export import tokenizer_metadata
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    p = tmp_path / "tokenizer.json"
    tokenizer.save(str(p))
    with pytest.raises(ValueError, match="not supported"):
        tokenizer_metadata(p, 0)
