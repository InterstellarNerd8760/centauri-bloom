"""Reversible UTF-8 byte tokenization for Bloom's 256-entry vocabulary."""

from collections.abc import Iterable

TOKENIZER_ID = "utf8-byte-v1"
VOCAB_SIZE = 256


def encode(text: str) -> list[int]:
    """Encode text as integer byte values without normalization or special tokens."""
    return list(text.encode("utf-8"))


def decode(tokens: Iterable[int], *, errors: str = "strict") -> str:
    """Decode byte IDs as UTF-8, rejecting invalid UTF-8 by default."""
    return bytes(iter(tokens)).decode("utf-8", errors=errors)
