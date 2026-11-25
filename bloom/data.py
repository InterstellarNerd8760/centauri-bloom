"""PyTorch next-token windows over a verified Bloom text dataset."""

import hashlib
import json
from bisect import bisect_right
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import verify_dataset
from .tokenizer import encode


class TextDataset(Dataset):
    """Expose fixed-length windows within one split, optionally using one shard.

    Consecutive windows share one boundary byte, so their targets do not overlap.
    Documents shorter than a full window and trailing partial windows are skipped.
    """

    def __init__(
        self,
        directory: str | Path,
        split: str,
        context_length: int,
        *,
        shard_id: str | None = None,
        expected_manifest_sha256: str | None = None,
    ):
        if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length < 1:
            raise ValueError("context_length must be a positive integer")
        directory = Path(directory)
        if expected_manifest_sha256 is None:
            expected_manifest_sha256 = hashlib.sha256(
                (directory / "manifest.json").read_bytes()
            ).hexdigest()
        manifest = verify_dataset(directory, expected_manifest_sha256=expected_manifest_sha256)
        shards = manifest["splits"].get(split)
        if shards is None:
            raise ValueError(f"unknown split {split!r}")
        if shard_id is not None:
            shards = [shard for shard in shards if shard["id"] == shard_id]
            if not shards:
                raise ValueError(f"shard {shard_id!r} does not belong to split {split!r}")

        self.context_length = context_length
        self.dataset_id = manifest["dataset_id"]
        self.dataset_version = manifest["dataset_version"]
        self.manifest_sha256 = expected_manifest_sha256
        self.shard_ids = tuple(shard["id"] for shard in shards)
        self._documents: list[bytes] = []
        self._ends: list[int] = []
        window_count = 0

        for shard in shards:
            data = (directory / shard["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != shard["sha256"]:
                raise ValueError(f"shard changed after verification: {shard['path']}")
            for line in data.split(b"\n"):
                if not line:
                    continue
                tokens = bytes(encode(json.loads(line)["text"]))
                count = (len(tokens) - 1) // context_length
                if count > 0:
                    window_count += count
                    self._ends.append(window_count)
                    self._documents.append(tokens)

        if not window_count:
            raise ValueError(f"split {split!r} has no windows of length {context_length}")

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        document = bisect_right(self._ends, index)
        first_index = self._ends[document - 1] if document else 0
        start = (index - first_index) * self.context_length
        stop = start + self.context_length + 1
        window = torch.tensor(list(self._documents[document][start:stop]), dtype=torch.long)
        return window[:-1], window[1:]


def make_dataloader(
    dataset: TextDataset, batch_size: int, *, shuffle: bool = False, seed: int = 0
) -> DataLoader:
    """Batch windows without starting worker processes, with optional seeded shuffling."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
