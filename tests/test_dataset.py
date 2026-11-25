import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest
import torch

from bloom.data import TextDataset, make_dataloader
from bloom.dataset import build_dataset, verify_dataset
from bloom.tokenizer import VOCAB_SIZE, decode, encode


FIXTURE = Path(__file__).parent / "fixtures" / "text"
SOURCE_LOCK = FIXTURE / "sources.json"


@pytest.fixture
def built_corpus(tmp_path):
    directory = tmp_path / "dataset"
    return directory, build_dataset(SOURCE_LOCK, directory, split_seed=3)


def shard_rows(directory, manifest, split):
    for shard in manifest["splits"][split]:
        for line in (directory / shard["path"]).read_bytes().splitlines():
            yield json.loads(line)


@pytest.mark.parametrize("text", ["", "abc\x00\r\n", "Cafe\u0301 ≠ Café — 你好 🌱"])
def test_byte_tokenizer_round_trip(text):
    tokens = encode(text)
    assert VOCAB_SIZE == 256
    assert tokens == list(text.encode("utf-8"))
    assert decode(tokens) == text


def test_byte_tokenizer_rejects_invalid_utf8_and_out_of_range_tokens():
    with pytest.raises(UnicodeDecodeError):
        decode([0xF0, 0x9F])
    assert decode([0xF0, 0x9F], errors="replace") == "\ufffd"
    with pytest.raises(ValueError):
        decode([VOCAB_SIZE])


def test_seeded_builds_are_byte_identical(tmp_path, built_corpus):
    first, manifest = built_corpus
    copy = tmp_path / "reordered-fixture"
    shutil.copytree(FIXTURE, copy)
    lock = copy / SOURCE_LOCK.name
    spec = json.loads(lock.read_bytes())
    spec["sources"].reverse()
    lock.write_text(json.dumps(spec), encoding="utf-8")
    second = tmp_path / "second"
    assert build_dataset(lock, second, split_seed=3) == manifest
    assert {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }


def test_normalized_documents_are_deduplicated_before_splitting(built_corpus):
    directory, manifest = built_corpus
    all_texts = []
    all_ids = []
    for split in ("train", "validation", "test"):
        documents = list(shard_rows(directory, manifest, split))
        assert documents
        all_texts.extend(document["text"] for document in documents)
        all_ids.extend(document["id"] for document in documents)
    expected = {
        path.read_text(encoding="utf-8").strip() for path in FIXTURE.glob("*.txt")
    }
    assert len(all_texts) == len(expected) == 8
    assert set(all_texts) == expected
    assert len(set(all_ids)) == len(all_texts)
    assert all_ids == [hashlib.sha256(text.encode()).hexdigest() for text in all_texts]


def test_verification_rejects_duplicates_with_valid_checksums(built_corpus):
    directory, manifest = built_corpus

    def canonical_bytes(value):
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n").encode("utf-8")

    row = next(shard_rows(directory, manifest, "train"))
    payload = canonical_bytes(row)
    digest = hashlib.sha256(payload).hexdigest()
    shard = manifest["splits"]["validation"][0]
    shard.update(id=f"validation-{digest}", sha256=digest, tokens=len(encode(row["text"])))
    (directory / shard["path"]).write_bytes(payload)
    manifest_data = canonical_bytes(manifest)
    (directory / "manifest.json").write_bytes(manifest_data)
    manifest_digest = hashlib.sha256(manifest_data).hexdigest()
    (directory / "manifest.sha256").write_bytes(f"{manifest_digest}  manifest.json\n".encode())
    with pytest.raises(ValueError, match="duplicate document"):
        verify_dataset(directory)


def test_manifest_identity_and_counts(built_corpus):
    directory, manifest = built_corpus
    digest = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    assert verify_dataset(directory, expected_manifest_sha256=digest) == manifest
    with pytest.raises(ValueError):
        verify_dataset(directory, expected_manifest_sha256="0" * 64)
    shard_ids = set()
    for split, shards in manifest["splits"].items():
        rows = list(shard_rows(directory, manifest, split))
        assert sum(shard["documents"] for shard in shards) == len(rows)
        assert sum(shard["tokens"] for shard in shards) == sum(
            len(encode(row["text"])) for row in rows
        )
        for shard in shards:
            assert shard["id"] not in shard_ids
            shard_ids.add(shard["id"])
            assert shard["sha256"] == hashlib.sha256(
                (directory / shard["path"]).read_bytes()
            ).hexdigest()


@pytest.mark.parametrize("corruption", ["manifest.json", "manifest.sha256", "shard"])
def test_verification_rejects_corruption(built_corpus, corruption):
    directory, manifest = built_corpus
    relative = manifest["splits"]["train"][0]["path"] if corruption == "shard" else corruption
    path = directory / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        verify_dataset(directory)


def test_build_rejects_unpinned_source_bytes(tmp_path):
    modified_fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE, modified_fixture)
    document = modified_fixture / "01-clouds.txt"
    document.write_bytes(document.read_bytes() + b"\n")
    directory = tmp_path / "dataset"
    with pytest.raises(ValueError):
        build_dataset(modified_fixture / SOURCE_LOCK.name, directory)
    assert not directory.exists()


@pytest.mark.parametrize("failure", [None, "hash", "body marker"])
def test_build_checks_and_extracts_pinned_gutenberg_source(tmp_path, monkeypatch, failure):
    modified_fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE, modified_fixture)
    lock = modified_fixture / SOURCE_LOCK.name
    spec = json.loads(lock.read_bytes())
    url = "https://example.invalid/ebook.txt"
    raw = (
        "\ufeffPublic-domain source header\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\r\n"
        "Transcriber notes\r\n"
        "The creek bent through the field.\r\n"
        "A row of bright stones broke the current.\r\n"
        "Illustration credits\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\r\n"
        "License footer\r\n"
    ).encode("utf-8")
    spec["sources"].append({
        "id": "gutenberg-example",
        "title": "Example",
        "author": "Centauri Bloom contributors",
        "url": url,
        "sha256": "0" * 64 if failure == "hash" else hashlib.sha256(raw).hexdigest(),
        "format": "gutenberg-text-v1",
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "body_start": "Missing" if failure == "body marker" else "The creek bent through the field.",
        "body_end": "Illustration credits",
    })
    lock.write_text(json.dumps(spec), encoding="utf-8")

    class Download(io.BytesIO):
        def geturl(self):
            return url

    def fetch(requested_url, *, timeout):
        assert requested_url == url
        assert timeout > 0
        return Download(raw)

    monkeypatch.setattr("bloom.dataset.urlopen", fetch)
    directory = tmp_path / "dataset"
    if failure:
        with pytest.raises(ValueError):
            build_dataset(lock, directory)
        assert not directory.exists()
    else:
        manifest = build_dataset(lock, directory)
        text = "The creek bent through the field.\nA row of bright stones broke the current."
        extracted = [
            row["text"] for split in manifest["splits"] for row in shard_rows(directory, manifest, split)
            if row["text"].startswith("The creek")
        ]
        assert extracted == [text]


@pytest.mark.parametrize("context_length", [7, 32])
def test_loader_generates_next_token_windows_inside_documents(built_corpus, context_length):
    directory, manifest = built_corpus
    shard_id = manifest["splits"]["train"][0]["id"]
    dataset = TextDataset(directory, "train", context_length, shard_id=shard_id)
    batches = list(make_dataloader(dataset, batch_size=2))
    for inputs, targets in batches:
        assert inputs.shape == targets.shape
        assert inputs.shape[1] == context_length
        assert inputs.dtype == targets.dtype == torch.long
        torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])

    actual = torch.cat(
        [torch.cat((inputs, targets[:, -1:]), dim=1) for inputs, targets in batches]
    ).tolist()
    expected = []
    for row in shard_rows(directory, manifest, "train"):
        tokens = encode(row["text"])
        expected.extend(
            tokens[start : start + context_length + 1]
            for start in range(0, len(tokens) - context_length, context_length)
        )
    assert actual == expected


def test_loader_shuffling_is_seeded(built_corpus):
    directory, _ = built_corpus
    dataset = TextDataset(directory, "train", 32)

    def order(seed):
        return torch.cat(
            [inputs for inputs, _ in make_dataloader(dataset, 4, shuffle=True, seed=seed)]
        )

    torch.testing.assert_close(order(11), order(11))
    assert not torch.equal(order(11), order(12))
