"""Build and verify small, pinned text corpora without third-party dependencies."""

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import urlopen

from .tokenizer import TOKENIZER_ID, VOCAB_SIZE

SPLITS = ("train", "validation", "test")
SCHEMA_VERSION = 1
PIPELINE_ID = "document-rank-v1"
MAX_SOURCE_BYTES = 4 * 1024 * 1024
SOURCE_FIELDS = {"id", "title", "author", "sha256", "format", "license", "license_url"}
SOURCE_FORMATS = {"plain-text-v1", "gutenberg-text-v1"}
BODY_FIELDS = {"body_start", "body_end"}
START_MARKER = re.compile(r"^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .+ \*\*\* *$", re.M)
END_MARKER = re.compile(r"^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .+ \*\*\* *$", re.M)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json_bytes(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


def _is_https(url):
    parsed = urlsplit(url)
    return parsed.scheme == "https" and bool(parsed.hostname)


def _validate_source(source):
    if not isinstance(source, dict):
        raise ValueError("each source must be an object")
    fields = SOURCE_FIELDS
    if source.get("format") == "gutenberg-text-v1":
        fields = fields | BODY_FIELDS
    locator = set(source) - fields
    if locator not in ({"path"}, {"url"}) or not fields <= source.keys():
        raise ValueError("each source needs metadata and exactly one path or URL")
    if any(not isinstance(value, str) or not value.strip() for value in source.values()):
        raise ValueError("source fields must be nonempty strings")
    if not _is_sha256(source["sha256"]) or source["format"] not in SOURCE_FORMATS:
        raise ValueError(f"invalid hash or format for source {source['id']}")
    if not _is_https(source["license_url"]):
        raise ValueError(f"source {source['id']} needs an HTTPS license URL")
    if "url" in source and not _is_https(source["url"]):
        raise ValueError(f"source {source['id']} needs an HTTPS source URL")
    if "path" in source:
        path = PurePosixPath(source["path"])
        if path.is_absolute() or not path.parts or ".." in path.parts or any(
            char in source["path"] for char in "\\:"
        ):
            raise ValueError(f"source {source['id']} needs a relative POSIX path")


def _source_bytes(source, root):
    if "url" in source:
        with urlopen(source["url"], timeout=30) as response:
            if not _is_https(response.geturl()):
                raise ValueError(f"source {source['id']} redirected outside HTTPS")
            data = response.read(MAX_SOURCE_BYTES + 1)
    else:
        with (root / source["path"]).open("rb") as handle:
            data = handle.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError(f"source {source['id']} exceeds {MAX_SOURCE_BYTES} bytes")
    if _sha256(data) != source["sha256"]:
        raise ValueError(f"SHA-256 mismatch for source {source['id']}")
    return data


def _preprocess(data, source):
    text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if source["format"] == "gutenberg-text-v1":
        starts = list(START_MARKER.finditer(text))
        ends = list(END_MARKER.finditer(text))
        if len(starts) != 1 or len(ends) != 1 or starts[0].end() >= ends[0].start():
            raise ValueError(f"missing or ambiguous Gutenberg markers for source {source['id']}")
        lines = text[starts[0].end():ends[0].end()].split("\n")
        start, end = source["body_start"], source["body_end"]
        if lines.count(start) != 1 or lines.count(end) != 1 or lines.index(start) >= lines.index(end):
            raise ValueError(f"missing or ambiguous body boundaries for source {source['id']}")
        text = "\n".join(lines[lines.index(start):lines.index(end)])
    return text.strip(" \t\n")


def _split_ids(doc_ids, seed):
    def rank(doc_id):
        return _sha256(f"{seed}:{doc_id}".encode("ascii")), doc_id

    ranked = sorted(doc_ids, key=rank)
    if len(ranked) < 3:
        raise ValueError("at least three unique nonempty documents are required")
    held_out = max(1, len(ranked) // 10)
    return {
        "train": ranked[:-2 * held_out],
        "validation": ranked[-2 * held_out:-held_out],
        "test": ranked[-held_out:],
    }


def _shard_path(split):
    return f"data/{split}-00000.jsonl"


def _shard_info(split, rows, payload):
    digest = _sha256(payload)
    return {
        "id": f"{split}-{digest}",
        "path": _shard_path(split),
        "sha256": digest,
        "documents": len(rows),
        "tokens": sum(len(row["text"].encode("utf-8")) for row in rows),
    }


def _check_dataset_identity(data):
    for key in ("dataset_id", "dataset_version"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"{key} must be a nonempty string")


def _source_records(sources):
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a nonempty list")
    for source in sources:
        _validate_source(source)
    if len({source["id"] for source in sources}) != len(sources):
        raise ValueError("duplicate source ID")
    return sorted(sources, key=lambda source: source["id"])


def _manifest_checksum(data):
    return f"{_sha256(data)}  manifest.json\n".encode("ascii")


def build_dataset(
    source_lock: str | Path, output_directory: str | Path, *, split_seed: int = 0
) -> dict:
    """Build into a new directory from checked raw bytes and return the manifest."""
    source_lock, output = Path(source_lock), Path(output_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    if type(split_seed) is not int:
        raise ValueError("split_seed must be an integer")
    spec = json.loads(source_lock.read_bytes())
    if not isinstance(spec, dict) or set(spec) != {"dataset_id", "dataset_version", "sources"}:
        raise ValueError("source lock must contain dataset_id, dataset_version and sources")
    _check_dataset_identity(spec)
    sources = _source_records(spec["sources"])
    documents, provenance = {}, []
    for source in sources:
        text = _preprocess(_source_bytes(source, source_lock.parent), source)
        doc_id = _sha256(text.encode("utf-8")) if text else None
        if doc_id is None:
            status = "empty"
        elif doc_id in documents:
            status = "duplicate"
        else:
            documents[doc_id] = text
            status = "kept"
        provenance.append({**source, "document_id": doc_id, "status": status})

    split_ids = _split_ids(documents, split_seed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": spec["dataset_id"],
        "dataset_version": spec["dataset_version"],
        "pipeline": PIPELINE_ID,
        "tokenizer": {"id": TOKENIZER_ID, "vocabulary_size": VOCAB_SIZE},
        "split_seed": split_seed,
        "sources": provenance,
        "splits": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        (stage / "data").mkdir(parents=True)
        for split, ids in split_ids.items():
            rows = [{"id": doc_id, "text": documents[doc_id]} for doc_id in ids]
            payload = b"".join(_json_bytes(row) for row in rows)
            (stage / _shard_path(split)).write_bytes(payload)
            manifest["splits"][split] = [_shard_info(split, rows, payload)]
        data = _json_bytes(manifest)
        (stage / "manifest.json").write_bytes(data)
        (stage / "manifest.sha256").write_bytes(_manifest_checksum(data))
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output already exists: {output}")
        stage.rename(output)
    return manifest


def _read_shard(root, split, entries, seen):
    path = _shard_path(split)
    if (not isinstance(entries, list) or len(entries) != 1
            or not isinstance(entries[0], dict) or entries[0].get("path") != path):
        raise ValueError(f"invalid shard entry for {split}")
    payload = (root / path).read_bytes()
    if _sha256(payload) != entries[0].get("sha256"):
        raise ValueError(f"SHA-256 mismatch for shard {path}")
    rows, ids = [], []
    for line in payload.splitlines(keepends=True):
        row = json.loads(line)
        if (not isinstance(row, dict) or set(row) != {"id", "text"}
                or not isinstance(row["text"], str) or not row["text"]
                or row["id"] != _sha256(row["text"].encode("utf-8"))
                or _json_bytes(row) != line):
            raise ValueError(f"invalid document in {path}")
        if row["id"] in seen:
            raise ValueError(f"duplicate document in {path}")
        seen.add(row["id"])
        rows.append(row)
        ids.append(row["id"])
    if _json_bytes(entries[0]) != _json_bytes(_shard_info(split, rows, payload)):
        raise ValueError(f"incorrect shard metadata for {path}")
    return ids


def _check_provenance(provenance, doc_ids):
    if not isinstance(provenance, list) or any(
        not isinstance(source, dict) or not {"document_id", "status"} <= source.keys()
        for source in provenance
    ):
        raise ValueError("invalid source provenance")
    sources = [{k: v for k, v in source.items() if k not in {"document_id", "status"}}
               for source in provenance]
    if sources != _source_records(sources):
        raise ValueError("source IDs are out of order")
    kept = set()
    for source in provenance:
        doc_id = source["document_id"]
        status = source["status"]
        if status == "empty" and doc_id is None:
            continue
        if not _is_sha256(doc_id) or status != ("duplicate" if doc_id in kept else "kept"):
            raise ValueError(f"invalid source disposition for {source['id']}")
        kept.add(doc_id)
    if kept != doc_ids:
        raise ValueError("source provenance does not match shard documents")


def verify_dataset(
    directory: str | Path, *, expected_manifest_sha256: str | None = None
) -> dict:
    """Verify the manifest and shards, optionally against a trusted manifest digest."""
    root = Path(directory)
    try:
        data = (root / "manifest.json").read_bytes()
        if expected_manifest_sha256 is not None and _sha256(data) != expected_manifest_sha256:
            raise ValueError("manifest SHA-256 does not match expected digest")
        if (root / "manifest.sha256").read_bytes() != _manifest_checksum(data):
            raise ValueError("manifest SHA-256 does not match manifest.sha256")
        manifest = json.loads(data)
        if not isinstance(manifest, dict) or _json_bytes(manifest) != data:
            raise ValueError("manifest is not a canonical JSON object")
        _check_dataset_identity(manifest)
        if (type(manifest.get("schema_version")) is not int
                or manifest["schema_version"] != SCHEMA_VERSION
                or manifest.get("pipeline") != PIPELINE_ID
                or manifest.get("tokenizer") != {"id": TOKENIZER_ID, "vocabulary_size": VOCAB_SIZE}
                or type(manifest.get("split_seed")) is not int):
            raise ValueError("unsupported dataset manifest")
        splits = manifest.get("splits")
        if not isinstance(splits, dict) or set(splits) != set(SPLITS):
            raise ValueError("invalid dataset splits")
        seen = set()
        recorded = {split: _read_shard(root, split, splits[split], seen) for split in SPLITS}
        if recorded != _split_ids(seen, manifest["split_seed"]):
            raise ValueError("shards do not match the deterministic split")
        _check_provenance(manifest.get("sources"), seen)
        return manifest
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read dataset: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="build a corpus from a source lock")
    build.add_argument("source_lock", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--split-seed", type=int, default=0)
    verify = subcommands.add_parser("verify", help="verify all local shard and manifest hashes")
    verify.add_argument("output", type=Path)
    verify.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    try:
        if args.command == "build":
            build_dataset(args.source_lock, args.output, split_seed=args.split_seed)
        else:
            verify_dataset(args.output, expected_manifest_sha256=args.expected_manifest_sha256)
        print((args.output / "manifest.sha256").read_text(encoding="ascii").rstrip())
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
