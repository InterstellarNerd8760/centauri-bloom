# Text dataset and tokenizer

The text baseline uses the existing 256-token causal transformer with randomly
initialized weights. Build and verify use only the Python standard library.
Batching and the CPU demo use the project's existing PyTorch dependency.

## Build and verify

With the repository's [test dependencies](../README.md#quick-start) installed,
run from the repository root:

```sh
python -m bloom.dataset build tests/fixtures/text/sources.json runs/text-fixture
python -m bloom.dataset verify runs/text-fixture
python -m bloom.text_smoke runs/text-fixture --context-length 32 --steps 4 --batch-size 4 --seed 42
```

The [offline fixture](../tests/fixtures/text/README.md) is original fictional text.
The demo runs on one CPU thread. It trains only on the train split, measures
cross-entropy in nats per target byte on the same first validation batch before
and after training, and reports the manifest hash, shard IDs, seed, step and token
counts, parameter count and PyTorch version. The test split stays held out.
These short runs exercise the input and training path. Loss can vary with the
software and hardware, and the measurement does not establish model quality.

Build the [initial historical English corpus](../datasets/public-domain-v1/README.md)
with network access:

```sh
python -m bloom.dataset build datasets/public-domain-v1/sources.json runs/public-domain-v1
python -m bloom.dataset verify runs/public-domain-v1
python -m bloom.text_smoke runs/public-domain-v1
```

With Python 3.12 and PyTorch 2.7.1+cpu, the 118,528-parameter model used
512 training target bytes and evaluated 128 held-out target bytes in each
demo. Validation loss went from 41.598 to 34.140 on the offline fixture
and from 38.549 to 34.052 on the historical corpus.

Build requires a new output directory and writes the completed files together.
To reproduce it, use the same source lock, pipeline code and split seed
(`--split-seed 0` by default). SHA-256 is checked on each complete input before
decoding. Local paths are relative to the source lock. Remote sources use HTTPS
and a 4 MiB per-file limit. If a source changes or is unavailable, the build fails.
Use a new dataset version for changes to the reviewed source list or release
recipe.

## Processing and splits

Each input is one document. The builder decodes strict UTF-8, removes an initial
BOM, and converts CRLF and CR to LF. For the Project Gutenberg editions it
validates the outer ebook markers and keeps the lines from the exact locked
`body_start` through the line before `body_end`. It trims spaces, tabs and
newlines at the document edges, drops empty documents, and deduplicates on
the SHA-256 of the resulting UTF-8 bytes. The first source in lexicographic
source-ID order supplies the document if multiple inputs match. Provenance
records every source and its disposition.

Before windowing, the builder sorts unique documents by
`SHA256(ASCII("{split_seed}:{document_id}"))`, breaking ties by document ID.
For `N` unique documents each held-out split gets `max(1, floor(N/10))` works.
The final group is test, the group before it is validation, and the first
group is train. This approximates 80/10/10 by document count, with at least one
work per split and at least three unique works required. Source order does not
affect the build. Exact cleaned duplicates and windows from the same work
cannot land in multiple splits. Near-duplicates, overlapping passages and
similar vocabulary can still occur. The splits can have very different token
counts, and changing the corpus can move existing documents between splits.

## Tokenizer and data loader

`bloom.tokenizer.encode(text)` emits the integer values of the UTF-8 bytes,
from 0 through 255. `decode(ids)` reverses valid UTF-8 sequences. There are no
special or padding tokens and the tokenizer does no normalization. A byte
window can start or end inside a multi-byte character. Decoding such a window
can raise `UnicodeDecodeError`; use `errors="replace"` for display if needed.

```python
from bloom.data import TextDataset, make_dataloader

dataset = TextDataset("runs/text-fixture", "train", context_length=32)
inputs, targets = next(iter(make_dataloader(dataset, batch_size=4, shuffle=True, seed=42)))
```

Inputs and targets are `torch.long` tensors of shape `(batch, context_length)`.
The target at each position is the next byte of the same document. Windows
advance by the context length so targets do not overlap. Partial windows and
documents with fewer than `context_length + 1` bytes are skipped. The model's
configured context limits the window length it can consume. The loader uses a
single process and holds compact byte strings for the selected data in memory.
A seeded PyTorch generator controls shuffle order. The build output is
independent of PyTorch.

## Manifest and checkpoint references

Each output directory contains:

```text
manifest.json
manifest.sha256
data/train-00000.jsonl
data/validation-00000.jsonl
data/test-00000.jsonl
```

The manifest records schema version, dataset ID and version, source URLs or
local paths, titles, authors, license terms, raw SHA-256 hashes, source
disposition, preprocessing version, tokenizer, split seed and shard entries.
Each JSONL row has an `id` (the cleaned document's SHA-256) and `text`. Each
shard entry records its path, SHA-256, document count and number of UTF-8 bytes
in the cleaned documents. This token count is measured before dropping tails
and before batching. JSON files and rows use UTF-8, sorted keys, compact
separators and a final LF. The SHA-256 sidecar is in `sha256sum` format.

For the [checkpoint protocol proposed in issue #4](https://github.com/InterstellarNerd8760/centauri-bloom/issues/4),
use the lowercase SHA-256 of the exact `manifest.json` bytes as
`dataset_sha256` and the assigned shard entry's `id` as `shard_id`. The latter
is the split name followed by a hyphen and the shard SHA-256. To enforce an
assignment, call `TextDataset` with both `expected_manifest_sha256` and
`shard_id`, or verify the entire build with
`python -m bloom.dataset verify runs/text-fixture --expected-manifest-sha256 DIGEST`.
A local checksum helps catch corruption. Compare with a digest from a trusted
assignment to detect replacement of both files. Schema v1 has one shard per
split, including one training shard. Disjoint training-shard assignment for
multiple workers is future work.

The shard layout is ready for a future Hugging Face dataset repository after
rights and maintainer review. The [dataset card](../datasets/public-domain-v1/README.md)
includes the split mapping and source-specific privacy and licensing review.
This pipeline has no publication step. Jaron's browser trainer source has not
been reviewed, so browser trainer compatibility remains unverified.
