# Architecture

The local CPU experiments use a tiny causal transformer with a 256-byte vocabulary, context 32, width 64, two layers and four heads. The two-worker smoke test uses synthetic integer sequences. The single-worker [text demo](text-dataset.md) starts from random initialization and uses UTF-8 byte IDs from independently split documents. Bloom 0.1 is a separate proposed configuration of roughly 15M parameters.

The [checkpoint exchange v1](checkpoints.md) implements local file validation. A later network coordinator should assign a round ID, parent checkpoint SHA-256, config hash, dataset manifest hash, shard ID, local step budget and expected artifact sizes. Each worker starts at exactly that parent and trains locally. The coordinator verifies metadata, tensor schema, sizes and finite values, rejects duplicates/stale parents, checks update norms and evaluates in isolation before promotion. Retain the previous checkpoint for rollback. Use SafeTensors rather than executable serialization. Checksums do not authenticate contributors or prevent poisoning.

The existing average_updates function only averages equal-weight in-memory deltas from a known common parent; it is not a secure network protocol and does not verify provenance. Test against a single-worker baseline with matched token budget before implementing DiLoCo or more complex optimizers.

Suggested future storage: GitHub for source, Hugging Face for weights and licensed datasets, maintainer-run coordination initially. No hosting accounts are provisioned.
