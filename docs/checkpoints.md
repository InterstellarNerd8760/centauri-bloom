# Checkpoint exchange v1

This protocol supports the Python research scaffold and local, reviewed worker
submissions. It does not assume compatibility with the browser trainer.

## Artifacts

A coordinator publishes a trusted parent `parent.safetensors`, its SHA-256,
the model configuration and a dataset manifest. The coordinator assigns each
worker a round ID, shard ID and step budget. The worker verifies the parent
hash, loads the same architecture and trains on the assigned shard.

Each submission comprises a full-weight SafeTensors file and a UTF-8 JSON
sidecar. No pickle, optimizer state, executable model code or GGUF is accepted
by this protocol. Full weights make it simple to inspect and reconstruct deltas.
Weights are stored on CPU; tensor names, shapes and dtypes must match the
trusted model state. Tied tensors are explicitly duplicated on disk and must
agree when the reference state preserves their aliasing.

## Manifest

Fields (all required, no extra or duplicate keys):

| Field | Type / meaning |
|---|---|
| version | integer 1 |
| kind | string full_weights |
| assignment | coordinator-issued object, defined below |
| steps | positive integer, at most step_budget |
| weights_sha256 | lowercase hex SHA-256 of exact file bytes |
| weights_bytes | positive integer, exact file length |

Assignment fields: round_id and shard_id (1–128 ASCII letters, digits, dots,
underscores or hyphens); parent_sha256, config_sha256 and dataset_sha256
(each 64 lowercase hex characters); step_budget (positive integer).

Config hash: SHA-256 of sorted compact UTF-8 JSON of the dataclass configuration,
with NaN disallowed. Dataset hash: SHA-256 of the coordinator's canonical dataset
manifest bytes (the smoke test instead hashes compact JSON of synthetic data).
Parent hash: SHA-256 of exact published SafeTensors bytes.

## Python API

```python
from bloom.checkpoint import (
    RoundAssignment, config_hash, save_weights, load_weights,
    save_checkpoint, validate_checkpoint,
)
```

Use save_weights(path, state, reference) to produce a trusted parent digest.
Use load_weights(path, digest, reference) to load the parent after verification.
Construct RoundAssignment from coordinator-owned values. Use
save_checkpoint(weights_path, manifest_path, state, reference, assignment, steps)
to write a worker submission. The coordinator calls
validate_checkpoint(weights_path, manifest_path, expected_assignment, reference)
to get validated full weights; subtract the trusted parent to obtain a delta.
Pass a live model's state_dict() as reference to retain tied-weight information.

Validation bounds file reads using the trusted model's tensor sizes, checks
the exact reported file size and hash, rejects mismatched assignments and
validates keys, shapes, dtypes, finite values and tied weights before aggregation.
Manifest reads are capped at 16 KiB. Inputs and outputs should be distinct
regular files in private staging directories.

Run `python -m bloom.smoke` for a local two-worker checkpoint exchange using
synthetic data, and `python -m pytest` for regression tests.

## Limitations and follow-up

These checks establish compatibility and integrity, not authenticity, honest
training, actual use of the assigned shard or model quality. A coordinator must
track consumed assignments and reject duplicate submissions, inspect update
norms and compare held-out metrics before approving any new parent. No network
coordinator, automatic uploads, trust system, production optimizer resume or
browser conversion is implemented here. Issue #5 will test independent workers
against a matched single-worker baseline. Browser integration requires review
of its source and tensor layout. Publish only reviewed artifacts with permission
and appropriate model/dataset licensing.
