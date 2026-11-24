# Roadmap

## 0 — Scaffold
- Review model, CPU smoke test and aggregation baseline.
- Approve code license, conduct and maintainers.
- Establish a reproducible dependency lock and CI.

## 1 — Real text baseline
- Select and document a licensed dataset, held-out split and deduplication strategy.
- Implement tokenizer, data loader, checkpoint manifest and a reproducible training run.
- Choose model dimensions and confirm actual parameter count.

## 2 — Independent workers
- Implement artifact validation, bounded jobs and exact parent matching.
- Compare a two-computer run with a matched single-worker baseline.
- Record loss, token count, wall time, failures and rollback tests.

## 3 — Community pilot
- Publish a reviewed contributor notebook and model card.
- Enable controlled artifact submission and validation; no automatic execution of contributor code.
- Publish measured results before expanding compute or model size.
