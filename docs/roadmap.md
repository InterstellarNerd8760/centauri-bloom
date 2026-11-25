# Roadmap

## 0 — Scaffold
- Review model, CPU smoke test and aggregation baseline.
- Code license: Apache-2.0. Review conduct and maintainers.
- Establish a reproducible dependency lock and CI.

## 1 — Real text baseline
- Use the [initial English corpus and offline fixture](text-dataset.md) with whole-document held-out splits and exact deduplication.
- Run the CPU text demo with the byte tokenizer, data loader and dataset manifest. Implement checkpoint exchange and measure a reproducible single-worker baseline.
- Choose model dimensions and confirm actual parameter count.

## 2 — Independent workers
- Implement artifact validation, bounded jobs and exact parent matching.
- Compare a two-computer run with a matched single-worker baseline.
- Record loss, token count, wall time, failures and rollback tests.

## 3 — Community pilot
- Publish a reviewed contributor notebook and model card; review the [OSAID release checklist](open-source-ai.md).
- Enable controlled artifact submission and validation; no automatic execution of contributor code.
- Publish measured results before expanding compute or model size.
