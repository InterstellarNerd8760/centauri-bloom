# Centauri Bloom

**One model. Many contributors. Built from scratch.**

An independent, community-driven experiment in training a small language model from random initialization using volunteer computing resources. The first model name was inspired by the Anti-Doomer community; this is not an official initiative of that community or affiliated with BigScience BLOOM.

[Project page](https://westonrosch.com/projects/centauri-bloom/) · [Discord](https://discord.gg/mThAEaEgWf) · [Contribution guide](CONTRIBUTING.md)

## Status

Founding phase. No trained weights or live distributed network exist yet. The first goal is a reproducible two-worker experiment, not a production chatbot.

## Quick start

Python 3.11 recommended. Use a virtual environment and authorized hardware.

```sh
git clone https://github.com/InterstellarNerd8760/centauri-bloom.git
cd centauri-bloom
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
python -m bloom.smoke
```

On Windows activate using the PowerShell script in the virtual environment's Scripts directory. The smoke test is CPU-only, uses synthetic data, and does not upload results.

## Roadmap

1. Review the tiny causal transformer and tokenizer baseline.
2. Agree on code, dataset and weight licensing; document data provenance.
3. Establish a reproducible single-worker baseline.
4. Train two workers from the same checkpoint on disjoint data; validate their updates and compare aggregation with the baseline.
5. Publish results, checkpoint formats and a volunteer notebook. Scale only after measuring reliability and available compute.

See [architecture](docs/architecture.md), [governance](GOVERNANCE.md), and [roadmap](docs/roadmap.md). Use public issues and reviewed pull requests; do not share secrets or private data. Discord is for project discussion, not submission of executable artifacts.
