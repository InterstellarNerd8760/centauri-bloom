# Centauri Bloom

**One model. Many contributors. Built from scratch.**

An independent, community-driven experiment in training a small language model from random initialization using volunteer computing resources. The first model name was inspired by the Anti-Doomer community; this is not an official initiative of that community or affiliated with BigScience BLOOM.

[Project page](https://westonrosch.com/projects/centauri-bloom/) · [Discord](https://discord.gg/mThAEaEgWf) · [Contribution guide](CONTRIBUTING.md)

## License and open AI commitment

Original project code and documentation are licensed under [Apache License 2.0](LICENSE), except where otherwise noted. Bloom targets the [Open Source AI Definition 1.0](https://opensource.org/ai/open-source-ai-definition) for official releases. See our [release policy and checklist](docs/open-source-ai.md) for model and dataset licensing, reproducibility and review requirements. No model is claimed to conform until reviewed.

## Status

Founding phase. No verified official model release or live distributed network exists yet. The first goal is a reproducible two-worker experiment, not a production chatbot. Community image and browser-based text training experiments are underway; their compatibility and reproducibility remain unverified.

## Project resources

GitHub is the canonical home for project documentation, source code, technical decisions and issues.

- [Design system](DESIGN.md) — shared visual standards for Bloom websites and tools.
- [Architecture](docs/architecture.md) — model and collaborative training protocol.
- [Roadmap](docs/roadmap.md) — development milestones.
- [Governance](GOVERNANCE.md) and [contributing](CONTRIBUTING.md) — community workflow.
- [Community model website](https://mironox.de/Bloom) and [Tiny GPT Trainer](https://mironox.de/Bloom/Training) — independent experimental tools, not verified official releases.
- Hugging Face — planned canonical storage for reviewed model weights and licensed datasets; organization setup pending confirmation.

### Community experiments

The Tiny GPT Trainer advertises WebGPU training, a 14.9M-parameter byte-level transformer (8 layers, hidden dimension 392, 8 attention heads, context 128 bytes, vocabulary 256), JSON datasets, checkpoint import/export and GGUF export. These are interface-reported specifications, not source-verified results. A mobile training attempt did not complete successfully. The trainer source, exact checkpoint format and inference compatibility need review before integration.

An experimental text-conditioned image checkpoint has also been shared privately. Its training code, dataset provenance, license and outputs need review before any public release. No contributor weights or datasets are published here without permission.

The proposed text baseline in this repository is separate from the browser prototype until architecture and checkpoint compatibility have been demonstrated. Cloud training is a planned option, not an active service.

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
2. Apply the approved code license; review dataset and weight licenses and document data provenance.
3. Establish a reproducible single-worker baseline.
4. Train two workers from the same checkpoint on disjoint data; validate their updates and compare aggregation with the baseline.
5. Publish results, checkpoint formats and a volunteer notebook. Scale only after measuring reliability and available compute.

See [architecture](docs/architecture.md), [governance](GOVERNANCE.md), and [roadmap](docs/roadmap.md). Use public issues and reviewed pull requests; do not share secrets or private data. Discord is for project discussion, not submission of executable artifacts.
