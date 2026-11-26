# Local CPU instruction tuning

The first run is a 20-step smoke test, not a quality benchmark. Uses the latest merged HF checkpoint; raw contributor data stays on your computer. No automatic upload. Stop with Ctrl-C if the computer gets too hot. Use only hardware and data you have permission to use.

## Intel Mac (experimental)
Use Python 3.11 and at least 3 GB free before installing. PyTorch 2.2.2 is the last official Intel Mac wheel, so do not install this repository's normal newer GPU requirements. Use an isolated environment and run from the repository root.

```sh
git clone https://github.com/InterstellarNerd8760/centauri-bloom.git
cd centauri-bloom
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "torch==2.2.2" "numpy==1.26.4" "safetensors>=0.4,<0.6" "tokenizers>=0.20,<0.22" "huggingface_hub>=0.30,<1"
python -m bloom.local_instruction --data "$HOME/Downloads/bloom_instruction_expanded.json" --steps 20
```

If the repository is already cloned, use its existing directory and run `git pull --ff-only` before creating a fresh venv. If Python 3.11 or a compatible binary wheel is unavailable, stop rather than changing system packages. This Intel environment is a compatibility experiment, not the supported GPU stack.

Place the original contributor JSON in Downloads or supply its actual path. The trainer prints the pinned parent revision and step, reports progress every optimizer step, saves a partial checkpoint every ten steps and a completed SafeTensors candidate plus metrics and evaluation in `runs/`. Review before considering a PR. A completed folder can be uploaded by the separate recovery notebook if copied into Google Drive. Do not merge competing or stale-parent candidates.

For subsequent tests, explicitly select a step budget with `--steps`; 20 is not a model limit. No optimizer-state resume is implemented.
