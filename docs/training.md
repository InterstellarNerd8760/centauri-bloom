# Train Bloom

The shared training CLI and [contributor notebook](../notebooks/contributor-training.ipynb) use the existing 118,528-parameter transformer, not the proposed 15M release. The notebook is a front end to the same CLI. Start on authorized CPU hardware; CUDA is optional. Review the repository and dependencies before execution. No account tokens, cloud provisioning or automatic uploads are needed.

## First public-domain baseline

From an installed repository checkout:

```sh
python -m bloom.dataset build datasets/public-domain-v1/sources.json runs/public-domain-v1
python -m bloom.dataset verify runs/public-domain-v1
python -m bloom.train runs/public-domain-v1 runs/baseline-001 --steps 100 --batch-size 8 --max-tokens 100000 --device cpu
```

Use a new output directory each time. To avoid network access for a test, substitute the bundled fixture source lock. Use `--device cuda` only on an available, authorized GPU, or `--device auto` to select a device already available in the environment. The trainer caps steps and target bytes; it does not impose a cloud spending or wall-clock limit. Use provider-side budgets and session limits as applicable. The baseline evaluates the same bounded validation windows before and after training; the test split stays unused. Results need not be bit-identical across hardware.

Artifacts: model.safetensors, config.json, tokenizer.json (byte-tokenizer specification, not a Transformers tokenizer export), manifest.json, metrics.json, README.md. The manifest records dataset and weights SHA-256, configuration hash, settings and software versions. Record the exact Git commit separately with the run. Verify weights using an independently trusted digest. To continue a compatible model, pass `--parent PATH --parent-sha256 DIGEST`. This initializes a new optimizer and is not exact resumption. No general-purpose chat capability is claimed.

## Notebook workflow

Open the notebook in an authorized Jupyter environment, including an eligible hosted notebook session, after reviewing and attaching the repository. Follow cells in sequence. Choose the fixture or reviewed corpus, then run with a small explicit budget and inspect local output. Do not add passwords, private data or access keys. Share metrics and reviewed weights through the project issue workflow; no automatic aggregation or uploads. The separate issue #5 will establish disjoint shard assignments, common-parent validation and comparison with a single worker. Issue #11 covers broader datasets; they are not integrated here.

## First experiment criteria

Record exact commit, dataset manifest digest, device, seed, number of steps, tokens, elapsed time, before/after validation loss and resulting artifact digest. A short successful run verifies the training path, not model usefulness. The proposed 15M model is a separate architecture and budget decision after evaluating this baseline. Publication is a separate review under issue #12.
