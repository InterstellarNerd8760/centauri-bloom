# Locked RFT evaluation set (v1)

`locked-eval.json` is the fixed evaluation set for Bloom-Chat rejection-sampling experiments. It has 10 probes, including "What is 12 times 5?", and 200 held-out arithmetic items. The file records a SHA-256 of the canonical item payload. `bloom.rft.LOCKED_EVAL_SHA256` must match it, and the loader also rebuilds the set from the deterministic builder so the prompts cannot drift.

Do not edit the items in place. Training prompts are sampled from the remaining arithmetic pool and from the GSM8K train split only; they are checked to have no overlap with this set. The official GSM8K test split is not in this file and is not used for training.
