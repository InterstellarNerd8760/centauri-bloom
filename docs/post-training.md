# Post-training Bloom Text 0.2: SFT, then response-level RL

Status: code and notebooks only. Nothing here has been run on a GPU yet, no weights are produced by this change, and nothing is uploaded until a person runs a notebook and merges the resulting Hugging Face PR.

Order of work once pretraining session 6 has landed as `Bloom-Text-0.2` PR #7 (about 1.15B token exposures):

1. [`notebooks/bloom-post-sft.ipynb`](../notebooks/bloom-post-sft.ipynb): supervised fine-tune from the newest pretrain PR on the Instruct mix v2. Opens a new PR on `Bloom-Chat-0.2`.
2. Review and merge that PR (or point the next step at the PR directly).
3. [`notebooks/bloom-post-rl.ipynb`](../notebooks/bloom-post-rl.ipynb): response-level RL (GRPO or best-of-n) from the SFT candidate. Opens another new PR only if it beats its parent on held-out checks.

Fine-tuning first, RL second: RL needs a model that already knows the `User: ...\nAssistant:` format and gets some answers right, otherwise nearly every sampled answer earns the same reward and nothing is learned.

## Step 1. SFT

| Choice | Value | Why |
|---|---|---|
| Parent | newest open PR on `Bloom-Text-0.2` by cumulative tokens (falls back to `main`), pinned to its commit SHA | Session 6 lands as PR #7. A moving `refs/pr/N` can change under a run; the SHA cannot. |
| Start guard | refuses a parent under 1.1B cumulative tokens (`MIN_PARENT_TOKENS`) | Stops a run from silently starting on PR #6 (983M) before PR #7 exists. Set it to 0 on purpose to override. |
| Data | `Instruct mix v2 (Jaron v2 + open)` | See below. |
| Optimiser | the existing instruct recipe: AdamW, peak LR 5e-5, cosine to 10%, batch 8 x 4 accumulation, loss on assistant tokens only | Unchanged from the Chat-0.2 SFT that already ran. |
| Steps | 3,000 (about two passes over 48,000 examples) | The earlier SFT used the same step count. |
| Eval | validation loss on identical windows for parent and candidate, per-source validation loss, and the locked eval (210 prompts) for both | The earlier SFT run never scored the locked eval, so parent and candidate were not comparable on task accuracy. |
| Upload | `submit_candidate`: a new PR against `Bloom-Chat-0.2`, pinned to the `main` commit seen at training time | Existing mechanism, unchanged. |

### Instruct mix v2

The cleaned Jaron v2 file (99,971 pairs, see the data note below) is about 3.5x the 28,121-row file the current mix was built for, but the v1 mix caps Jaron at 8,000 unique pairs. v2 raises the caps and keeps the same filters, licenses and loss masking.

| Source | Weight | Unique train pairs | Mixed train examples | Passes |
|---|---:|---:|---:|---:|
| Jaron v2 (private) | 4 | 30,000 | 24,000 | 0.8 |
| Dolly-15k (CC BY-SA 3.0) | 2 | 11,000 | 12,000 | 1.1 |
| OASST1 first-turn English (Apache-2.0, PII dropped) | 1 | 2,030 | 6,000 | 3.0 |
| GSM8K train (MIT) | 1 | 6,500 | 6,000 | 0.9 |
| Total | | | 48,000 | |

These counts were measured locally by running `prepare_instruct_mix` on the real v2 file and the real open datasets, with the Bloom tokenizer. Notes:

- The v1 mix repeats OASST1 about 5 times (2,030 unique rows, 5,333 slots) and Dolly about 0.9. v2 halves OASST1's weight so it repeats about 3 times.
- Only about 2,000 OASST1 rows survive the English, first-turn, PII and 512-token filters. If more diversity is wanted, that is the source to widen.
- The mean supervised length is about 72 tokens, so 3,000 steps should cost about the same as the earlier Chat SFT.

### Jaron's data stays private

- The notebook reads Jaron only from the attached private Kaggle input. It has no Hugging Face dataset download path and no Drive upload.
- Only allowlisted files are accepted (by SHA-256). This change adds the cleaned v2 hash `395fbe80...88b0`, in its own commit so it can be dropped. A hash is not data.
- The candidate PR carries only: `model.safetensors`, `config.json`, `tokenizer.json`, `metrics.json`, `data-provenance.json` (counts, hashes and file names), `evaluation.json` (fixed prompts, no Jaron rows) and `bloom-f32.gguf`. `assert_upload_clean` fails closed on anything else, and the notebook refuses to upload if the provenance file contains `prompt` or `response` fields.
- The run folder holds `processed-corpus.npz`, which is the tokenized (so decodable) training windows. The last cell deletes it, plus resume state, after the upload. The notebook also refuses to start until you tick that the notebook is **Private**; Kaggle saves everything in `/kaggle/working` with the notebook version.
- The RL stage never reads Jaron at all.
- The model weights are themselves trained on Jaron's data. That was already true of Chat-0.2. Whether publishing weights trained on it is acceptable is a decision for Weston and Jaron, not for this code.

## Step 2. Response-level RL

### Why not extend the existing RFT pilot

The existing rejection-sampling code scores one thing (an exact arithmetic answer plus format and repetition) over arithmetic and GSM8K prompts only. The full run on Chat-0.2 kept 254 of 4,000 prompts (6.4%), trained on 339 examples, and finished with validation loss 14.15 and arithmetic exact 3.5% (parent 2.0%). The method was not the problem so much as the reward and data: too few kept answers, and nothing that rewards stopping, being concise or not looping. The new stage uses a broader reward over many kinds of checkable task and a faster sampler so more prompts can be tried.

### Method

Both algorithms share one loop (`bloom/grpo.py`):

1. Draw `prompts_per_step` tasks (default 16) from a seeded mix.
2. Sample `group_size` answers per task (default 8) from the current policy with a KV-cached, batched sampler (`bloom/fastgen.py`, checked against `Bloom.forward` in tests; about 12x faster than the existing one-prompt-at-a-time sampler on CPU).
3. Score every whole answer (`bloom/rewards.py`).
4. Update.

**GRPO** (default). For each group, advantage = (reward - group mean) / group std. One on-policy policy-gradient step per batch, so the importance ratio is exactly 1 and there is no PPO clipping or value network. Loss is the token-level mean of `-advantage * log p(token)` plus `beta * KL` to the frozen starting model (k3 estimator, `beta` = 0.02). Groups whose answers all earn the same reward carry no signal and are dropped from the update (and counted in the log as inactive). Fixed LR 1e-5 after a 10-step warmup, gradient clip 1.0, fp16 autocast on CUDA.

**Best-of-n** (the conservative alternative). Same rollouts and rewards, but only the single best answer per prompt is kept, and only if it clears a bar (0.9 for checkable tasks, 0.3 for open ones) and beats the group mean. The update is plain supervised fine-tuning on the kept answers. It cannot push probability away from bad answers, so it is slower, but it is more stable.

Why GRPO-style rather than PPO or a reward model: no extra networks, one hyperparameter that matters (beta), and the rewards are rules, so there is nothing to over-optimise except the rules. DPO is not used here because it needs preference pairs from a larger model to be useful, which fits poorly at 55M parameters (`rl-plan.md`).

### Rewards

Total reward = task + partial + format + eos + repeat + length (+ judge). All weights are in `RewardConfig`.

| Term | Value | Notes |
|---|---|---|
| task | +1.0 if the answer is correct | Rule-checked (below). Open prompts have no task term. |
| partial | up to +0.25 for a wrong but close number | Closeness = 1 - relative error. Needed because a 55M model gets about 2-4% of arithmetic exactly right, so exact match alone is sparse. Never exceeds the correct reward. |
| format | +0.15 / -0.15 | Clean assistant-side answer: no `User:` echo, no repeated prompt, non-empty. A malformed answer cannot score as correct. |
| eos | +0.15 / -0.15 | Emitted the end token before the length limit. |
| repeat | -0.25 per flag, at most two | Repeated 5-gram, repeated line, or tight repeating span (`bloom.rft.repetition_flags`). |
| length | 0 inside a band, down to -0.2 outside | 1-40 tokens for checkable tasks, 5-80 for open prompts. |
| judge | -0.5 to +0.5 | Optional, open-ended prompts only (see below). |

Checkable task types, all generated from templates (`bloom/rewards.py`), with answers checked by rules: arithmetic (five phrasings), word problems, successor/predecessor, even or odd, which number is bigger or smaller, count a letter, first letter, spell backwards, sort three numbers, unit conversion, capital cities (France is excluded because it is a locked-eval probe), and optionally GSM8K train questions (final number after `####`). Open-ended prompts ("Explain what gravity is in one sentence", "Give me one short tip for ...") are scored on format, stopping, repetition and length (and the judge if on). Default mix: 26% arithmetic, 10% word problems, 4% each of eight small skills, 6% GSM8K, 22% open.

Prompts are never taken from Jaron's data. Training prompts exclude the 210 locked-eval prompts, the numbered-probe facts, and the held-out set.

### Optional judge

Off by default. If `JUDGE_MODEL` is set (for example `Qwen/Qwen2.5-0.5B-Instruct`, Apache-2.0), `bloom/judge.py` asks it for a 1-5 rating of each open-ended answer and uses the expected rating from the digit logits, mapped to -0.5..+0.5. It only touches open-ended prompts, so it cannot override a checkable answer. Judges at this size are noisy and can be gamed; treat a judge-on run as an experiment and compare it with a rules-only run on the same held-out checks. The judge sees only the prompt and the answer, never contributor data.

### Held-out evaluation and checkpoint selection

Every `eval_every` steps (default 25) the policy is evaluated greedily on:

- the locked eval (`datasets/rft-eval-v1`, 210 prompts, SHA-pinned), and
- a fixed held-out set of 72 generated tasks (6 of each of 12 generator types), plus 24 GSM8K train questions when GSM8K is on, that training prompts can never equal.

A composite score = 0.6 x held-out accuracy + 0.2 x locked arithmetic accuracy + 0.1 x (1 - repetition rate) + 0.1 x ended rate picks the best checkpoint. The candidate is that best checkpoint, not the last one. It is written, and the notebook offers an upload, **only if the composite beat the parent by at least 0.01**; otherwise `candidate_uploadable` is false, no weights are written and nothing is uploaded.

Other guards: early stop if the recent mean per-token KL to the parent passes 1.0; a warning if under 20% of groups carry signal (raise the sampling temperature to 1.1-1.3 or add easier tasks); the run folder is new and the parent is never written to.

### Known limits

- The held-out tasks are the same families the model trains on. A gain there shows the model learned those skills, not general reasoning. Locked-eval arithmetic is the cleaner out-of-template check.
- A near-deterministic policy gives groups with identical rewards and therefore no gradient. This showed up in a toy run: a heavily over-trained model produced no signal at all, and a lightly trained one improved held-out accuracy from 0.21 to 0.35 in 200 steps. Do not over-train the SFT stage.
- Rewards for open-ended answers only constrain form (stop, no loops, sensible length). They do not make answers true. Real open-ended quality needs the judge or human review.
- Best-of-n and GRPO have not been compared on the real model yet; that is the first experiment.

## Compute estimate (free Kaggle T4)

Only CPU numbers were measured here (8 cores): a 128-sequence, 96-token rollout of the 54.8M model takes 7.1 s with the new sampler, versus about 86 s for the same 128 sequences with the existing sampler (5.4 s per 8 sequences). The GPU figures below are estimates from that, from the earlier Chat SFT (3,000 steps took 1,237 s on a T4) and from FLOP counts, and should be checked on the first real run.

| Stage | Estimate | Basis |
|---|---|---|
| SFT, 3,000 steps | about 25 min training, 35-45 min wall clock | Earlier SFT: 21 min for 3,000 steps, similar example lengths. Plus data prep (about 1.5 min), two locked evals, GGUF export and upload. |
| RL, 300 steps, 16 x 8 samples | about 25-35 min including 13 evaluations | About 3-5 s per step (rollout 1-2 s, update 1-2 s), plus roughly 15 s per evaluation. |
| First study: GRPO vs best-of-n, one seed each | about 1-1.5 h | Two runs of the above. |
| One more sweep (temperature or beta, judge on/off) | about 1-2 h | Three or four short runs. |

Total for SFT plus a first RL study is roughly 2-3 GPU-hours, under 10% of one week's 30-hour Kaggle quota. A 100M pretraining run is still about 36 GPU-hours, so post-training is cheap by comparison; do not let it displace a waiting pretraining session.
