# Bloom reinforcement learning plan

Status: first experiment implemented as a rejection-sampling (RFT) *pilot* in the unified trainer. DPO, GRPO, TRL, and playground feedback are intentionally not built yet.

Bloom-Chat-0.2 is a custom 50M PyTorch model (width 640, 10 layers, context 512, 8,192-token BPE) after one supervised fine-tune on the Jaron + Dolly + OASST1 + GSM8K mix. The chat template is `User: {prompt}\nAssistant:`. Validation loss fell from about 3.06 to 2.10, and the model now answers instead of echoing, but it still misses easy arithmetic (for example, 12 × 5) and sometimes repeats itself. The playground at westonrosch.com/projects/centauri-bloom/ runs weights in the browser. Compute is free Kaggle T4 GPUs (about 30 hours a week).

## Where this sits relative to pretraining

Continued pretraining toward about **1 billion token exposures** remains the main priority. RFT does not change pretraining defaults, dataset budgets, learning rates, or the 14M path. The notebook default is still `STAGE=sft`, so existing pretrain and instruction recipes behave as they did. RFT is a smaller, later experiment on Bloom-Chat-0.2 only. Do not spend the weekly GPU quota on a long RFT run if a pretraining continuation is waiting.

A 50M model will not become a reliable general assistant from this work. The realistic aim is narrower: fewer arithmetic and format mistakes, less repetition, and slightly more consistent short answers. Measure that with task accuracy and repetition, not validation loss.

## 1. Methods, ranked by value per GPU hour

The first implemented method is rejection-sampling fine-tuning. The others stay on the backlog.

| Rank | Method | Role now |
|---|---|---|
| 1 | Rejection-sampling fine-tune (RFT) | Built. Generate N answers, keep only verifiably correct, non-repetitive, well-formatted ones, then (later) ordinary SFT. Start with a 500-prompt yield pilot before any full run or training. |
| 2 | DPO or ORPO on *on-policy* pairs | Not built. Same generations as RFT, pairing a better answer with a worse one. |
| 3 | GRPO-style RL with verifiable rewards | Not built. Higher cost because every step generates. |
| 4 | Offline DPO on public preference sets | Not built. Weak fit at 50M because chosen answers were written by much larger models. |

Do not start with PPO, a large reward model, or a paid judge. Implement any later preference loss against `bloom.model.Bloom` in this repo; do not pull in TRL.

## 2. Data sources

### Open preference sets (later; unused by the pilot)

| Dataset | Stated license (as of 26 November 2026) | Use for Bloom? |
|---|---|---|
| OpenAssistant OASST1 sibling ranks | Apache-2.0 | Best public starting point later. Already in the SFT mix; keep PII filters. |
| nvidia/HelpSteer2 | CC BY-4.0 | Optional later, with attribution. |
| argilla/ultrafeedback-binarized-preferences-cleaned | MIT on the card | Optional later, drop remaining ShareGPT-sourced rows. |
| Anthropic/hh-rlhf | MIT | Low priority. Intended for preference modeling, not dialogue SFT. |
| HuggingFaceH4/orca_dpo_pairs | MIT on the card | Skip for now (GPT-4 traces). |
| stanfordnlp/SHP | Unclear Reddit scrape | Skip. |

Jaron's reviewed set stays private and is not required for the yield-only pilot. It is required for the SFT baseline and for an RFT training run that replays the original mix.

### Auto-generated verifiable tasks (what the pilot uses)

- **Locked evaluation set** (`datasets/rft-eval-v1/locked-eval.json`): 10 fixed probes, including "What is 12 times 5?", plus 200 held-out arithmetic items. The file hash is pinned as `bloom.rft.LOCKED_EVAL_SHA256`. The loader rebuilds the set from the deterministic builder so the prompts cannot drift. Training prompts are checked to have no overlap with this set. Numbered probe facts are also withheld from training under other wordings.
- **Pilot prompts**: 500 items, 250 arithmetic from the remaining pool plus 250 GSM8K *train* questions (MIT; official test split is not used for training). If GSM8K is short after filtering, extra arithmetic fills the 500.
- **Rewards**: exact match on the parsed final number, a clean assistant-side format (no `User:` echo), and no repeated 5-gram, repeated line, or tight looping span.
- **Simple code rewards**: later, not in the pilot.

### Playground thumbs

Not implemented. Do not collect visitor prompts or votes until that is a separate, reviewed change. The playground still runs locally in the browser.

## 3. How it is wired into the universal notebook

The same `notebooks/bloom-trainer.ipynb` cells, recovery path, and PR upload path are used. A `STAGE` control was added; leaving it at `sft` preserves the previous behavior.

```
STAGE = "sft"          # sft | rft | rft-baseline
RFT_MODE = "pilot"     # pilot (500 prompts) or full
RFT_TRAIN = False      # leave off for the yield pilot
REPO_REF = "main"      # set to the unmerged branch on Kaggle/Colab until this lands on main
```

- Parent must be `Centauri-Bloom/Bloom-Chat-0.2`. The loader tries `PARENT_HF_REVISION` (default recommendation: `refs/pr/1`) and then `main`. The original checkpoint is never written to; new weights go to a new run folder and, only after a training run, a new Hugging Face PR against Bloom-Chat-0.2.
- Learning rate for RFT and the SFT baseline is `2e-5`, with batch 8 and accumulation 4.
- `rft-baseline` trains on the original Instruct mix (Jaron required) with that same step count and learning rate, and scores the locked eval.
- A yield-only pilot writes `rft-yield.json`, `rft-generations.json`, and `locked-eval-parent.json`, sets `candidate_uploadable=false`, and the upload cell will not open a PR or publish copies of the parent weights.
- Generation is batched per prompt (the N samples share a prefix). Mid-run generation is saved every 10 prompts and can resume with `RECOVER_RUN_DIR`.

### Runtime on a T4

The Chat-0.2 SFT run completed 3,000 steps in about 23 minutes. The 500-prompt yield pilot (8 samples, max 128 new tokens, batched) should fit in about 30–90 minutes on a T4, plus a few minutes for the parent locked eval. A later 1,500-step RFT or SFT-baseline train is about 15–25 minutes after generation. Unbatched generation would be much slower and is not used.

## 4. Recommended first experiment (the pilot)

**Measure whether Chat-0.2 already produces enough correct answers to train on.** Do not start the full 4,000-prompt run, and do not set `RFT_TRAIN=True`, until this yield report looks usable.

Exact Kaggle settings (cell 1), after importing this notebook from the branch or setting `REPO_REF` to it:

```
MODEL = "50M"
DATASET = "Instruct mix (Jaron + open)"
STEPS = 3000
STAGE = "rft"
RFT_MODE = "pilot"
RFT_TRAIN = False
REPO_REF = "rft-rejection-sampling-pilot"
CONFIRM_DATA_PERMISSION = True
RECOVER_RUN_DIR = ""
PARENT_RUN_DIR = ""
PARENT_HF_REPO = "Centauri-Bloom/Bloom-Chat-0.2"
PARENT_HF_REVISION = "refs/pr/1"
JARON_FILE = ""
```

Enable Internet and a GPU (T4 x2 is fine; one T4 is enough). The Hugging Face `HF_TOKEN` secret is still used to read the Chat parent; a yield-only run will not upload. Jaron's private dataset is not required for this pilot. After the run, read `rft-yield.json` for kept/total overall and per source (`arithmetic`, `gsm8k`), and `locked-eval-parent.json` for Chat-0.2's score on the locked set, including the "12 times 5" probe.

If kept responses are well under a few dozen, do not train. Diversify or simplify verifiable prompts first. If yield is healthy, a later pair of runs can train: `STAGE=rft`, `RFT_TRAIN=True`, `STEPS=1500`, with a 25% original-SFT replay, and `STAGE=rft-baseline` at the same steps and learning rate, both scored on the locked eval. Success is a gain on the 200 held-out arithmetic items, a lower repetition rate, a correct "12 times 5" more often than not, and no collapse on the open probes. Official GSM8K test accuracy will probably stay low; treat any gain as a bonus. Do not merge on validation loss alone, and do not overwrite Chat-0.2 except by reviewing a new Hugging Face PR.
