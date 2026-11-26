# Unified Bloom trainer

[Open Colab](https://colab.research.google.com/github/InterstellarNerd8760/centauri-bloom/blob/main/notebooks/bloom-trainer.ipynb) · [Upload recovery](https://colab.research.google.com/github/InterstellarNerd8760/centauri-bloom/blob/main/notebooks/bloom-upload-recovery.ipynb)

The same notebook also runs on Kaggle and locally. It auto-detects the environment: Colab uses Drive plus a Colab `HF_TOKEN` secret, Kaggle uses `/kaggle/working` plus `UserSecretsClient`, and a local checkout uses `./runs` plus the `HF_TOKEN` environment variable. The trainer picks CUDA, then MPS, then CPU. fp16 autocast is used only on CUDA.

1. On Colab choose a T4 GPU runtime. On Kaggle enable Internet and a GPU accelerator. Locally open the notebook from a repo checkout.
2. Choose a model size, dataset, and optimizer step budget. Current defaults are 50M, Instruct mix (Jaron + open), and 3,000 steps at 5e-5, starting from `Centauri-Bloom/Bloom-Text-0.2` at `refs/pr/1` until that repo is merged. Confirm permission for the selected data.
3. Run setup, authorize the Hugging Face token, then run training.
4. Run the last cell to submit the saved candidate PR. Review evaluation and merge before another run.

## 14M and 50M

The 14M path is unchanged: it continues from the latest merged [Bloom-Text-0.1](https://huggingface.co/Centauri-Bloom/Bloom-Text-0.1) checkpoint and opens a PR against that repo. Do not modify `configs/bloom-0.1.json`, the baseline, or Bloom-Text-0.1 from a 50M run.

The 50M path (`configs/bloom-0.2-50m.json`) is a new from-scratch model: width 640, 10 heads, 10 layers, context 512, 54,807,040 parameters, measured with `bloom.model.Bloom`. It reuses the 14M 8192-vocab BPE tokenizer but does not load 14M weights. Pretraining candidates are submitted to `Centauri-Bloom/Bloom-Text-0.2`. Instruction fine-tuning of that checkpoint, including the Instruct mix, starts from merged Bloom-Text-0.2 weights, an unmerged Hugging Face revision set with `PARENT_HF_REPO` and `PARENT_HF_REVISION` (for example `refs/pr/1`), or a completed run folder in `PARENT_RUN_DIR`, and is submitted to `Centauri-Bloom/Bloom-Chat-0.2`. Create the target Hugging Face repo once if it does not exist; the trainer will not create it. Do not merge instruction weights into Bloom-Text-0.2. On a free T4 the 50M run uses batch size 8, gradient accumulation 4, and a 3e-4 learning rate, without gradient checkpointing. Effective tokens per step are the same as the 14M continuation (16,384). A 10,000-step run is about 164M token exposures. Expect roughly 5–11 hours on a free T4, or about 5–9k tokens/sec; Colab and Kaggle sessions can disconnect, so set `RECOVER_RUN_DIR` to the same folder to resume from the last checkpoint, including optimizer state when `partial-optimizer.pt` is present. CPU is supported but impractical for a full 50M run; MPS or CUDA is preferred locally.

For the 50M pretraining mix choose **FineWeb-Edu + literature + TinyStories**: about 20M FineWeb-Edu `sample-10BT` tokens (ODC-By 1.0; underlying web rights vary), the full verified public-domain-v1 Gutenberg corpus, and about 500k TinyStories tokens (CDLA-Sharing-1.0). TinyStories is a much smaller share than in the 14M story-heavy mix. Do not use unlicensed sources such as `open-phi/textbooks`, and keep the contributor SFT corpus held unless you are doing a separate instruction run with permission.

| Dataset | Objective | Preparation |
|---|---|---|
| Jaron | Response-only instruction tuning | Original private JSON, hash checked; normalized-prompt held-out split |
| FineWeb-Edu + TinyStories | Next-token pretraining | Approximately 7.5M and 2.5M source tokens respectively; fixed budgets, independent source validation |
| FineWeb-Edu | Next-token pretraining | sample-10BT, approximately 7.5M source tokens |
| TinyStories | Next-token pretraining | Approximately 2.5M source tokens |
| Cosmopedia | Next-token pretraining | Stanford subset, approximately 3M source tokens |
| Public-domain literature | Next-token pretraining | Repository's pinned and verified public-domain-v1 corpus |
| Public-domain + TinyStories | Next-token pretraining | Verified literature corpus plus approximately 2.5M TinyStories tokens |
| FineWeb-Edu + literature + TinyStories | Next-token pretraining | Approximately 20M FineWeb-Edu tokens, the full public-domain-v1 corpus, and approximately 500k TinyStories tokens; intended for 50M from-scratch runs |
| Arithmetic instruction pilot | Response-only instruction tuning | Original generated arithmetic and basic examples; experimental |
| Instruct mix (Jaron + open) | Response-only instruction tuning | Hash-checked private Jaron set, upweighted, plus capped English Dolly-15k (CC BY-SA 3.0), OASST1 first-turn replies (Apache-2.0, PII dropped), and GSM8K train (MIT). Examples longer than context 512 are dropped. Starts from the 50M pretrained checkpoint; 14M runs target Bloom-Chat-0.1 and leave Bloom-Text-0.1 unchanged |

Source token budgets are not the number of training exposures. 14M continuation steps use 16 examples per microbatch and two accumulated microbatches. 50M from-scratch steps use 8 examples per microbatch and four accumulated microbatches, so both keep a 32-example effective batch. Instruction batches mask prompts and padding in the loss and count non-padding input tokens separately. The 14M tokenizer is reused for 50M; the 14M weights are not. Repeated passes over small datasets can overfit. The original literature + TinyStories notebook used a different story sampling budget; the new selector uses bounded streaming.

Jaron's reviewed JSON is hash-checked and never committed. On Colab it is read from Drive's `MyDrive/centauri-bloom-data/bloom_instruction_expanded.json`, from the optional private `JARON_HF_REPO` dataset, or from an upload that is then saved onto Drive. Multiple reviewed files can be combined and de-duplicated; only allowlisted hashes are accepted. It is required for the Instruct mix and the Jaron recipe. Do not redistribute it without permission. This repository does not record a public license for that contributor set.

The Instruct mix uses the same `User: {prompt}\nAssistant:` template as the Jaron recipe, with loss on assistant tokens only. After filtering and unique caps, Jaron is upweighted (relative weight 4) against Dolly (2), OASST1 (2), and GSM8K (1), with at most 24,000 training examples after repeats so a 3,000-step 50M run fits about 1–2 hours on a free T4 (batch 8, accumulation 4, learning rate 5e-5). Dolly is CC BY-SA 3.0; review attribution and share-alike obligations before merging Bloom-Chat-0.2. OASST1 conversations can contain personal data; labelled PII and disallowed-content rows are dropped, but a residual risk remains. GSM8K's official test split is not used. HuggingFaceTB/smol-smoltalk is not included because it is synthetic and includes model-generated subsets such as Magpie.

Review licenses, provenance, accuracy and underlying content rights before training and merging. FineMath and Wikipedia were optional gated sources in a prior experimental notebook, not confirmed training datasets; the unified selector does not enable them.

## Recovery

Each run stores its settings, pinned parent or tokenizer revision, token windows, provenance, periodic latest weights, optimizer state, evaluation and completed checkpoint in the environment's output directory (Drive on Colab, `/kaggle/working` on Kaggle, `./runs` locally). To resume, set RECOVER_RUN_DIR to that unified run folder and keep the original settings. A restart before the first checkpoint reuses the processed corpus. Mid-run resume loads the last `partial.safetensors` and, when present, `partial-optimizer.pt`. Completed candidates can be submitted from the separate upload recovery notebook without retraining; failed uploads never justify rerunning training. For 14M, a changed parent blocks submission. The token must be authorized by Hugging Face; authentication alone cannot guarantee PR permission.

## Historical notebooks

Older notebooks remain at their original paths for reproducibility and existing links. Do not start new continuation runs from them. The original scratch experiment and contributor smoke test are distinct from canonical continuation.

## GGUF inference export

The final upload cell exports `bloom-f32.gguf` from the exact completed SafeTensors checkpoint and submits it in the same candidate PR alongside all existing artifacts. SafeTensors remains canonical for training. The exporter verifies the checkpoint and config hashes, accepts only a compatible GPT-2 byte-level BPE tokenizer, preserves full F32 weights and reads tensors back to check equality. The GPT-2 mapping is experimental; test actual llama.cpp tokenization and logits before calling it a supported runtime. If conversion fails, the original run remains saved and can be retried using upload recovery. Never retrain just to regenerate an export.

To add GGUF to the current merged Hugging Face model without training, open upload recovery, choose CURRENT_MODEL, authorize an HF write token and run the final cell. It pins the current revision and submits a GGUF-only PR. Review and merge only if its source-weight digest matches the current SafeTensors file. No other model artifacts are changed. Choose COMPLETED_RUN for the original recovery behavior.
