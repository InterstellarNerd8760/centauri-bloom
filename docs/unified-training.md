# Unified Bloom trainer

[Open Colab](https://colab.research.google.com/github/InterstellarNerd8760/centauri-bloom/blob/main/notebooks/bloom-trainer.ipynb) · [Upload recovery](https://colab.research.google.com/github/InterstellarNerd8760/centauri-bloom/blob/main/notebooks/bloom-upload-recovery.ipynb)

1. Select a GPU runtime.
2. Choose a dataset and optimizer step budget; defaults are Jaron and 3,000. Confirm permission for the selected data.
3. Run setup, authorize the Hugging Face token in the password widget, then run training.
4. Run the last cell to submit the saved candidate PR. Review evaluation and merge before another run.

| Dataset | Objective | Preparation |
|---|---|---|
| Jaron | Response-only instruction tuning | Original private JSON, hash checked; normalized-prompt held-out split |
| FineWeb-Edu + TinyStories | Next-token pretraining | Approximately 7.5M and 2.5M source tokens respectively; fixed budgets, independent source validation |
| FineWeb-Edu | Next-token pretraining | sample-10BT, approximately 7.5M source tokens |
| TinyStories | Next-token pretraining | Approximately 2.5M source tokens |
| Cosmopedia | Next-token pretraining | Stanford subset, approximately 3M source tokens |
| Public-domain literature | Next-token pretraining | Repository's pinned and verified public-domain-v1 corpus |
| Public-domain + TinyStories | Next-token pretraining | Verified literature corpus plus approximately 2.5M TinyStories tokens |
| Arithmetic instruction pilot | Response-only instruction tuning | Original generated arithmetic and basic examples; experimental |

Source token budgets are not the number of training exposures. Each step uses 16 examples per microbatch and two accumulated microbatches. Instruction batches mask prompts and padding in the loss and count non-padding input tokens separately. The model size and tokenizer do not change. Repeated passes over small datasets can overfit. The original literature + TinyStories notebook used a different story sampling budget; the new selector uses bounded streaming.

Jaron's original JSON is read from Drive's `MyDrive/centauri-bloom-data/bloom_instruction_expanded.json`, or selected when requested. Do not redistribute it without permission. Review licenses, provenance, accuracy and underlying content rights before training and merging. FineMath and Wikipedia were optional gated sources in a prior experimental notebook, not confirmed training datasets; the unified selector does not enable them.

## Recovery

Each run stores its settings, pinned parent, token windows, provenance, periodic latest weights, evaluation and completed checkpoint in Drive. For a restart before the first checkpoint, set RECOVER_RUN_DIR to that unified run folder and keep the original settings and unchanged parent. Partial optimizer-state resume is not implemented. Completed candidates can be submitted from the separate upload recovery notebook without retraining; failed uploads never justify rerunning training. A changed parent blocks submission. The token must be authorized by Hugging Face; authentication alone cannot guarantee PR permission.

## Historical notebooks

Older notebooks remain at their original paths for reproducibility and existing links. Do not start new continuation runs from them. The original scratch experiment and contributor smoke test are distinct from canonical continuation.

## GGUF inference export

The final upload cell exports `bloom-f32.gguf` from the exact completed SafeTensors checkpoint and submits it in the same candidate PR alongside all existing artifacts. SafeTensors remains canonical for training. The exporter verifies the checkpoint and config hashes, accepts only a compatible GPT-2 byte-level BPE tokenizer, preserves full F32 weights and reads tensors back to check equality. The GPT-2 mapping is experimental; test actual llama.cpp tokenization and logits before calling it a supported runtime. If conversion fails, the original run remains saved and can be retried using upload recovery. Never retrain just to regenerate an export.

To add GGUF to the current merged Hugging Face model without training, open upload recovery, choose CURRENT_MODEL, authorize an HF write token and run the final cell. It pins the current revision and submits a GGUF-only PR. Review and merge only if its source-weight digest matches the current SafeTensors file. No other model artifacts are changed. Choose COMPLETED_RUN for the original recovery behavior.
