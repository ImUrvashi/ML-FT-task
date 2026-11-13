# NimbusPay ticket triage

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters). The selected adapter was trained with 4-bit NF4 QLoRA. Best dev result: **41.0% exact match, 88.9% mean field accuracy** (200 tickets).

The implementation is in `notebook.ipynb`. It audits the supplied training labels, runs both baselines and five controlled experiments, scores dev with the assignment's strict metric, and generates the 400 test predictions. `adapter/` contains the selected run A adapter in PEFT format; `predictions_test.jsonl` contains raw greedy generations.

## Reproduce on a free Colab T4

1. From the supplied assignment pack, create `candidate_pack_colab.zip` containing `SCHEMA.md`, `data/train.jsonl`, `data/dev.jsonl`, and `data/test_inputs.jsonl`, then put it in the top level of MyDrive. From the pack directory: `zip candidate_pack_colab.zip SCHEMA.md data/train.jsonl data/dev.jsonl data/test_inputs.jsonl`. The data archive is intentionally absent from Git.
2. Open `notebook.ipynb` in a fresh free Colab T4 runtime. Leave `RUN_EXPERIMENTS=False` and `SELECTED_RUN='A'` for final reproduction. Set `RUN_EXPERIMENTS=True` to rerun all comparisons.
3. Run top to bottom. The notebook uses the model's chat template, trains only on assistant tokens, and generates with greedy `model.generate(max_new_tokens=200)`.

The T4 run used 18,464,768 trainable LoRA parameters, a 4,096-token sequence limit, one epoch, gradient accumulation of eight, and seed 2026. The tokenizer measured a 99th-percentile training length of 1,936 tokens and a maximum of 2,215. The final fresh run used 5.93 GiB peak allocated GPU memory and took 13.64 minutes to train. Inference and artifact copying take additional time.

The repository excludes the supplied datasets and base-model weights. See `report.md` for the audit, experiment results, and error analysis.
