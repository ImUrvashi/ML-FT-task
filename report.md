# NimbusPay ticket triage: results

## Audit and cleaning

I checked 1,445 supplied training rows for JSON shape, types, formats, rule consistency, duplicate text, and train/dev overlap. The audit found 25 incomplete JSON labels, 70 labels using legacy `prio` instead of `priority`, 30 legacy dates, 20 transaction IDs needing normalization, and 135 wrong or missing priorities (including the 70 legacy rows). Thirty-two training tickets matched dev text after trimming whitespace; these were excluded from **all** training variants to keep dev evaluation uncontaminated. Sixty-one repeated training tickets were removed from the cleaned variant. Some rows had multiple issues.

The reproducible cleaning cell writes 1,327 cleaned rows and `cleaning_log.csv` with 273 distinct IDs: 155 fixes and 118 drops. I dropped incomplete JSON labels because I could not recover all eight fields from them with confidence. I fixed legacy keys, dates, IDs, and rule-derived priorities where the ticket and schema supported a specific correction. I removed the 32 dev-text overlaps from the raw comparison too, leaving 1,413 rows; otherwise that comparison could benefit from seeing dev tickets. Priority repair still trusts the existing category and amount labels, so some semantic errors may remain. The original input files were never overwritten.

## Model and controlled experiments

I used `Qwen/Qwen2.5-1.5B-Instruct` with PEFT LoRA because it fits the 4B model cap and permits both 16-bit and 4-bit runs on a free Colab T4. The selected 4-bit NF4 run had 18,464,768 trainable parameters, below the 50M limit. The tokenizer measured median/p95/p99/max training lengths of 168/226/1,936/2,215 tokens; 4,096 covered every cleaned row. Training used the model chat template, assistant-token-only loss, rank 16, alpha 32, dropout 0.05, one epoch, effective batch size eight, seed 2026, and learning rate `1e-4` except run D. All dev/test predictions used greedy `model.generate`, 200 maximum new tokens, and whitespace trimming only.

Both untrained baselines scored 0% exact and 0% mean field accuracy under the strict scorer. The fixed-prompt base model generated other ticket schemas; the rules-in-prompt base model often generated Markdown code fences, so its output was not a bare JSON object. These are measured formatting failures, not estimates of field reasoning quality.

The Colab setup needed one correction before training: its PyTorch build used CUDA 13.0, while `bitsandbytes==0.46.1` lacked the matching `libbitsandbytes_cuda130.so` binary. I switched to `bitsandbytes==0.50.2` and added a check for that binary before loading the model.

| Run | One change from A | Dev exact | Mean field | Peak GPU GiB | Train min |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Cleaned 4-bit QLoRA reference | **41.0%** | 88.9% | 5.93 | 14.01 |
| B | Leakage-safe raw training labels | 37.5% | 88.5% | 5.93 | 14.97 |
| C | 16-bit LoRA base weights | 34.5% | 88.1% | 7.12 | 9.61 |
| D | Learning rate `5e-5` | 34.5% | 86.1% | 5.93 | 13.72 |
| E | One extra copy of each cleaned fraud row | 40.0% | **89.3%** | 5.93 | 14.58 |

I selected A because exact match is the main dev measure: it reached 41.0%, versus E's 40.0%, although E had slightly better mean field accuracy. Run C's shorter measured training time is reported as observed. A fresh T4 run with `RUN_EXPERIMENTS=False` repeated A's 41.0% exact and 88.9% field scores in 13.64 training minutes. Five test outputs changed between training sessions, so I kept the final run's adapter and predictions together. All 400 test IDs are present and their raw outputs parse as JSON with eight keys. Two outputs have invalid channel values; I left the raw generations unchanged because the assignment forbids output repair.

## Dev error analysis

Run A's weakest fields were `priority` (55.0%) and `needs_human` (77.5%). Category was 97.0%; language was 100%. Exact match was weakest for `account_access` (15.8%), `payment_failed` (28.3%), and `refund` (29.1%). The examples below show repeated mistakes on legal threats, amount thresholds, and prior support contacts; they also show separate amount and date conversion errors.

| Dev ID | What the output missed |
| --- | --- |
| `dv-00003` | “Consumer court” requires P1/human true, but the output has P3/false. It also resolves “day before yesterday” as one day before receipt. |
| `dv-00005` | The ₹49,680 failed transfer needs P2 (over ₹5,000, below ₹50,000); it outputs P3 despite extracting the amount. |
| `dv-00011` | The ₹84,137 refund crosses the P1 threshold and therefore needs human review; the output has P3/false. |
| `dv-00013` | The ₹55 declined card needs P3 and `card`; the output has P2 and null channel despite extracting ₹55 correctly. The reason for P2 is unclear. |
| `dv-00014` | “Again and again” says support has already been contacted; the output misses `needs_human=true`. |
| `dv-00018` | `4.2k` rupees is 420,000 paise; the output has 4,200,000, a tenfold conversion error. |
| `dv-00022` | For unauthorised card payments, the output keeps fraud/P1 but gives 1,677 instead of 167,700 paise, null instead of `card`, and human false. |
| `dv-00026` | “Already written 4 times” calls for human review; the output has false. |
| `dv-00031` | “Consumer court” requires P1/human true even though the refund is only ₹824; the output has P3/false. |
| `dv-00047` | OTP failure prevents sign-in, so `account_access` requires P2; the output keeps the category but gives P3. |

With another week, I would first check the remaining category and amount labels by hand, then add training examples drawn only from the train data and schema for the ₹5,000/₹50,000 boundaries and legal/repeat-contact wording. I would compare longer training and multiple seeds, and rerun the permitted rules-prompt baseline with an explicit bare-JSON instruction. I would keep dev and test labels out of training.

**Tools used:** An AI assistant helped draft and review the audit/training notebook, troubleshoot Colab package versions, and organize the error analysis. Counts and scores above were checked against the supplied data, saved T4 logs, and `score.py`.
