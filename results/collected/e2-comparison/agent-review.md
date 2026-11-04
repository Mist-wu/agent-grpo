# E2 adapter transfer to E3: completed comparison

Agent review, 2026-11-05 06:55 JST. `human_validation_complete=false`; `human_validation_required=false`. No new human labels were requested or created.

The actual E2 main seed42 adapter was evaluated once on all 6,000 E3 test tasks, with the existing E3 tool prompt/protocol, pinned Qwen2.5-1.5B revision, greedy decoding, BF16 and L4 batch16. `summary.json` contains actual scores and confidence intervals; `../e2-comparison-complete.tgz` retains original generations, execution metadata and separately rescored outputs. Run completed with exit0, including Drive copies. The L4 was released after collection.

| Dataset | n | Correctness / answer F1 |
|---|---:|---:|
| GSM8K | 500 | 0 |
| GSM-Symbolic | 5,000 | 0 |
| HotpotQA | 500 | 0.0008895104895104895 |

All three dataset summaries record zero mean tool calls, zero E3 `<final>` format compliance and zero truncation. The nonzero HotpotQA F1 is an answer-token-overlap score, not evidence of successful tool use. Preserve this negative transfer result; do not regenerate it merely because it is poor.

A bounded agent inspection of the first three saved GSM8K rows (`gsm8k:test:588`, `gsm8k:test:1214`, `gsm8k:test:57`) found the identical final text:

> I cannot complete this request with the available tools.

This is consistent with refusal/negative-example behavior failing to transfer to the E3 environment; the three examples alone do not quantify refusal prevalence or establish a causal mechanism. No retraining, prompt tuning or post-hoc protocol adaptation was performed. This is performance under the prespecified E3 prompt, not a claim that E2 is universally incapable of mathematical reasoning. E2's Glaive training data are model-generated; the separate agent review found a mislabeled negative and semantic near-duplicates. Those limitations were not silently repaired in this run.

Only one E2 adapter seed was used for this control. This condition is complete, but E3's trained three-seed comparisons, uncertainty and ten before/after trajectory analysis remain unfinished. The account usage readings are shared-account observations, not attributable E3 charges. The common-runner duration1238.0345 seconds includes dependency installation, restoration, evaluation, analysis and copies; it is not pure inference time.
