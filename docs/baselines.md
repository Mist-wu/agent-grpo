# E3 sources and baseline comparison

Verified 2026-11-04. This file is a source map, not a claim of reproduced scores.

| Source | What is used | Boundary |
|---|---|---|
| [DeepSeekMath §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1) | group-relative advantages, clipped token ratios, separate KL regularizer | This code uses two optimization epochs per rollout batch so clipping can activate. A single step would begin with ratios one. |
| [Qwen2.5 technical report](https://arxiv.org/abs/2412.15115) | public reference for the 1.5B model | [Table 10](https://arxiv.org/html/2412.15115v2#S5.T10) reports GSM8K **73.2% for Qwen2.5-1.5B-Instruct** and **49.6% for 0.5B-Instruct**. These are published references, not our results. Exact prompt/shot and extraction alignment still must be checked before treating the difference as an implementation bug. |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | model and native chat template | Resolved model commit is stored for each run; tokenizer/tool protocol compatibility tested separately. |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k/blob/main/README.md) | main, train/test, MIT | There is no validation split. Hold dev out of train; final test stays untouched. |
| [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa/blob/main/README.md) | distractor, train/validation, CC-BY-SA-4.0 | Hold dev out of train; released validation becomes this experiment's final test. Preserve source titles, paragraphs and license. |
| [GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic/blob/main/README.md) | main test, CC-BY-NC-ND-4.0 | Dataset is **apple/GSM-Symbolic**, not openai. Preserve template and instance IDs; cluster CI by template. Do not republish modified dataset. |

## Comparisons

Five conditions: base-direct, base-tools, E2 toolcall LoRA, new cold-start SFT, SFT+GRPO. The cold-start adapter is trained from correct **training** trajectories; E2 adapter is a separate comparison. Baselines use seed 42; cold-SFT and GRPO use 42/43/44. Qwen greedy decoding is used for final evaluation.

HotpotQA base-direct sees full question context; tool conditions see titles and retrieve question-local context via BM25. This is a meaningful direct-context versus tool-retrieval comparison; report prompt length and truncation. Do not label a closed-book comparison as an isolated training effect.

The reference policy for KL is the **base model with LoRA disabled**, including when the policy starts from cold-SFT. Thus it is not a frozen SFT reference. LoRA and base dropout are disabled; rollout temperature is exactly 1 with no top-k/top-p/repetition transformation, matching stored old-policy log probabilities.

## Completion status

A formal A100 campaign is running. Its completed 500-question GSM8K base-direct subset scored 319/500 (63.8%) under v2 math extraction; the original strict-v1 score was 15/500, and only 2/500 outputs conformed to the final-tag format. All original outputs/scores remain available. The complete 6,000-question condition and all training comparisons are unfinished. Our zero-shot tool-oriented system prompt, 512-token turn budget and subset differ from published benchmark settings, so this is not a reproduced Table 10 score. See [scoring versions](scoring.md). Public GSM8K/HotpotQA contamination is possible; GSM-Symbolic is derived from GSM8K and is a complementary robustness check, not proof of contamination absence.
