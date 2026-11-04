# Scoring version 2

The initial formal GSM8K outputs exposed an actual extraction mismatch: the first completed 500 questions received 15/500 under the original strict parser, which required a bare numeric final field. Correct numeric conclusions with units or ordinary prose could receive zero. This observation is a diagnostic of version 1, not a claim about the model's benchmark accuracy.

Version `v2-qwen-gsm-last-number-hotpot-f1` fixes the mathematical answer extractor before cold-start collection or GRPO has begun. It adopts the last-number regular expression from [Qwen's published GSM8K chat evaluation](https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_chat_gsm8k.py). The number is normalized for commas and a leading plus sign and compared using Decimal exact equality; model output is never evaluated as Python. This is the published Qwen extraction rule with exact numeric equality, not a full reproduction of that evaluator's prompt or float-tolerance settings. Qwen2.5-Math's [parser](https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/parser.py) also uses a last-number fallback, and [lm-eval GSM8K](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml) separately reports strict and flexible extraction.

All math conditions use the same v2 extractor, including GSM-Symbolic. The original v1 result and a separate `<final>...</final>` format-compliance flag remain in every rescored record. This prevents answer accuracy from being confused with XML compliance. Last-number extraction can still accept a coincidentally correct final number in unfinished reasoning; truncation and v1 metrics remain visible. Do not selectively choose whichever metric is higher.

HotpotQA keeps the existing normalized answer-token F1. Its [official evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py) scores a submitted answer string and does not prescribe an arbitrary free-form paragraph extractor. No new text heuristic, model judge or manual label is used to increase that score.

Original generations and rewards remain unchanged at their original paths. Uniform offline rescoring writes a separate `results/scored-v2/` tree and records input SHA256, row count and whether the entire evaluation is present. Partial snapshots remain explicitly incomplete. The final report requires one scoring version across all conditions and also reports strict-v1 accuracy/F1 and format compliance.

```sh
mkdir -p results/scored-v2
for path in results/eval-*.jsonl; do
  python scripts/run.py rescore --data data/test.jsonl --input "$path" --out "results/scored-v2/$(basename "$path")"
done
python scripts/run.py report results/scored-v2/eval-*.jsonl
```

No generation is repeated, no question is removed, and training uses v2 from its first real collection/GRPO rollout. Version 1 files are historical evidence and must not be silently replaced.
