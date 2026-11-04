# Formal execution changes

On 2026-11-04, control returned to the Mac agents; server pi is stopped. The existing E3 A100 and run directory are retained.

The original base-direct path completed 91 actual GSM8K rows in 798.50 seconds of recorded generation time (8.77 seconds/question). Extrapolating that initial rate to the remaining 5,909 questions gives 14.4 hours for this condition alone; this estimate does not cover other datasets or training.

The formal launcher now requests base-direct batches of 16. It preserves the model/revision, test data, prompt construction, greedy decoding, EOS, 512 new-token limit and 4,096 total-token budget. Prompts with less than 512 tokens of remaining context use the existing single-item path. Inputs use left padding with attention masks. Outputs retain the original unpadded prompt and real generated tokens through EOS; batch completion padding is excluded from trajectories and action masks.

Each resumed execution records its source commit, batch size and existing-row count in the evaluation `.executions.jsonl`; existing rows and their original metadata stay intact. New rows record the execution commit, batch size and wall time. `seconds` is amortized batch time, not single-request latency. BF16 batching may change some greedy branches through numerical rounding; no token-for-token identity or speedup is claimed without actual observations.

Tool evaluation and training remain unchanged in this first optimization. No new runtime or preliminary model run is involved. The change follows the [Transformers 4.51.3 batching guidance](https://huggingface.co/docs/transformers/v4.51.3/llm_tutorial#padding-side) and the pinned generation implementation.

The first two formal direct batches completed 32 new questions in 24.8064 seconds (0.7752 amortized seconds/question), while the 134 previously completed rows were retained. This small initial observation is roughly 11.3 times the original throughput; it is not an estimate for every dataset or training stage.

Subsequent tool-enabled evaluations now request batches of 8. Each trajectory independently stops at its EOS, closing tool-call delimiter, or remaining context budget; a stopping-criteria callback records its exact first stopping length before other rows finish. Per-item tool limits, tool execution, parsing, rewards and tool-output loss masks remain unchanged. Active trajectories are regrouped at each model turn. The base-direct process already running is left untouched. The controller writes `.runtime-revision` with the deployed tracked-source commit so later Python stages record the current code rather than inheriting the old wrapper commit. Existing rows keep their earlier provenance.
