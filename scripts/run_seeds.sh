#!/usr/bin/env bash
# Requires allocated GPU and prepared local data; never allocates a runtime.
set -euo pipefail
config=${1:-configs/a100.json}
test_data=data/test.jsonl
eval_dir=results
collection_batch=32
if [[ ${2:-} == core ]]; then
  test_data=data/test-core.jsonl
  eval_dir=results/core
  collection_batch=${E3_COLLECTION_BATCH_SIZE:-128}
fi

train_grpo() {
  python scripts/run.py train --config "$config" --data data/train.jsonl --dev data/dev.jsonl --out "results/grpo-${seed}" --seed "$seed" --adapter "results/cold-${seed}/adapter"
}

for seed in 42 43 44; do
  python scripts/run.py collect --config "$config" --data data/train.jsonl --out "data/cold-${seed}.jsonl" --seed "$seed" --target 500 --batch-size "$collection_batch"
  python scripts/run.py sft --config "$config" --data "data/cold-${seed}.jsonl" --out "results/cold-${seed}" --seed "$seed"
  if [[ ${2:-} == core ]]; then train_grpo; fi
  python scripts/run.py evaluate --config "$config" --data "$test_data" --out "$eval_dir/eval-cold-${seed}.jsonl" --seed "$seed" --condition cold-sft --adapter "results/cold-${seed}/adapter" --batch-size "${E3_TOOL_BATCH_SIZE:-32}"
  if [[ ${2:-} != core ]]; then train_grpo; fi
  python scripts/run.py evaluate --config "$config" --data "$test_data" --out "$eval_dir/eval-grpo-${seed}.jsonl" --seed "$seed" --condition grpo --adapter "results/grpo-${seed}/checkpoint-0300" --batch-size "${E3_TOOL_BATCH_SIZE:-32}"
done
