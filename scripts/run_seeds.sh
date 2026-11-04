#!/usr/bin/env bash
# Requires allocated GPU and prepared local data; never allocates a runtime.
set -euo pipefail
config=${1:-configs/a100.json}
for seed in 42 43 44; do
  python scripts/run.py collect --config "$config" --data data/train.jsonl --out "data/cold-${seed}.jsonl" --seed "$seed" --target 500
  python scripts/run.py sft --config "$config" --data "data/cold-${seed}.jsonl" --out "results/cold-${seed}" --seed "$seed"
  python scripts/run.py evaluate --config "$config" --data data/test.jsonl --out "results/eval-cold-${seed}.jsonl" --seed "$seed" --condition cold-sft --adapter "results/cold-${seed}/adapter"
  python scripts/run.py train --config "$config" --data data/train.jsonl --dev data/dev.jsonl --out "results/grpo-${seed}" --seed "$seed" --adapter "results/cold-${seed}/adapter"
  python scripts/run.py evaluate --config "$config" --data data/test.jsonl --out "results/eval-grpo-${seed}.jsonl" --seed "$seed" --condition grpo --adapter "results/grpo-${seed}/checkpoint-0300"
done
