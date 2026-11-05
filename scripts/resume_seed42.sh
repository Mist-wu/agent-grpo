#!/usr/bin/env bash
# Execute once in a fresh, allocated runtime workspace using source 1e9b43d.
# Argument: uploaded directory produced by prepare_seed42_recovery.py.
set -euo pipefail
recovery_inputs=$1
export EXP_CHECKPOINT_REMOTE=gdrive:exp/runs/E3
rclone copy gdrive:exp/datasets/e3 data
rclone copy gdrive:exp/runs/E3/cold-42/adapter results/cold-42/adapter
rclone copy gdrive:exp/runs/E3/grpo-42/checkpoint-0200 results/grpo-42/checkpoint-0200
cp "$recovery_inputs/train.jsonl" results/grpo-42/train.jsonl
cp "$recovery_inputs/dev.jsonl" results/grpo-42/dev.jsonl
mkdir -p results/recovery-provenance
cp "$recovery_inputs/manifest.json" "$recovery_inputs/observed-lost-segment-201-214.jsonl" results/recovery-provenance/
python scripts/run.py train --config configs/a100.json --data data/train.jsonl \
  --dev data/dev.jsonl --out results/grpo-42 --seed 42 \
  --adapter results/cold-42/adapter --resume results/grpo-42/checkpoint-0200
