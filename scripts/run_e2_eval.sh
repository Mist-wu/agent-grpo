#!/usr/bin/env bash
# Required E3 comparison against the actual completed E2 main seed42 adapter.
set -euo pipefail
python -m pip install -e '.[train]'
python -m pip uninstall -y torchvision torchaudio
mkdir -p data results/e2-comparison results/scored-v2
python -m pip freeze > results/e2-comparison/env.txt
nvidia-smi > results/e2-comparison/gpu.txt
rclone copy gdrive:exp/datasets/e3 data
adapter=/content/e2-artifacts/main-seed42/adapter
rclone copy gdrive:exp/runs/E2/E2-main-resume-20261104-03-main-seed42-adapter "$adapter"
# Same pinned model, BF16, context and generation limits as the other E3 conditions.
# The actual GPU (L4 for this allocation) is captured above and in evaluation metadata.
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl \
  --out results/eval-e2-lora-42.jsonl --condition e2-lora --seed 42 \
  --adapter "$adapter" --batch-size 16
python scripts/run.py rescore --data data/test.jsonl --input results/eval-e2-lora-42.jsonl \
  --out results/scored-v2/eval-e2-lora-42.jsonl
python scripts/run.py report results/scored-v2/eval-e2-lora-42.jsonl \
  --out results/e2-comparison/summary.json
rclone copy results/e2-comparison gdrive:exp/runs/E3/e2-comparison
rclone copy results/scored-v2 gdrive:exp/runs/E3/results/scored-v2 --include 'eval-e2-lora-42*'
