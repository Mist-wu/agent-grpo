#!/usr/bin/env bash
# Run the independent E3 formal stages on the assigned GPU runtime.
set -euo pipefail
config=${1:-configs/a100.json}
python -m pip install -e '.[train]'
mkdir -p results
python -m pip freeze > results/env.txt
nvidia-smi > results/gpu.txt
python scripts/run.py prepare --out data
python scripts/run.py evaluate --config "$config" --data data/test.jsonl --out results/eval-base-direct.jsonl --condition base-direct
python scripts/run.py evaluate --config "$config" --data data/test.jsonl --out results/eval-base-tools.jsonl --condition base-tools
bash scripts/run_seeds.sh "$config"
