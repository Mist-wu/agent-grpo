#!/usr/bin/env bash
# Run the independent E3 formal stages on the assigned GPU runtime.
set -euo pipefail
config=${1:-configs/a100.json}
python -m pip install -e '.[train]'
# Text-only training: Colab's preinstalled vision/audio binaries target another torch.
python -m pip uninstall -y torchvision torchaudio
mkdir -p results
python -m pip freeze > results/env.txt
nvidia-smi > results/gpu.txt
if [ -n "${E3_DATA_DIR:-}" ] && [ ! -e data ]; then
  ln -s "$E3_DATA_DIR" data
fi
if [ ! -f data/sources.json ]; then
  python scripts/run.py prepare --out data
fi
python scripts/run.py evaluate --config "$config" --data data/test.jsonl --out results/eval-base-direct.jsonl --condition base-direct --batch-size "${E3_DIRECT_BATCH_SIZE:-16}"
python scripts/run.py evaluate --config "$config" --data data/test.jsonl --out results/eval-base-tools.jsonl --condition base-tools --batch-size "${E3_TOOL_BATCH_SIZE:-8}"
bash scripts/run_seeds.sh "$config"
