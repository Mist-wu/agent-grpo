# Multi-turn tool GRPO

Hand-written Qwen2.5 LoRA experiment: compare base models, E2 toolcall LoRA, a separate cold-start SFT, and SFT+GRPO on calculator/search tasks. Core baselines and the 6,000-item E2 control are collected; seed-42 SFT is complete, and GRPO can recover from checkpoint 200 of 300. Seeds 43/44 and final trained comparisons remain unfinished. No claimed training improvement is present. See the [current recovery inputs and measured time basis](docs/mac-resume.md).

## Formal execution

Use an already allocated GPU runtime assigned by root; run commands from its unpacked repository workspace, normally `/content/experiment-runs/<run_id>/workspace`. This repository does not allocate GPUs, start paid APIs or publish repositories.

```sh
python -m pip install -e '.[train]'
python -m pip uninstall -y torchvision torchaudio
python scripts/run.py prepare --out data
export RCLONE_HEADER='X-Goog-User-Project:pi-workspace-wu-4b7f3a'
export EXP_CHECKPOINT_REMOTE='gdrive:exp/runs/E3'
```

The data command resolves dataset revisions and writes source hashes, train/dev/test JSONL. GSM8K dev is held out from train; its official test is final test. HotpotQA uses distractor training data and reserves its released validation set for final test. GSM-Symbolic main includes all released instances. Large input data is staged on local runtime disk; archive source data with rclone.

Controller supplies private `RCLONE_CONFIG` and approved `RCLONE_HEADER` (quota-project header) to the runtime. `EXP_CHECKPOINT_REMOTE` is optional: when set, each completed local checkpoint is copied with `rclone copy` to `<prefix>/<run-directory>/<checkpoint-directory>`. A failed copy stops the command and leaves local files in place. There are no receipt state machines or required persistence gates.

Formal 1.5B configuration is `configs/a100.json`: 300 steps, 8 questions × 8 trajectories, two updates per sampled batch, three seeds. `configs/l4-full.json` is the disclosed 0.5B alternative with the same formal step count. Model sizes and adapter compatibility must be reported accurately.

```sh
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl --out results/eval-base-direct.jsonl --condition base-direct
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl --out results/eval-base-tools.jsonl --condition base-tools
# Set after E2 exports its 1.5B main adapter.
export E2_ADAPTER=/content/e2-artifacts/main-seed42/adapter
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl --out results/eval-e2.jsonl --condition e2-lora --adapter "$E2_ADAPTER"
bash scripts/run_seeds.sh configs/a100.json
mkdir -p results/scored-v2
for path in results/eval-*.jsonl; do
  python scripts/run.py rescore --data data/test.jsonl --input "$path" --out "results/scored-v2/$(basename "$path")"
done
python scripts/run.py report results/scored-v2/eval-*.jsonl
python scripts/verify_completion.py
```

The current A100 uses direct batch16 and subsequent tool/collection batches32; GRPO groups stay at eight responses and four groups are generated together. The seed script performs collection, SFT, SFT evaluation, GRPO, and GRPO evaluation for seeds 42/43/44. It can begin independently of E2; E2's adapter is only required for the separate E2 baseline and final complete comparison.

## Recovery

Checkpoints contain LoRA weights, optimizer/scaler state, CPU/CUDA RNG and sampling RNG. Local files are written with normal model/tokenizer saves and `torch.save`; optional rclone copy follows saving.

```sh
# Retrieve a checkpoint on a fresh runtime.
rclone copy gdrive:exp/runs/E3/grpo-42/checkpoint-0050 results/grpo-42/checkpoint-0050
# Also retrieve the original cold-SFT adapter/checkpoint and stage the same data.
python scripts/run.py train --config configs/a100.json --data data/train.jsonl --dev data/dev.jsonl --out results/grpo-42 --seed 42 --adapter results/cold-42/adapter --resume results/grpo-42/checkpoint-0050
# Retry an optional upload without loading a model.
python scripts/run.py copy-checkpoint results/grpo-42/checkpoint-0050
```

SFT checkpoints every 25 updates and supports `sft --resume <checkpoint>` with the original arguments. A saved SFT checkpoint itself can serve as a LoRA adapter. Collection resumes accepted task IDs and attempt counts; its restarted sampling stream is not claimed bitwise identical. Evaluation resumes completed task IDs. After an interrupted seed script, run the individual stage's resume command instead of restarting every stage.

## Method

- Frozen old-policy action log probabilities, clipped current/old ratios, and base-reference k3 KL implement the GRPO objective. LoRA/base dropout is zero and sampling temperature is 1 without top-k/top-p truncation.
- Loss is averaged over generated action tokens per trajectory, then over trajectories. Prompt and inserted tool tokens are not loss targets. One trajectory per gradient microbatch limits GPU memory.
- The reference for KL is the base model with LoRA disabled; cold-start SFT is the policy initialization, not the reference model.
- Reward is answer correctness plus one per-trajectory parsing bonus of 0.1, minus 0.05 per tool call; at most four calls. Math correctness uses the pinned Qwen last-number extraction rule and Decimal exact equality. Original strict-v1 scores and final-tag compliance are reported separately. HotpotQA keeps normalized answer-token F1. See [scoring versions](docs/scoring.md).
- Cold-start SFT uses correct trajectories from the training split. Checkpoint evaluations use held-out dev, never final test.
- Direct HotpotQA baseline sees full context; tool conditions see titles and use question-local BM25 retrieval. Report truncation and this prompt difference.
- Calculator uses bounded AST arithmetic. Python stays disabled unless Linux bubblewrap isolation passes; it never falls back to unisolated host execution.

Reports include raw rows, seeds, hardware, config, model/data revisions and commit. Train-derived results use three seeds; GSM-Symbolic uncertainty resamples template clusters. Public benchmark contamination remains a limitation. See [sources](docs/baselines.md), [cost ledger](docs/cost.md), and [handoff](HANDOFF.md).
