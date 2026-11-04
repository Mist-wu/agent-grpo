# E3 formal handoff

## State and scope

Implementation is ready for formal execution. No E3 GPU or paid API run has started, and formal metrics are absent. Dedicated preliminary-run scripts, results and debug configurations were removed at the user's request. Proceed directly to the formal pipeline.

Existing implementation: strict Qwen tool protocol shared with E2; calculator, BM25 search and optional isolated Python; source-normalized dataset preparation; correct-trajectory collection; action-only SFT; clipped GRPO with frozen old/reference log probabilities; resumable checkpoints; final-test evaluation and three-seed summaries.

## Runtime setup and formal data

On the allocated runtime, with controller-provided rclone credentials/quota header:

```sh
cd /content/agent-grpo
python -m pip install -e '.[train]'
python scripts/run.py prepare --out data
rclone copy data gdrive:exp/datasets/e3
export EXP_CHECKPOINT_REMOTE='gdrive:exp/runs/E3'
```

Use local `/content/agent-grpo/data` for training, not a Drive-mounted dataset. `prepare` writes 3,000 train + 100 dev rows each for GSM8K and HotpotQA, 500 final rows each, and full GSM-Symbolic main instances. It records actual dataset revision hashes, license and split provenance. Native Drive mounting is not part of unattended execution.

## Formal commands

Run A100 / 1.5B using `configs/a100.json`. If only L4 is available and root chooses the smaller model, use `configs/l4-full.json` and disclose 0.5B. Do not load the E2 1.5B adapter into a 0.5B model.

```sh
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl --out results/eval-base-direct.jsonl --condition base-direct
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl --out results/eval-base-tools.jsonl --condition base-tools
bash scripts/run_seeds.sh configs/a100.json
```

The seed script independently samples correct cold-start trajectories, trains a new SFT adapter, evaluates it, performs 300 GRPO steps and evaluates final adapters for seeds 42/43/44. These stages need not wait for E2. E2 is a separate baseline, never the cold-start initialization:

```sh
# E2 exports <E2_ARTIFACTS>/main-seed42/adapter; working default /content/e2-artifacts/main-seed42/adapter.
export E2_ADAPTER=/content/e2-artifacts/main-seed42/adapter
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl --out results/eval-e2.jsonl --condition e2-lora --adapter "$E2_ADAPTER"
python scripts/run.py report results/eval-*.jsonl
python scripts/verify_completion.py
rclone copy results gdrive:exp/runs/E3/results
```

## Simple checkpoints and resume

Saving is ordinary model/tokenizer output plus `torch.save` for optimizer/RNG; when `EXP_CHECKPOINT_REMOTE` is set, follow with `rclone copy`. No mandatory hook, receipt validation or durable-pointer state machine remains. Private `RCLONE_CONFIG` and approved `RCLONE_HEADER` come from root, never Git.

Remote layout: `gdrive:exp/runs/E3/grpo-42/checkpoint-0050`. Copies failing with nonzero exit keep the local checkpoint. Retry with `python scripts/run.py copy-checkpoint results/grpo-42/checkpoint-0050`.

```sh
rclone copy gdrive:exp/runs/E3/grpo-42/checkpoint-0050 results/grpo-42/checkpoint-0050
python scripts/run.py train --config configs/a100.json --data data/train.jsonl --dev data/dev.jsonl --out results/grpo-42 --seed 42 --adapter results/cold-42/adapter --resume results/grpo-42/checkpoint-0050
```

Also restore the original cold-SFT adapter or SFT checkpoint. SFT resumes using `sft --resume <checkpoint>` with the same original arguments. Collection and evaluation resume by task IDs. Archive collection data, evaluation rows and logs separately with rclone; checkpoint copying covers checkpoint directories only.

## Required result work

- Transfer E2 main adapter once available, run its independent comparison, and finish all seeds/conditions.
- Compare measured GSM8K with Qwen report Table 10 references: 1.5B 73.2%, 0.5B 49.6%; prompt/shot/extraction differences must be disclosed.
- Fill actual CU/session cost ledger and ten before/after trajectory pairs. Only report observed reward-exploitation examples, not invented ones.
- Keep Python disabled if namespace isolation is unavailable; report calculator/search-only capability.
- Preserve source licenses. GSM-Symbolic is CC-BY-NC-ND; do not publish modified data. Public dataset contamination remains a limitation.
- `verify_completion.py` checks actual formal coverage, final steps/adapters and raw result consistency. It currently correctly reports incomplete because formal runs have not occurred.
