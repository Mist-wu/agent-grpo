# E3 formal handoff

## State and scope

The independent formal E3 campaign has been submitted to the assigned A100 session. Latest observed state is RUNNING, with dependencies downloading/installing; actual GRPO training has not yet been observed and no complete formal results exist. Dedicated preliminary-run scripts, results and debug configurations were removed at the user's request. Proceed directly to the formal pipeline.

Existing implementation: strict Qwen tool protocol shared with E2; calculator, BM25 search and optional isolated Python; source-normalized dataset preparation; correct-trajectory collection; action-only SFT; clipped GRPO with frozen old/reference log probabilities; resumable checkpoints; final-test evaluation and three-seed summaries.

## Actual dependency failure and same-session recovery

Attempt `E3-formal-independent-20261104-01` failed with return code 1 after 246.05 seconds, after data preparation. The real error was `operator torchvision::nms does not exist`, imported through PEFT/Transformers Bloom; preinstalled torchvision binaries did not match pinned torch 2.6.0. The failed attempt's status/output and prepared data remain in place.

The text-only formal launcher now uninstalls incompatible optional `torchvision` and `torchaudio` after installing its pinned stack. It accepts `E3_DATA_DIR` to reuse the existing prepared data. Recovery uses the **same** `exp-e3-formal` A100 with a new attempt ID `E3-formal-independent-20261104-02`; its data points to `/content/experiment-runs/E3-formal-independent-20261104-01/workspace/data`, preserving the first attempt's files. No preliminary run or new GPU allocation is involved.

E2 reports that its pending main adapter is trained from the Glaive fallback (train 18,547 / dev 2,003 / test 2,000), not xLAM/60k. Final exported location will be recorded in E2 `results/main-seed42.adapter-receipt.json`, accompanied by `main-seed42.train.json`. Do not label the comparison as xLAM trained or completed before those real artifacts exist.

## Live campaign handoff

- Session: `exp-e3-formal`; A100 assignment confirmed by root, endpoint `gpu-a100-s-ass1c0-fz5a9fduhlbu`.
- Run ID: `E3-formal-independent-20261104-01`; submitted at 2026-11-04 21:15:01 JST.
- Runtime source: `8d811623127094d0e13ea9252653e2269f1be191`, clean. The launcher is executing the original `configs/a100.json` (1.5B), without a preliminary run.
- Status observed: `running`, PID `1997`, return code unset. `output.log` shows installation of the pinned training stack, including the real torch 2.6.0/CUDA dependencies. This is not a claim that training has already completed or that a checkpoint exists.
- Server receipt/status/log directory: `/srv/exp-runner/state/remote/E3-formal-independent-20261104-01`.
- Runtime directory: `/content/experiment-runs/E3-formal-independent-20261104-01`; code under `workspace/`, log at `output.log`.
- Submitted command: `env EXP_CHECKPOINT_REMOTE=gdrive:exp/runs/E3 bash scripts/run_formal.sh configs/a100.json`.
- Stages: formal data preparation, base-direct, base-tools, then seeds 42/43/44 each with cold-start collection/SFT, SFT evaluation, 300-step GRPO, and final GRPO evaluation. E2 adapter comparison is explicitly not part of this independent launch and remains pending the actual E2 export.

Root owns ongoing polling and allocation. Do not submit a second copy of this run or allocate another session. Local/remote code writing is frozen after this handoff; documentation-only commit does not change the launched source snapshot.

## Earlier 2026-11-04 coordination milestone

Both GPU assignments are occupied by E2 (`exp-preflight-l4`) and E5 (`exp-e5-formal`); the CPU assignment is E1 (`exp-e1-prep`). `colab usage` reported 197.13 CU, 3.96 CU/hour total, and three active assignments. Do not start another GPU job until a slot is released. E2's formal baseline receipt is `/srv/exp-runner/state/remote/E2-formal-baselines-20261104-01`; its actual seed-42 adapter export is not yet available. The E2 and E3 `tool_format.py` files were compared and are identical at E2 commit `00932e4` and E3 parent commit `eeea3ce`.

`scripts/run_formal.sh` sequences the independent formal preparation, both base evaluations and the three-seed cold-SFT/GRPO pipeline on an assigned GPU; it records the installed environment and GPU. It does not run the E2 comparison. Once capacity is available, stage private rclone configuration, then submit this real job detached (replace SESSION with the assigned session; use `configs/l4-full.json` only for the disclosed 0.5B L4 fallback):

```sh
python3 /srv/exp-runner/app/ops/stage_runtime.py SESSION
PYTHONPATH=/srv/exp-runner/app:/srv/exp-runner/app/common python3 -m exp_common.colab --repo /srv/exp-runner/repos/agent-grpo --session SESSION --run-id E3-formal-independent-20261104-01 --output-dir /srv/exp-runner/state/remote/E3-formal-independent-20261104-01 --env-file /content/experiment-env.json --artifact-path results --artifact-path data --detach -- bash scripts/run_formal.sh configs/a100.json
PYTHONPATH=/srv/exp-runner/app:/srv/exp-runner/app/common python3 -m exp_common.colab --poll --session SESSION --run-id E3-formal-independent-20261104-01 --output-dir /srv/exp-runner/state/remote/E3-formal-independent-20261104-01
PYTHONPATH=/srv/exp-runner/app:/srv/exp-runner/app/common python3 -m exp_common.collect --session SESSION --run-id E3-formal-independent-20261104-01 --output-dir /srv/exp-runner/state/remote/E3-formal-independent-20261104-01 results data
```

The launch above has now been submitted to `exp-e3-formal`; use only its polling/collection commands. Collect outputs into this repository before releasing a runtime, including partial outputs after failure; retain the receipt and local results if Drive is unavailable. For Drive checkpoint copies, set `EXP_CHECKPOINT_REMOTE` on the remote job only when Drive is available, as described below. Evaluate the actual E2 adapter independently once exported.

## Runtime setup and formal data

Run only on an existing runtime assigned by root, after its current job is finished. Do not allocate a new Colab runtime or run large-data preparation on the 2 GB AWS orchestrator. The shared transport starts commands in `/content/experiment-runs/<run_id>/workspace`; the example below uses run ID `e3-formal`. Controller supplies private rclone configuration separately:

```sh
cd /content/experiment-runs/e3-formal/workspace
export RCLONE_HEADER='X-Goog-User-Project:pi-workspace-wu-4b7f3a'
python -m pip install -e '.[train]'
python scripts/run.py prepare --out data
rclone copy data gdrive:exp/datasets/e3
export EXP_CHECKPOINT_REMOTE='gdrive:exp/runs/E3'
```

Use the workspace's local `data/` directory on the Colab runtime for training, not a Drive-mounted dataset. `prepare` writes 3,000 train + 100 dev rows each for GSM8K and HotpotQA, 500 final rows each, and full GSM-Symbolic main instances. It records actual dataset revision hashes, license and split provenance. Native Drive mounting is not part of unattended execution.

## Formal commands

Run A100 / 1.5B using `configs/a100.json`. If only L4 is available and root chooses the smaller model, use `configs/l4-full.json` and disclose 0.5B. Do not load the E2 1.5B adapter into a 0.5B model.

```sh
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl --out results/eval-base-direct.jsonl --condition base-direct
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl --out results/eval-base-tools.jsonl --condition base-tools
bash scripts/run_seeds.sh configs/a100.json
```

The seed script independently samples correct cold-start trajectories, trains a new SFT adapter, evaluates it, performs 300 GRPO steps and evaluates final adapters for seeds 42/43/44. These stages need not wait for E2. E2 is a separate baseline, never the cold-start initialization. Its local working path is not guaranteed to exist on an E3 runtime: wait for E2/root to provide the exported adapter location and copy those adapter files into the path below before this command. The completed E3 comparison waits for that real E2 artifact; do not create placeholder weights:

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
