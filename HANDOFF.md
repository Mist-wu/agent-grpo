# E3 formal handoff

## State and scope

The Mac agents now control the existing Colab runtime directly; server pi is stopped. E3 remains RUNNING in attempt `E3-formal-independent-20261104-02`. The latest downloaded formal result contained 630/6,000 base-direct rows (500 GSM8K, 130 HotpotQA). Of these, 496 batched rows averaged 0.975 seconds/question; long-run throughput varies with question length. Cold-SFT and GRPO have not yet been reached, and complete formal results do not exist. Dedicated preliminary-run scripts, results and debug configurations were removed at the user's request.

Existing implementation: strict Qwen tool protocol shared with E2; calculator, BM25 search and optional isolated Python; source-normalized dataset preparation; correct-trajectory collection; action-only SFT; clipped GRPO with frozen old/reference log probabilities; resumable checkpoints; final-test evaluation and three-seed summaries.

## Actual dependency failure and same-session recovery

Attempt `E3-formal-independent-20261104-01` failed with return code 1 after 246.05 seconds, after data preparation. The real error was `operator torchvision::nms does not exist`, imported through PEFT/Transformers Bloom; preinstalled torchvision binaries did not match pinned torch 2.6.0. The failed attempt's status/output and prepared data remain in place.

The text-only formal launcher now uninstalls incompatible optional `torchvision` and `torchaudio` after installing its pinned stack. It accepts `E3_DATA_DIR` to reuse the existing prepared data. Recovery uses the **same** `exp-e3-formal` A100 with a new attempt ID `E3-formal-independent-20261104-02`; its data points to `/content/experiment-runs/E3-formal-independent-20261104-01/workspace/data`, preserving the first attempt's files. No preliminary run or new GPU allocation is involved.

E2 reports that its pending main adapter is trained from the Glaive fallback (train 18,547 / dev 2,003 / test 2,000), not xLAM/60k. Final exported location will be recorded in E2 `results/main-seed42.adapter-receipt.json`, accompanied by `main-seed42.train.json`. Do not label the comparison as xLAM trained or completed before those real artifacts exist.

## Active Mac control

- Session: `exp-e3-formal`; endpoint `gpu-a100-s-ass1c0-fz5a9fduhlbu`; GPU `NVIDIA A100-SXM4-40GB`.
- Active run ID remains `E3-formal-independent-20261104-02`, wrapper PID `10606` after the planned batching recovery.
- Base-direct running source: `0f237cd117e6ffabe44f394b80534769edf01384`, clean. Later stages are deployed from `554b909bf1c05191027406a4412e3f417c448b08`, recorded by `workspace/.runtime-revision` and `source-updates.jsonl`.
- Configuration `configs/a100.json`, model `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Runtime root: `/content/experiment-runs/E3-formal-independent-20261104-02`; code under `workspace/`, main output at `output.log`.
- Local downloaded outputs: `results/live-attempt02/`. This is a partial live snapshot, not a completed report.
- Original attempt `01` failed at dependency import (rc1, 246.05 seconds). Attempt `02`'s initial serial segment was intentionally interrupted after 134 completed rows, then resumed in the same directory with batch16. `serial-status.json` records that planned interruption as rc143; `batching-transition.json` records the reason, retained-row count and SHA256. Neither record should be mistaken for the current running state.
- Base-direct uses batch16; subsequent tool evaluations use batch8. Cold collection batches up to eight questions, GRPO samples each existing group of eight jointly. Dataset sizes, 500 cold examples, seeds 42/43/44, 300 GRPO steps and the objective are unchanged. See `docs/execution.md` for numerical and timing details.
- E2 adapter comparison still waits for its actual export. The independent cold-SFT/GRPO pipeline does not wait for E2.

The Mac E3 agent owns this job. Do not submit another copy, allocate another session, or start server pi for this task. No SSH connection needs to remain open for the detached pipeline to continue. The current direct process was left running while later-stage source was updated.

Next progress commands from this repository on Mac:

```sh
colab download -s exp-e3-formal /content/experiment-runs/E3-formal-independent-20261104-02/status.json results/live-attempt02/status.json
colab download -s exp-e3-formal /content/experiment-runs/E3-formal-independent-20261104-02/output.log results/live-attempt02/output.log
colab download -s exp-e3-formal /content/experiment-runs/E3-formal-independent-20261104-02/workspace/results/eval-base-direct.jsonl results/live-attempt02/eval-base-direct.jsonl
```

For actual remote operations, the CLI's Jupyter WebSocket can fail while the detached job keeps running. Use the existing session's official SSH transport when needed:

```sh
ssh -o ConnectTimeout=60 -o "ProxyCommand=/Users/wu/.local/bin/colab ssh --proxy-mode -s exp-e3-formal -i /Users/wu/.ssh/id_ed25519" -i /Users/wu/.ssh/id_ed25519 root@colab-runtime
```

Use `colab upload`/`download` for file transfers; the SCP subsystem was unavailable. Do not restart the kernel or resubmit the experiment merely because a monitoring connection fails. `colab status` may show IDLE for the notebook kernel while detached model processes continue; inspect the actual job status/results.

Archive outputs before releasing the runtime, including partial outputs after a real failure. Attempt `02` has a `data` symlink to attempt `01`; archiving that symlink does not copy the actual dataset. Run these on the existing runtime with its staged private rclone environment, keeping large data off the AWS controller:

```sh
export RCLONE_HEADER='X-Goog-User-Project:pi-workspace-wu-4b7f3a'
rclone copy /content/experiment-runs/E3-formal-independent-20261104-01/workspace/data gdrive:exp/datasets/e3
rclone copy /content/experiment-runs/E3-formal-independent-20261104-02/workspace/results gdrive:exp/runs/E3/results
```

Only copy completed checkpoint directories while training is active. A whole-results copy is a final/paused-run archive; checkpoints are copied after saving. If Drive is unavailable, retain runtime/local files and report the failed copy. Native Drive mounting is not used.

## Runtime setup and formal data

The active runtime above is already assigned. The following commands describe setup for an explicitly scheduled recovery, not an instruction to launch a second copy. Do not allocate a new Colab runtime or run large-data preparation on the 2 GB AWS orchestrator. The shared transport starts commands in `/content/experiment-runs/<run_id>/workspace`; the example below uses run ID `e3-formal`. Controller supplies private rclone configuration separately:

```sh
cd /content/experiment-runs/e3-formal/workspace
export RCLONE_HEADER='X-Goog-User-Project:pi-workspace-wu-4b7f3a'
python -m pip install -e '.[train]'
python -m pip uninstall -y torchvision torchaudio
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
- `verify_completion.py` checks actual formal coverage, final steps/adapters and raw result consistency. It must remain incomplete until all real formal stages and comparisons finish; the current partial base-direct evaluation does not satisfy completion.
