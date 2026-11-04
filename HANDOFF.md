# E3 live handoff

Observed 2026-11-04 23:34 JST. **Mac owns the job; server pi remains stopped.**

- Session `exp-e3-recovery-a100`, endpoint `gpu-a100-s-ass1c0-2kpagha8e7ubp`, A100-SXM4-40GB.
- Active run `E3-formal-independent-20261104-03`, foreground kernel PID **1266** (Mac Colab client PID80810). Runtime root `/content/experiment-runs/E3-formal-independent-20261104-03`; workspace is its `workspace/` directory.
- Current stage: restoring formal runtime dependencies/data, then seed42 cold collection batch128. **Actual recovered files contain91/500 correct trajectories and620 attempts.** The former runtime disappeared at23:27–23:28; its last observed350/500 counter is not recoverable trajectory data. SFT/GRPO have not started; no training checkpoint is yet claimed.
- Runtime source `1e9b43d341a99b0f0499bfd0e4e1835f6500a987`; GRPO generation batch64. Every60 seconds, at the next fully closed collection batch, actual accepted rows/attempts/executions/metadata copy to `gdrive:exp/runs/E3/data`. All steps, seeds and data are unchanged.
- Running command: `env EXP_CHECKPOINT_REMOTE=gdrive:exp/runs/E3 E3_TOOL_BATCH_SIZE=32 E3_DIRECT_BATCH_SIZE=32 E3_COLLECTION_BATCH_SIZE=128 bash scripts/run_priority.sh configs/a100.json training`.
- Model `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; BF16, Q8 × G8, 300 GRPO steps, seeds42/43/44. Math scoring v2 is documented in `docs/scoring.md`; original v1 scores remain saved.

## Next actual command

The running script automatically executes this after collection reaches500 and copies its closed outputs. **Do not launch a duplicate while the existing foreground kernel job is active.**

```sh
cd /content/experiment-runs/E3-formal-independent-20261104-03/workspace
python scripts/run.py sft --config configs/a100.json --data data/cold-42.jsonl --out results/cold-42 --seed 42
# Automatically follows SFT:
python scripts/run.py train --config configs/a100.json --data data/train.jsonl --dev data/dev.jsonl --out results/grpo-42 --seed 42 --adapter results/cold-42/adapter
```

Queue: finish seed42 collection/SFT/GRPO → seed42 core comparisons → seeds43/44 likewise → all deferred full evaluations. Every full condition still includes GSM500 + Hotpot500 + GSM-Symbolic5000. Core results have explicit1,000-row metadata; they are not reported as6,000 completed. The E2 comparison waits for its real `main-seed42.adapter-receipt.json` and `main-seed42.train.json`; its data is Glaive18,547/2,003/2,000, not xLAM60k.

## Read status without creating work

On Mac, download `status.json` and `output.log` from the run root with `colab download -s exp-e3-formal`. Current local snapshots/client output are in `results/recovery-03/`; earlier logs remain in `results/live-attempt02/`. The current foreground client is `colab exec --timeout604800`, executing `/tmp/e3-recovery03-foreground.py` through the existing common remote transport, without a detached worker. Coordinate any client transfer before Mac shutdown; never run this script twice.

On the existing AWS controller as `exp-runner`:

```sh
export PATH=/srv/exp-runner/runtime/python/bin:/usr/local/bin:/usr/bin:/bin
colab download -s exp-e3-formal /content/experiment-runs/E3-formal-independent-20261104-03/status.json /srv/exp-runner/state/remote/E3-formal-independent-20261104-03/status.json
colab download -s exp-e3-formal /content/experiment-runs/E3-formal-independent-20261104-03/output.log /srv/exp-runner/state/remote/E3-formal-independent-20261104-03/output.log
```

A notebook-kernel `IDLE` status does not describe the detached model process. WebSocket monitoring may disconnect without stopping training. Use actual status/results, and never allocate another GPU or resubmit on a monitoring error. Mac's working official SSH transport is:

```sh
ssh -o ConnectTimeout=60 -o "ProxyCommand=/Users/wu/.local/bin/colab ssh --proxy-mode -s exp-e3-formal -i /Users/wu/.ssh/id_ed25519" -i /Users/wu/.ssh/id_ed25519 root@colab-runtime
```

## Completed cache and durable artifacts

- Prepared6,000 train /200 dev /6,000 test data is restored directly from `gdrive:exp/datasets/e3` into the current workspace; previous Colab filesystems are no longer available. The unchanged train SHA256 is `606fd2dda50be0aa5dfc091617520a8850942c5fe6585ba9ab92941f4951bbab`.
- Direct evaluation retained **2,102 rows**: GSM500, Hotpot500, Symbolic1102. Core base-tools completed **1,000 rows**. These generations will be reused by task ID, not repeated.
- Actual successful rclone copies: `gdrive:exp/datasets/e3`; `gdrive:exp/runs/E3/snapshots/direct-before-priority`; `gdrive:exp/runs/E3/results/core/`; `gdrive:exp/runs/E3/snapshots/cold-before-batch128` (91 accepted /620 attempts at that snapshot).
- Core baseline report/raw-derived package: `gdrive:exp/runs/E3/analysis/core-baseline-artifacts.tgz` (4.26MB). Local report: `results/core-scored-v2/baseline-summary.md`; its JSON explicitly sets `experiment_complete=false` and preserves negative results, v1 scores, format compliance and confidence intervals.
- Source archives are under `gdrive:exp/runs/E3/source/`, including `beb13aa.tgz`, `63a0c1d.tgz` and the copied recovery source `1e9b43d.tgz`. Do not put datasets/models on the2GB AWS host; restore them directly into the assigned Colab runtime.
- No human annotation or review is claimed. Full trained comparisons, final CU accounting and ten before/after training pairs remain unfinished.

## Checkpoints and recovery

SFT saves `results/cold-42/checkpoint-0025`, `-0050`, `-0063`, plus `adapter`; GRPO saves every50 steps through `results/grpo-42/checkpoint-0300`. These are **future paths until files actually appear**. Closed checkpoints copy to `gdrive:exp/runs/E3/<run-directory>/<checkpoint-directory>`. Collection files copy to `.../data/` on completion; evaluation files to `.../results/` or `.../results/core/`. Private rclone configuration and the quota-project header are already staged in `/content/experiment-env.json`; never print its contents.

After a real failure, preserve the run status/log and resume only the stopped stage in the same workspace. Collection reuses accepted IDs/attempt counts; its changed batch segments are recorded in `.executions.jsonl`. SFT/GRPO resume uses the latest checkpoint that actually exists:

```sh
python scripts/run.py collect --config configs/a100.json --data data/train.jsonl --out data/cold-42.jsonl --seed 42 --target 500 --batch-size 128
# Example only after the named checkpoint exists:
python scripts/run.py sft --config configs/a100.json --data data/cold-42.jsonl --out results/cold-42 --seed 42 --resume results/cold-42/checkpoint-0025
python scripts/run.py train --config configs/a100.json --data data/train.jsonl --dev data/dev.jsonl --out results/grpo-42 --seed 42 --adapter results/cold-42/adapter --resume results/grpo-42/checkpoint-0050
```

Keep the existing foreground execution/exit-code transport for recovery; do not rerun the entire priority script over completed SFT directories. No new scheduling framework is needed. Earlier logs/transition evidence are retained in Mac snapshots and the recorded Drive packages where copied; do not assume an expired Colab filesystem survives. Actual incidents include: initial torchvision incompatibility, JSONL U+2028 reader failure, and archived `serial-*`, `before-priority-*`, `jsonl-failure-*`, `cold32-*` records. The23:27 runtime loss has no observed final process exit code. Details are in `docs/execution.md`.
