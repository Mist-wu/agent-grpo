# E3 live handoff

## 2026-11-05 00:26 JST server observation

The existing `E3-formal-independent-20261104-03` on `exp-e3-recovery-a100` remains in seed 42 cold collection. Output reached **487/500 accepted from 3,414 attempts**; the latest completed Drive copy in that log covers **482 accepted / 3,381 attempts** at 00:25:04 JST. SFT and GRPO have not started; the running script should advance after collection. The existing output subscriber is active; do not resubmit. Shared account usage was 144.43 CU, 17.34 CU/hour across four assignments, not E3-specific. E2's final adapter is still pending. Poll using the commands below.

## 2026-11-05 00:19 JST server observation

Existing run `E3-formal-independent-20261104-03` continues seed 42 cold collection on `exp-e3-recovery-a100`. Its log reports **455/500 accepted from 3,171 attempts** and the corresponding completed Drive copy at 00:19:01 JST. SFT/GRPO have not started; the running script will transition to SFT after collection. The existing output subscriber remains active; do not resubmit. Shared account usage was 145.89 CU remaining, 17.34 CU/hour across four assignments, not E3-specific. E2's final adapter remains pending. Poll with the commands below.

## 2026-11-05 00:12 JST server observation

The existing `E3-formal-independent-20261104-03` on `exp-e3-recovery-a100` remains in seed 42 cold collection. Output reports **404/500 accepted from 2,815 attempts**, with a corresponding completed Drive copy at 00:12:13 JST. SFT/GRPO have not started. The existing output subscriber is active; do not resubmit the running job. Shared account usage observed: 147.34 CU, 17.34 CU/hour across four assignments, not E3-specific. The E2 comparison remains pending its real final adapter. Poll using the commands below.

## 2026-11-05 00:06 JST server observation

The existing `E3-formal-independent-20261104-03` on `exp-e3-recovery-a100` remains running. Output reports seed 42 cold collection reached **350/500 accepted from 2,462 attempts** and completed the corresponding Drive copy at 00:06:19 JST. This is the new runtime's actual saved collection, not the unrecoverable old 350-count log. SFT and GRPO have not started. The existing `exp-iopub-e3-recovery.service` remains active; no job was resubmitted. Shared account usage is 150.23 CU, 17.34 CU/hour across four assignments, not E3-specific cost. Poll via the commands below; E2's final adapter remains pending.

## 2026-11-05 00:00 JST server observation

Existing run `E3-formal-independent-20261104-03` reports running on `exp-e3-recovery-a100`. Its output reached **304/500 seed 42 accepted trajectories from 2,132 attempts**. The last completed Drive copy shown in this downloaded log is **276 accepted / 2,004 attempts**; a copy of the 304 batch was not yet reported at observation. SFT/GRPO have not started. Existing `exp-iopub-e3-recovery.service` remains active. Shared account usage was 151.69 CU, 17.34 CU/hour across four assignments (not E3-specific). E2's final adapter is still pending. No job was resubmitted; use the download commands below to observe continuation and Drive copy.

## 2026-11-04 23:54 JST server observation

Run `E3-formal-independent-20261104-03` remains in seed 42 cold collection: output reports **233/500 accepted from 1,664 attempts**, with the latest completed Drive copy at 23:53:44 JST. The existing output subscriber remains active. SFT/GRPO have not started. E2's final adapter still awaits the training run. Shared account `colab usage` reported 153.13 CU remaining and 17.34 CU/hour across four assignments; not an E3-only cost. Do not resubmit the active job. Poll using the commands below.

Collected the previously completed core baseline artifact package from Drive to `results/collected/core-baseline-artifacts.tgz` (also retained at `/srv/exp-runner/state/remote/E3-formal-independent-20261104-03/core-baseline-artifacts.tgz`), using `rclone copy gdrive:exp/runs/E3/analysis/core-baseline-artifacts.tgz /srv/exp-runner/state/remote/E3-formal-independent-20261104-03/`. It contains the real core direct/tools JSONL and metadata, their scored summaries, the derived direct rows and diagnostics; these are not full 6,000-row evaluations.

## 2026-11-04 23:47 JST server observation

Run `E3-formal-independent-20261104-03` on `exp-e3-recovery-a100` remains running (PID 1266); the actual seed 42 cold collection advanced to **184/500 accepted from 1,301 attempts** and reported a Drive copy at 23:47:32 JST. SFT and GRPO have not started. The existing `exp-iopub-e3-recovery.service` is active. At 23:47 shared account usage was 156.04 CU remaining, 17.34 CU/hour, four assignments, not E3-specific cost. E2's latest handoff reports seed42 step440/1288 at 23:41; its final adapter is not yet available. No job was resubmitted. The next download commands and receipt paths are immediately below.

## 2026-11-04 23:40 JST server observation

The existing foreground run on `exp-e3-recovery-a100` is still reporting `running` (PID 1266, source `1e9b43d`); its actual output advanced seed 42 cold collection to **123/500 accepted, 828 attempts**, and recorded a Drive copy at 23:39:25 JST. SFT and GRPO have not yet started. `exp-iopub-e3-recovery.service` is active and subscribes to the existing job; do not replace the job or its output client. Shared account usage observed at 23:40: 157.48 CU remaining, 17.34 CU/hour, four active assignments; this is not E3-specific cost. Original downloaded status and log snapshots are `/srv/exp-runner/state/remote/E3-formal-independent-20261104-03/status-latest.json` and `output-latest.log`.

```sh
colab download -s exp-e3-recovery-a100 /content/experiment-runs/E3-formal-independent-20261104-03/status.json /srv/exp-runner/state/remote/E3-formal-independent-20261104-03/status-latest.json
colab download -s exp-e3-recovery-a100 /content/experiment-runs/E3-formal-independent-20261104-03/output.log /srv/exp-runner/state/remote/E3-formal-independent-20261104-03/output-latest.log
```

The following is the historical 23:34 handoff, superseded by the observation above where it differs.

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
colab download -s exp-e3-recovery-a100 /content/experiment-runs/E3-formal-independent-20261104-03/status.json /srv/exp-runner/state/remote/E3-formal-independent-20261104-03/status.json
colab download -s exp-e3-recovery-a100 /content/experiment-runs/E3-formal-independent-20261104-03/output.log /srv/exp-runner/state/remote/E3-formal-independent-20261104-03/output.log
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
