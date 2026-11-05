# E3 local continuation, 2026-11-05

The Mac now owns continuation; the server coordinator is inactive. GPU work is
blocked by the observed zero Colab balance and absence of active sessions.
The experiment remains incomplete. Review is by agents; human annotation and
relabeling are not requested.

## Actual completed and outstanding work

| Component | Collected evidence | Remaining |
|---|---|---|
| Core base-direct/base-tools | Each has the same 500 GSM8K and 500 HotpotQA test items, with paired bootstrap analysis | Full 6,000-item evaluations, reusing compatible prior generations |
| E2 adapter control | All 6,000 items, raw/scored outputs and report collected | No further execution |
| Seed 42 cold start | 500 accepted trajectories from 3,493 attempts; SFT completed 63 optimizer steps; adapter saved | Final cold-SFT evaluation |
| Seed 42 GRPO | Metrics through 214; model/optimizer/RNG checkpoint through 200; dev at 50/100/150/200 | Resume 100 steps from 200, then final evaluation |
| Seeds 43/44 | No completed cold/SFT/GRPO result | Cold collection, SFT, 300 GRPO steps each, final comparisons |
| Final analysis | Core baselines, E2 control and intermediate training summaries available | Three-seed uncertainty and trajectory/reward-hacking analysis after actual outputs exist |

The completed E2 control scored 0 on GSM8K and GSM-Symbolic, and mean answer F1
0.0008895 on HotpotQA under the fixed E3 prompt. This negative transfer finding
does not measure the effect of E3's unfinished GRPO training.

## Intermediate development results

Each development point uses 50 items per dataset. These are aggregate dev
observations, not final-test results or a paired significance analysis.

| GRPO step | GSM8K correctness | HotpotQA mean answer F1 |
|---|---:|---:|
| 50 | 0.74 | 0.407410 |
| 100 | 0.76 | 0.467603 |
| 150 | 0.76 | 0.503603 |
| 200 | 0.68 | 0.518853 |

The GSM8K value decreased at step 200. Keep that observation; no final trained
improvement or three-seed claim is supported yet. The source of this table is
`results/collected/seed42/grpo-42-dev-through-step200.jsonl`.

## Prepared recovery inputs

`python3 scripts/prepare_seed42_recovery.py` generates
`results/recovery-seed42/`: 200 canonical training rows, eight development rows,
the separately archived 14 observed but unsaved model steps, and a manifest with
input/output hashes and timing estimates. The union of the 200 and 14 training
rows was checked against the original 214 records without changing any values.

The training implementation restores `state["step"]` and starts the loop at 200,
so its next output is step 201 and it does not repeat development step 200.
It appends to existing logs. Therefore restore only the prepared canonical
history, not the 214-row snapshot; otherwise steps 201–214 would be counted twice.
The initial model/optimizer state beyond 200 was lost, despite the preserved
metrics. Full rollout trajectories from the old runtime were not collected.
The resumed `rollouts.jsonl` will cover steps 201 onward only, and its future
trajectory analysis must disclose that coverage.

Once an allocated A100 runtime is available, stage the original pinned source
`1e9b43d341a99b0f0499bfd0e4e1835f6500a987`, original config and Python/CUDA
dependencies using the existing runtime setup. Upload the prepared inputs and
`scripts/resume_seed42.sh`, retaining a new run receipt and real output client.
Supply the private rclone config and existing quota header through the runtime
environment. In that fresh source workspace run:

```sh
bash /content/resume_seed42.sh /content/e3-recovery-seed42
```

This script restores the unchanged dataset, original cold adapter, actual
checkpoint 200, and canonical logs, then resumes only seed-42 GRPO. It has not
been executed on a GPU. After successful completion, continue the pending
evaluations and seeds 43/44 using their individual stages. Do not restart
`run_priority.sh ... training` at seed 42: that loop also invokes its already
completed cold collection/SFT. The completed E2 control is already collected.

## Measured time basis

The four observed 50-step blocks average 134.88, 129.74, 132.97 and 123.91 seconds
per training step. Applying the minimum/maximum block means gives:

| Remaining GRPO computation | Estimate on comparable A100 execution |
|---|---:|
| Seed 42, steps 201–300 | 3.44–3.75 GPU hours |
| All seeds, 100 + 300 + 300 steps | 24.09–26.23 GPU hours |

These are conditional compute estimates, not confidence intervals or deadlines.
They exclude cold collection, SFT, checkpoint copies, dev, final evaluation and
setup. Parallel independent seeds can reduce elapsed time when sufficient GPUs
are available. Full completion time cannot be derived from training time alone.
