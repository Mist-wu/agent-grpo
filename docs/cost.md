# Resource and cost ledger

PENDING final CU accounting; do not infer zero cost from missing balances.

## Latest observation: 2026-11-05 01:56 JST

Active attempt `E3-formal-independent-20261104-03` uses A100 session `exp-e3-recovery-a100`, source `1e9b43d341a99b0f0499bfd0e4e1835f6500a987`, and the unchanged `configs/a100.json`. Seed42 cold collection and 63-step SFT are complete; actual GRPO records now cover 34/300 steps. `colab usage` reported **118.46 CU remaining, 17.34 CU/hour, four active assignments**. These are shared-account observations, not measured E3-only cost. Final E3 CU attribution and runtime end remain unavailable. This observation and current `HANDOFF.md` supersede the historical stage descriptions below; preserve the existing job and output client.

2026-11-05 02:04 JST: the same run has produced 37/300 seed42 GRPO steps. `colab usage`: 115.58 CU remaining, 17.34 CU/hour, four active assignments; shared-account values only, not E3-attributed charges. No runtime end or final E3 cost yet.

2026-11-05 02:12 JST: the same run produced 41/300 seed42 GRPO steps. `colab usage`: 114.12 CU remaining, 17.34 CU/hour, four active assignments; these remain shared-account values, not E3-only cost.

2026-11-05 02:18 JST: same run reached 44/300 seed42 GRPO steps. `colab usage`: 111.23 CU remaining, 17.34 CU/hour, four active assignments. These are shared-account values, not E3-attributed cost; final accounting remains pending.

2026-11-05 02:25 JST: same run produced 47/300 seed42 GRPO steps. `colab usage`: 109.77 CU remaining, 17.34 CU/hour, four active assignments; shared-account observations, not E3-only cost.

2026-11-05 02:32 JST: seed42 GRPO reached step50 and checkpoint-0050 was observed on Drive with adapter and optimizer/RNG state. `colab usage`: 108.35 CU remaining, 17.34 CU/hour, four assignments; shared-account values only. Training and final E3 accounting remain incomplete.

2026-11-05 02:39 JST: `colab usage` again reported 108.35 CU remaining, 17.34 CU/hour, four assignments. Output still ends at GRPO step50 and dev.jsonl is not yet available; no extra GPU was allocated (L4 and A100 comparison requests refused). Account readings are not E3-only charges.

2026-11-05 02:46 JST: step50 GSM8K dev result (50 questions) collected; HotpotQA dev not yet observed. `colab usage`: 106.87 CU remaining, 17.34 CU/hour, four assignments. Shared-account readings only, not E3-attributed cost; existing run preserved.

2026-11-05 02:54 JST: step50 dev evaluation completed for both datasets and actual training reached step51. `colab usage`: 103.95 CU remaining, 17.34 CU/hour, four assignments; shared-account readings, not E3-attributed charges. Existing run and output client preserved.

2026-11-05 03:01 JST: seed42 GRPO reached step55/300. `colab usage`: 102.51 CU remaining, 17.34 CU/hour, four assignments; shared-account observations, not E3-attributed charges. Training remains active.

2026-11-05 03:07 JST: seed42 GRPO reached step57/300. `colab usage`: 99.62 CU remaining, 17.34 CU/hour, four assignments; shared-account values only, not E3-attributed cost. Existing run preserved.

2026-11-05 03:13 JST: seed42 GRPO reached step60/300. `colab usage`: 98.20 CU remaining, 17.34 CU/hour, four assignments; shared-account readings, not E3-only cost. Existing job and output client preserved.

2026-11-05 03:20 JST: seed42 GRPO reached step63/300. `colab usage`: 96.73 CU remaining, 17.34 CU/hour, four assignments; shared-account observations only, not E3-attributed cost. Existing run preserved.

2026-11-05 03:26 JST: seed42 GRPO reached step66/300. `colab usage`: 95.29 CU remaining, 17.34 CU/hour, four assignments; shared-account values, not E3-only cost. Existing job and output client preserved.

2026-11-05 03:35 JST: seed42 GRPO reached step70/300. `colab usage`: 92.42 CU remaining, 17.34 CU/hour, four assignments; shared-account readings, not E3-only charges. Independent comparison L4/A100 allocations both refused; existing run preserved.

2026-11-05 03:41 JST: seed42 GRPO reached step73/300. `colab usage` again reported 92.42 CU remaining, 17.34 CU/hour, four assignments; shared-account observations, not E3-only cost. Do not interpret an unchanged account reading as zero incremental cost.

2026-11-05 03:48 JST: seed42 GRPO reached step76/300. `colab usage`: 89.52 CU remaining, 17.34 CU/hour, four assignments; shared-account readings only, not E3-attributed cost. Existing run preserved.

2026-11-05 03:55 JST: seed42 GRPO reached step79/300. `colab usage`: 88.09 CU remaining, 17.34 CU/hour, four assignments; shared-account readings only, not E3-attributed charges. Existing run preserved.

2026-11-05 04:03 JST: seed42 GRPO reached step83/300. `colab usage`: 85.20 CU remaining, 17.34 CU/hour, four assignments; shared-account observations only, not E3-attributed charges. Existing job/output client preserved.

2026-11-05 04:11 JST: seed42 GRPO reached step87/300. `colab usage`: 83.77 CU remaining, 17.34 CU/hour, four assignments; shared-account observations only, not E3-attributed cost. Existing run preserved.

2026-11-05 04:17 JST: seed42 GRPO reached step90/300. `colab usage`: 82.30 CU remaining, 17.34 CU/hour, four assignments; shared-account observations only, not E3-attributed cost. Existing job/output client preserved.

2026-11-05 04:23 JST: seed42 GRPO reached step93/300. `colab usage`: 79.42 CU remaining, 17.34 CU/hour, four assignments; shared-account observations only, not E3-attributed cost. Existing run preserved.

2026-11-05 04:30 JST: seed42 GRPO reached step96/300. `colab usage`: 77.97 CU remaining, 17.34 CU/hour, four assignments; shared-account readings only, not E3-attributed cost. Existing run preserved.

2026-11-05 04:37 JST: seed42 GRPO reached step99/300. `colab usage`: 76.53 CU remaining, 17.34 CU/hour, four assignments; shared-account readings only, not E3-attributed cost. Existing run preserved.

2026-11-05 04:43 JST: seed42 GRPO reached step100/300 and checkpoint-0100 was observed on Drive with adapter and training state. `colab usage`: 73.65 CU remaining, 17.34 CU/hour, four assignments; shared-account observations only, not E3-attributed cost. Existing run preserved.

2026-11-05 04:50 JST: output still ends at step100 and no step100 dev result observed. `colab usage`: 72.20 CU remaining, 17.34 CU/hour, four assignments; shared-account readings only. Recorded first100 training-step times sum to 13,231.0475 seconds, excluding dev evaluation/checkpoint copying; not total runtime or E3 CU cost.

2026-11-05 04:57 JST: step100 GSM8K dev result collected; HotpotQA dev pending. `colab usage`: 70.76 CU remaining, 17.34 CU/hour, four assignments; shared-account readings only, not E3-attributed cost. Existing run preserved.

2026-11-05 05:04 JST: step100 dev evaluation complete for both datasets; actual training reached step101/300. `colab usage`: 67.86 CU remaining, 17.34 CU/hour, four assignments; shared-account readings only, not E3-attributed cost. Existing run preserved.

2026-11-05 05:11 JST: seed42 GRPO reached step104/300. `colab usage`: 66.41 CU remaining, 17.34 CU/hour, four assignments; shared-account readings only, not E3-attributed cost. Existing run preserved.

2026-11-05 05:19 JST: seed42 GRPO reached step107/300. `colab usage`: 63.52 CU remaining, 17.34 CU/hour, four assignments; shared-account readings only, not E3-attributed cost. Independent comparison L4/A100 allocations refused; existing run preserved.

2026-11-05 05:25 JST: seed42 GRPO reached step110/300. `colab usage`: 62.08 CU remaining, 17.34 CU/hour, four assignments; shared-account readings only, not E3-attributed cost. Existing run preserved.

2026-11-05 05:31 JST: seed42 GRPO reached step113/300. `colab usage`: 60.64 CU remaining, 17.34 CU/hour, four assignments; shared-account readings only, not E3-attributed cost. Existing run preserved.

## Historical observations

| Run | Session | Requested / allocated GPU | Job start | Latest stage | Observed CU cost |
|---|---|---|---|---|---|
| E3-formal-independent-20261104-01 | exp-e3-formal | A100 | 2026-11-04 21:15:01 JST | FAILED rc1 after 246.05 s; incompatible torchvision import after data preparation | Not yet measured |
| E3-formal-independent-20261104-02 | exp-e3-formal | NVIDIA A100-SXM4-40GB | 2026-11-04 21:28:33 JST | RUNNING priority core base-tools; direct paused with 2,102/6,000 retained rows | Not yet measured |

Attempt 01 runtime source was `8d811623127094d0e13ea9252653e2269f1be191`. Attempt 02 source is `6eb668b51a14e9f3e2ab4c1dcf8d993b1b9a615c` (clean), using `configs/a100.json` (Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`). The initial attempt 02 wrapper PID was 5521; after a planned interruption preserving 134 rows, the same attempt resumed under PID 10606 with base-direct source `0f237cd`. Later stages use `554b909`; the same session was reused, and prepared data from attempt 01 was retained.

At the latest observation the A100 reported 40,960 MiB total memory, 3,690 MiB in use and 28% GPU utilization. The runtime records GPU details in `results/gpu.txt` and the installed stack in `results/env.txt`. Server status/log directories are `/srv/exp-runner/state/remote/E3-formal-independent-20261104-01` and `/srv/exp-runner/state/remote/E3-formal-independent-20261104-02`.

Add allocation start/end UTC, observed CU/hour, account balances before/after, checkpoint location and final session end when available. No E3 paid API call was made by the controller. The formal base evaluation is in progress; cold-SFT and GRPO training have not yet been reached. Root controls the existing session; this repository does not purchase more credit or allocate another GPU.

The first 496 batched formal rows averaged 0.975 seconds/question; the first 91 serial rows averaged 8.77 seconds/question. This is partial throughput evidence, not a completed benchmark comparison. Whole-account `colab usage` during recovery reported 187.69 CU remaining, 18.57 CU/hour and six assignments; these shared-account values must not be attributed entirely to E3.

The same attempt was reordered at 2026-11-04 22:18:43 JST under source `9a3c642` and wrapper PID 18199. The base-direct segment had generated 2,102 rows before planned interruption, archived with rc143 and copied to Drive. The first 128 formal core tool rows used batch32 at 1.001 amortized seconds/question; their target is 1,000, with the remaining full evaluation queued. No new session was allocated.

The first cold32 recovery segment (`cab986c`) ran 953.21 seconds and retained 91 accepted trajectories / 620 attempts before a planned throughput adjustment. The same attempt resumed as `beb13aa` under PID26936 at 2026-11-04 22:53:21 JST. Actual cold128 memory was 16,838/40,960 MiB with 98% utilization, and 123/500 accepted trajectories were observed. Full GRPO training has not yet started; no completed-training cost or accuracy is claimed.
