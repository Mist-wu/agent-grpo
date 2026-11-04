# Resource and cost ledger

PENDING final CU accounting; do not infer zero cost from missing balances.

## Latest observation: 2026-11-05 01:56 JST

Active attempt `E3-formal-independent-20261104-03` uses A100 session `exp-e3-recovery-a100`, source `1e9b43d341a99b0f0499bfd0e4e1835f6500a987`, and the unchanged `configs/a100.json`. Seed42 cold collection and 63-step SFT are complete; actual GRPO records now cover 34/300 steps. `colab usage` reported **118.46 CU remaining, 17.34 CU/hour, four active assignments**. These are shared-account observations, not measured E3-only cost. Final E3 CU attribution and runtime end remain unavailable. This observation and current `HANDOFF.md` supersede the historical stage descriptions below; preserve the existing job and output client.

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
