# Resource and cost ledger

PENDING final CU accounting; do not infer zero cost from missing balances.

| Run | Session | Requested / allocated GPU | Job start | Latest stage | Observed CU cost |
|---|---|---|---|---|---|
| E3-formal-independent-20261104-01 | exp-e3-formal | A100, root-confirmed ready | 2026-11-04 21:15:01 JST | RUNNING; dependency installation observed | Not yet measured |

Runtime code snapshot: `8d811623127094d0e13ea9252653e2269f1be191` (clean), configuration `configs/a100.json` (Qwen2.5-1.5B-Instruct). Job PID 1997. Server logs/status: `/srv/exp-runner/state/remote/E3-formal-independent-20261104-01`.

The actual GPU model/memory will be recorded by the formal launcher's `results/gpu.txt`; the installed stack by `results/env.txt`. Add allocation start/end UTC, observed CU/hour, account balances before/after, checkpoint location and final session end when available. No E3 paid API call was made by the controller. This live campaign is not yet a completed formal training run. Root controls its existing session; no automatic top-up or extra allocation is performed by this repository.
