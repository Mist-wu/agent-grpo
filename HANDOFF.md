# E3 handoff

## Current state

Repository: `agent-grpo`. Implementation covers data preparation, tool rollouts, rejection-sampled cold-start data, assistant-action-token SFT, clipped GRPO, final evaluation, three-seed aggregation and independent completion verification. No GPU session or paid API was started. `runner-task.json` is present and correctly reports incomplete until real formal results exist.

Validated locally:

- 16 tests: strict parser, malicious arithmetic rejection, masked tokens, source splitting, reward metrics, clipped ratio/KL numerics, detached old/reference gradients, real tiny Qwen2+LoRA backward and checkpoint round-trip.
- Native Qwen2.5-0.5B tool-response suffix matches `apply_chat_template` (revision `7ae557604adf67be50417f59c2c2f167def9a775`).
- Two real 0.5B CPU questions called calculator/search successfully; trace is `results/smoke-cpu.json`, explicitly a smoke fixture rather than a benchmark. Post-load inference took a few seconds per question with one CPU thread.
- Smoke output exposed strict numeric-extraction rejection of correct prose. Explicit “answer/result is N” and boxed answers are now supported; arbitrary numbers from working remain rejected. HotpotQA token F1 remains sensitive to extra prose, as recorded by raw output.

## Next commands

Use an allocated Colab runtime managed by the shared runner. Do not create a second runtime from this repository. On a CPU runtime, prepare and persist data to Drive:

```sh
python -m pip install -e '.[train]'
python scripts/run.py prepare --out data
```

Copy `data/` to durable `MyDrive/exp/datasets/e3/`, then stage it into local `/content/data` for training. Run the L4 debug commands in README before any formal run. Inspect at least one full trajectory, check action masks and tool results, and examine memory over 20 steps. Do not demand monotonic policy loss/reward from noisy rollouts as a pass criterion.

After successful debug, run the formal README commands with A100/1.5B or disclosed L4/0.5B fallback. Model/tokenizer/adapter shapes must match. E2 adapter is a separate baseline; do not initialize cold-start SFT from it. If using 0.5B fallback, the original E2 1.5B adapter cannot be loaded into the 0.5B model; run the E2 baseline with its own 1.5B config and label model sizes in the comparison instead of pretending an architecture match.

## Durable recovery

- On Colab, link `results/` to the mounted Drive run directory before starting any long stage (example below). This makes checkpoint saves durable at each save. Stage input `data/` on the local runtime disk; never train by streaming dataset files from Drive. Code does not mount Drive or manage cloud sessions.

  ```sh
  mkdir -p /content/drive/MyDrive/exp/runs/e3/run-01/results
  # Only when the local results directory has been inspected and its smoke artifacts preserved:
  mv results results-code-snapshot
  ln -s /content/drive/MyDrive/exp/runs/e3/run-01/results results
  ```
- GRPO resumes exact adapter/optimizer/scaler/RNG via `train --resume .../checkpoint-NNNN` with original `--adapter`, config, seed and data.
- SFT checkpoints every 25 optimizer updates; `sft --resume .../checkpoint-NNNN` with same arguments restores it.
- Collection resumes accepted IDs and attempt counts when source/config/seed metadata match. Post-restart sampling is not claimed bitwise identical; restart count is recorded.
- Evaluation resumes task IDs and refuses incompatible metadata. Incomplete JSONL or `.partial` checkpoint directories require inspection; never blindly mark them complete.
- `scripts/run_seeds.sh` starts fresh stages, so after any interruption use individual resumable commands rather than rerun the entire shell script without inspection.

## Pending real work / gates

1. GPU smoke and dependency compatibility on the allocated CUDA image. Direct pins in `pyproject.toml` were installed and tested on macOS; do not claim that proves Colab compatibility. Record actual torch/CUDA/GPU and pip freeze after first GPU success.
2. Actual dataset download/source hashes (the script resolves dataset commits at run time) and source license retention. GSM-Symbolic has CC-BY-NC-ND; do not publish modified data.
3. Approximately 500 correct training trajectories per cold-start seed; if the sampler cannot reach the target, it fails explicitly. Inspect extraction/prompt/quality before modifying protocol.
4. All five comparisons; E2 adapter must be transferred once E2 completes. Base and cold/GRPO comparisons can proceed independently first.
5. Three seeds for cold-SFT and GRPO, held-out dev checks, 500 GSM8K + 500 HotpotQA final questions plus full GSM-Symbolic main instances.
6. Python tool is disabled until `probe-python` passes in Linux. Colab may forbid namespaces; retain calculator/search profile and disclose missing Python rather than execute it on the host.
7. Compare against published GSM8K references (1.5B 73.2%, 0.5B 49.6%, Qwen report Table 10); fill, real cost ledger, ten before/after trajectory pairs and observed reward-exploitation analysis. Do not invent an example when none is observed.
8. Run `python scripts/verify_completion.py`. It intentionally rejects partial, smoke or synthetic results and missing coverage.

## Known methodology choices

GSM8K has no validation split; dev is held out of train. HotpotQA released validation is used only as final test. Direct baseline gets full local contexts; tool baselines get document titles plus BM25 retrieval. Report context truncation. Frozen KL reference is base-with-LoRA-disabled, not cold-SFT. Two optimizer updates per rollout enable actual clipping. Public contamination remains a limitation. There is no user interaction required for formal metrics, but manual qualitative trajectory interpretation must be honest and grounded in actual runs.
