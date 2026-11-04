# Multi-turn tool GRPO

A hand-written, auditable Qwen2.5 LoRA experiment: can outcome-reward RL improve calculator/search use beyond a separate cold-start SFT adapter?

**Status:** executable CPU-tested implementation; GPU smoke, cold-start sampling and formal benchmark results are pending. There are no claimed accuracy improvements yet.

## Reproduce

Python 3.11+; Node/server tools are not required inside this repository. Install training dependencies only on a suitably provisioned runtime, not the 2 GB orchestration server:

```sh
python -m pip install -e '.[train]'
python -m unittest discover -s tests -v
python scripts/run.py prepare --out data
```

`prepare` downloads pinned-at-run dataset revisions and writes disjoint `train.jsonl`, `dev.jsonl`, `test.jsonl` with source hashes. Large data stays on the GPU/CPU runtime or Drive. GSM-Symbolic keeps all released main instances; bootstrap resamples templates.

On an **already allocated** L4, inspect the first trajectories and do a 20-step debug run:

```sh
python scripts/run.py collect --config configs/l4-debug.json --data data/train.jsonl --out data/debug-cold.jsonl --target 2
python scripts/run.py sft --config configs/l4-debug.json --data data/debug-cold.jsonl --out results/debug-sft
python scripts/run.py train --config configs/l4-debug.json --data data/train.jsonl --dev data/dev.jsonl --out results/debug-grpo --adapter results/debug-sft/adapter
python scripts/run.py evaluate --config configs/l4-debug.json --data data/dev.jsonl --out results/debug-eval.jsonl --condition grpo --adapter results/debug-grpo/checkpoint-0020 --limit 2
```

Debug artifacts are marked smoke and cannot pass the formal verifier. Inspect reward changes, masks, trajectories and memory; a 20-step loss decline is not an unconditional correctness requirement for noisy policy optimization.

For formal 1.5B training use `configs/a100.json` (300 steps). If A100 is unavailable, `configs/l4-full.json` runs all planned steps with 0.5B; disclose the model change. T4 debug selects fp16 with GradScaler. Do not mix model/tokenizer shapes or reference baselines across profiles.

```sh
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl --out results/eval-base-direct.jsonl --condition base-direct
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl --out results/eval-base-tools.jsonl --condition base-tools
python scripts/run.py evaluate --config configs/a100.json --data data/test.jsonl --out results/eval-e2.jsonl --condition e2-lora --adapter /path/to/toolcall-lora/adapter
bash scripts/run_seeds.sh configs/a100.json
python scripts/run.py report results/eval-*.jsonl
python scripts/verify_completion.py
```

The seed script starts fresh stages and stops on any error; it deliberately refuses to overwrite work. After interruption inspect existing stage artifacts and run individual commands. Evaluation resumes by validated task IDs. GRPO resumes optimizer, RNG and adapter using:

```sh
python scripts/run.py train --config configs/a100.json --data data/train.jsonl --dev data/dev.jsonl --out results/grpo-42 --seed 42 --adapter results/cold-42/adapter --resume results/grpo-42/checkpoint-0050
```

SFT checkpoints every 25 optimizer updates and supports `sft --resume results/cold-42/checkpoint-0025` with the same stage arguments. Cold-start collection resumes accepted IDs and attempted-question counts with matching provenance; a resumed sampling stream is recorded as a restart rather than claimed bitwise identical. Partial/corrupt JSONL or checkpoint directories fail closed for inspection.

## Objective and implementation

- Sample G trajectories per question, center/scale outcome rewards within each group (population standard deviation, epsilon 1e-6).
- Freeze sampled old-policy and base-reference log probabilities before updates. Optimize the clipped current/old ratio with a k3 KL regularizer. Two updates per rollout make clipping operational.
- Average loss over **generated action tokens within each trajectory**, then trajectories. Prompt text and inserted tool outputs have zero gradient mask, even though later actions condition on them.
- One trajectory per gradient microbatch limits peak GPU memory. FP16 uses loss scaling; bf16 is rejected on unsupported GPUs. Checkpoints include optimizer, scaler, CPU/CUDA and sampling RNG states.
- The parsing reward is once per trajectory (+0.1), tool cost is -0.05/call, at most four calls. This prevents repeated syntactically valid calls from farming the parse bonus.
- Checkpoint evaluation uses held-out training dev data. Final test is never used for checkpoint selection.

## Tools and boundaries

Calculator is bounded AST arithmetic, not `eval`. Search is BM25 over the current question's provided paragraphs, never the network. Python is disabled in all supplied configs. To enable it, install Linux `bubblewrap`, pass `python scripts/run.py probe-python`, then set `python_tool: true` in the JSON config. It gets a minimal read-only filesystem, no network namespace connectivity, five-second timeout and bounded output. There is no host execution fallback. Colab may prohibit user namespaces; retain the two-tool profile and disclose this rather than bypass isolation.

Tool format is copied from E2's `toolcall_lora/tool_format.py`; it rejects duplicate JSON keys, nonfinite values and malformed tags. Credentials are never required for the standard public datasets/models; gated E2 access is a separate experiment.

## Results and completion

Raw test rows, configs, commit, model revision, hardware and data hashes accompany every report. Formal completion requires five conditions, three cold-SFT/GRPO seeds, raw final-test coverage, 300 training steps and real final adapters, a filled cost ledger and trajectory analysis. `scripts/verify_completion.py` is an independent stdlib gate; it currently fails correctly because no GPU experiments have been run.

See [baseline sources](docs/baselines.md), [cost log](docs/cost.md), and [handoff](HANDOFF.md). No data or code is published automatically.
