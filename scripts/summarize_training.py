"""Summarize recorded training steps; these are not held-out test results."""
import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("records", type=Path)
parser.add_argument("--block-size", type=int, default=20)
args = parser.parse_args()
if args.block_size < 1:
    raise SystemExit("--block-size must be >= 1")
rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
if not rows:
    raise SystemExit("no training records found")
keys = ["reward", "correctness", "kl", "tool_calls", "response_tokens",
        "zero_advantage_group_fraction", "truncated_fraction", "seconds"]
blocks = [rows[i:i + args.block_size] for i in range(0, len(rows), args.block_size)]
print(json.dumps({
    "source": str(args.records),
    "scope": "intermediate_training_descriptive_statistics",
    "limitations": ["Questions and sampled trajectories vary across steps; block means are not paired comparisons.",
                    "Training correctness mixes math correctness and HotpotQA F1; it is not held-out accuracy.",
                    "Step seconds exclude checkpoint copies and dev evaluation; they are not total runtime or CU cost.",
                    "One partial seed cannot establish final improvement or three-seed uncertainty."],
    "training_step_seconds_sum": sum(r["seconds"] for r in rows),
    "blocks": [{"first_step": b[0]["step"], "last_step": b[-1]["step"], "n_steps": len(b),
                "means": {k: statistics.mean(r[k] for r in b) for k in keys}} for b in blocks],
}, indent=2))
