"""Prepare the saved step-200 history for the interrupted seed-42 run.

This operates only on collected records. It never loads a model or starts a job.
"""
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1]
source = root / "results/collected/seed42"
output = root / "results/recovery-seed42"
output.mkdir(parents=True, exist_ok=True)
train_path = source / "grpo-42-train-steps1-214.jsonl"
dev_path = source / "grpo-42-dev-through-step200.jsonl"
train = [json.loads(line) for line in train_path.read_text().splitlines()]
dev = [json.loads(line) for line in dev_path.read_text().splitlines()]
saved = [row for row in train if row["step"] <= 200]
lost = [row for row in train if row["step"] > 200]
files = {
    "train.jsonl": saved,
    "dev.jsonl": [row for row in dev if row["step"] <= 200],
    "observed-lost-segment-201-214.jsonl": lost,
}
for name, rows in files.items():
    (output / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

block_means = [statistics.mean(r["seconds"] for r in saved[i:i + 50])
               for i in range(0, 200, 50)]
manifest = {
    "scope": "recovery_inputs_not_a_new_training_run",
    "seed": 42,
    "checkpoint_step": 200,
    "last_observed_step": 214,
    "next_step_after_resume": 201,
    "canonical_steps": [row["step"] for row in saved],
    "lost_model_segment_steps": [row["step"] for row in lost],
    "dev_rows_preserved": len(files["dev.jsonl"]),
    "original_source_commit": "1e9b43d341a99b0f0499bfd0e4e1835f6500a987",
    "checkpoint_remote": "gdrive:exp/runs/E3/grpo-42/checkpoint-0200",
    "cold_adapter_remote": "gdrive:exp/runs/E3/cold-42/adapter",
    "dataset_remote": "gdrive:exp/datasets/e3",
    "data_sha256": "606fd2dda50be0aa5dfc091617520a8850942c5fe6585ba9ab92941f4951bbab",
    "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (train_path, dev_path)},
    "prepared_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                        for name in files},
    "training_timing_estimate": {
        "basis": "minimum/maximum mean seconds of four observed 50-step blocks; not a confidence interval",
        "observed_block_mean_seconds": block_means,
        "remaining_seed42_steps": 100,
        "remaining_all_seeds_steps": 700,
        "seed42_compute_hours_range": [100 * min(block_means) / 3600, 100 * max(block_means) / 3600],
        "all_seeds_compute_hours_range": [700 * min(block_means) / 3600, 700 * max(block_means) / 3600],
        "excludes": ["cold collection", "SFT", "checkpoint copies", "dev evaluation", "final evaluation", "setup"],
    },
    "limitations": [
        "The original GRPO rollout trajectories for steps 1-214 were not collected before runtime loss.",
        "Metrics for steps 201-214 do not restore model or optimizer state; they are archived separately.",
        "A resumed rollout file will cover only the recovery segment starting at step 201.",
        "Only seed 42 has partial GRPO observations; no final training-effect or three-seed conclusion is available.",
    ],
}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"output": str(output), "canonical_train_rows": len(saved),
                  "archived_lost_segment_rows": len(lost), "dev_rows": len(files["dev.jsonl"])}))
