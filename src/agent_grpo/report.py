"""Aggregate genuine evaluations; bootstrap GSM-Symbolic by template cluster."""
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def bootstrap_mean(values, seed=42, count=1000):
    if not values:
        raise ValueError("empty sample")
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(count))
    return [estimates[int(0.025 * count)], estimates[min(count - 1, int(0.975 * count))]]


def summarize(paths, output):
    groups = defaultdict(list)
    provenance = []
    for path in paths:
        path = Path(path)
        meta = json.loads(Path(str(path) + ".metadata.json").read_text())
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows or meta["split"] != "test" or len(rows) != meta["tasks"]:
            raise ValueError(f"not a complete formal final-test evaluation: {path}")
        if len({row["task_id"] for row in rows}) != len(rows):
            raise ValueError("duplicate task IDs in result")
        provenance.append({"result_file": str(path), **meta})
        for row in rows:
            if row["split"] != "test":
                raise ValueError("development rows cannot enter formal report")
            groups[(row["condition"], row["seed"], row["dataset"])].append(row)
    summaries = []
    for (condition, seed, dataset), rows in sorted(groups.items()):
        values = [row["reward"]["correctness"] for row in rows]
        units = values
        ci_unit = "question"
        if dataset == "gsm-symbolic":
            clustered = defaultdict(list)
            for row in rows:
                clustered[row["source"]["template_id"]].append(row["reward"]["correctness"])
            units = [statistics.fmean(group) for group in clustered.values()]
            ci_unit = "template"
        summaries.append({"condition": condition, "seed": seed, "dataset": dataset, "n": len(rows),
                          "mean": statistics.fmean(values), "ci95": bootstrap_mean(units), "ci_unit": ci_unit,
                          "tool_calls_mean": statistics.fmean(row["reward"]["tool_calls"] for row in rows),
                          "truncated_fraction": statistics.fmean(row["truncated"] for row in rows)})
    seed_groups = defaultdict(list)
    for row in summaries:
        seed_groups[(row["condition"], row["dataset"])].append(row["mean"])
    seeds_summary = [{"condition": condition, "dataset": dataset, "seeds": len(values),
                     "mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else None}
                    for (condition, dataset), values in sorted(seed_groups.items())]
    result = {"formal": True, "provenance": provenance, "evaluations": summaries, "seed_summary": seeds_summary,
              "limitations": ["Public benchmarks may be contaminated; GSM-Symbolic remains derived from GSM8K.",
                              "Python tool is disabled unless Linux isolation probe passed.",
                              "No claim of speed, convergence, or positive improvement without results."]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    table = ["# Final test results", "", "| Condition | Dataset | Seeds | Mean | Std |", "|---|---|---:|---:|---:|"]
    for row in seeds_summary:
        std = "—" if row["std"] is None else f"{row['std']:.4f}"
        table.append(f"| {row['condition']} | {row['dataset']} | {row['seeds']} | {row['mean']:.4f} | {std} |")
    output.with_suffix(".md").write_text("\n".join(table) + "\n")
    return result
