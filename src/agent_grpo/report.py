"""Aggregate genuine evaluations; bootstrap GSM-Symbolic by template cluster."""
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def rescore_saved(data_path, input_path, output_path):
    """Rescore saved generations without editing or regenerating their original records."""
    from datetime import datetime, timezone
    import hashlib
    from .schema import read_tasks
    from .rewards import SCORE_VERSION, score_answer, score_answer_strict
    import re
    tasks = {task.id: task for task in read_tasks(data_path)}
    source = Path(input_path)
    output = Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("use a separate output path to retain original generation records")
    original_bytes = source.read_bytes()
    rows = [json.loads(line) for line in original_bytes.decode().split("\n") if line.strip()]
    meta = json.loads(Path(str(source) + ".metadata.json").read_text())
    if hashlib.sha256(Path(data_path).read_bytes()).hexdigest() != meta["data_sha256"]:
        raise ValueError("rescoring requires the original evaluation dataset")
    for row in rows:
        task = tasks[row["task_id"]]
        old = row["reward"]
        score = score_answer(task.dataset, row["final_text"], task.answer)
        row["original_reward"] = old
        row["reward"] = {**old, "correctness": score,
            "total": score + old["parse_bonus"] - old["tool_penalty"],
            "strict_correctness_v1": score_answer_strict(task.dataset, row["final_text"], task.answer),
            "final_format_compliant": bool(re.search(r"<final>[^<>]+</final>\s*$", row["final_text"])),
            "score_version": SCORE_VERSION}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    meta.update(score_version=SCORE_VERSION, rescoring={"created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source), "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "rows": len(rows), "complete": len(rows) == meta["tasks"], "regenerated": False})
    Path(str(output) + ".metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"out": str(output), "rows": len(rows), "complete": len(rows) == meta["tasks"], "score_version": SCORE_VERSION}


def bootstrap_mean(values, seed=42, count=1000):
    if not values:
        raise ValueError("empty sample")
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(count))
    return [estimates[int(0.025 * count)], estimates[min(count - 1, int(0.975 * count))]]


def summarize(paths, output):
    from .rewards import SCORE_VERSION
    groups = defaultdict(list)
    provenance = []
    for path in paths:
        path = Path(path)
        meta = json.loads(Path(str(path) + ".metadata.json").read_text())
        rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
        if not rows or meta["split"] != "test" or len(rows) != meta["tasks"]:
            raise ValueError(f"not a complete formal final-test evaluation: {path}")
        if any(row["reward"].get("score_version") != SCORE_VERSION for row in rows):
            raise ValueError("all conditions must use the same scoring version; rescore saved generations first")
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
                          "score_version": SCORE_VERSION,
                          "strict_v1_mean": statistics.fmean(row["reward"]["strict_correctness_v1"] for row in rows),
                          "format_compliance": statistics.fmean(row["reward"]["final_format_compliant"] for row in rows),
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
