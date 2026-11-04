"""Cold-start rejection sampling, assistant-token SFT and held-out evaluation."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from .schema import Trajectory, append_jsonl, read_tasks
from .training import load_config, seed_all, model_and_tokenizer, token_logprobs, metadata
from .rollout import sample_trajectory, batch_direct_trajectories, batch_tool_trajectories
from .tools import probe_python_sandbox


def collect(config_path, data_path, out_path, seed=42, target=500, attempts_per_question=2):
    config = load_config(config_path)
    tasks = read_tasks(data_path)
    if not tasks or any(task.split != "train" for task in tasks):
        raise ValueError("cold-start collection requires only training data")
    if config.get("python_tool") and not probe_python_sandbox():
        raise RuntimeError("Python tool sandbox probe failed")
    out = Path(out_path)
    meta_path = Path(str(out) + ".metadata.json")
    attempts_path = Path(str(out) + ".attempts.jsonl")
    accepted_rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    accepted_ids = {row["task_id"] for row in accepted_rows}
    if len(accepted_ids) != len(accepted_rows):
        raise ValueError("duplicate accepted collection rows")
    if out.exists() and not meta_path.exists():
        raise ValueError("collection has no provenance; inspect it before resuming")
    attempts = [json.loads(line) for line in attempts_path.read_text().splitlines() if line.strip()] if attempts_path.exists() else []
    counts = {}
    for row in attempts:
        counts[row["task_id"]] = counts.get(row["task_id"], 0) + 1
    seed_all(seed)
    random.Random(seed).shuffle(tasks)
    model, tokenizer = model_and_tokenizer(config)
    meta = metadata(config, data_path, model)
    meta.update(target=target, seed=seed, attempts_per_question=attempts_per_question)
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        for key in ["config", "data_sha256", "model_revision_resolved", "seed", "target", "attempts_per_question"]:
            if previous[key] != meta[key]:
                raise ValueError(f"collection resume mismatch: {key}")
        meta["resume_count"] = previous.get("resume_count", 0) + 1
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    accepted, attempted = len(accepted_ids), len(attempts)
    for task in tasks:
        if accepted >= target:
            break
        if task.id in accepted_ids:
            continue
        for _ in range(counts.get(task.id, 0), attempts_per_question):
            trajectory = sample_trajectory(model, tokenizer, task, max_new_tokens=config["max_new_tokens"],
                          max_sequence_length=config["max_sequence_length"], enable_python=config.get("python_tool", False))
            attempted += 1
            success = trajectory.reward["correctness"] == 1 and not trajectory.truncated
            if success:
                append_jsonl(out, {**asdict(trajectory), "source_split": "train", "seed": seed})
                accepted += 1
            append_jsonl(attempts_path, {"task_id": task.id, "accepted": success, "reward": trajectory.reward})
            if success:
                break
    meta.update(accepted=accepted, attempted=attempted, reached_target=accepted >= target)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    if accepted < target:
        raise RuntimeError(f"only {accepted}/{target} correct trajectories collected; inspect before changing protocol")


def sft(config_path, trajectories_path, out_dir, seed=42, epochs=1, resume=None):
    import torch
    config = load_config(config_path)
    seed_all(seed)
    records = [json.loads(line) for line in Path(trajectories_path).read_text().splitlines() if line.strip()]
    if not records or any(row.get("source_split") != "train" or row["reward"]["correctness"] != 1 for row in records):
        raise ValueError("SFT accepts only verified correct training trajectories")
    out = Path(out_dir)
    if out.exists() and not resume:
        raise ValueError("SFT output exists; use --resume checkpoint or a new run directory")
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = model_and_tokenizer(config, resume, trainable=True)
    collection_meta = json.loads(Path(str(trajectories_path) + ".metadata.json").read_text())
    if collection_meta["config"]["model"] != config["model"] or collection_meta["model_revision_resolved"] != getattr(model.config, "_commit_hash", None):
        raise ValueError("SFT trajectories came from another model/tokenizer revision")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.get("sft_learning_rate", 2e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=config["dtype"] == "float16" and device.type == "cuda")
    accumulation = config.get("sft_gradient_accumulation", 8)
    from .training import save_checkpoint
    meta = metadata(config, trajectories_path, model)
    meta.update(seed=seed, epochs=epochs, trajectories=len(records), purpose="cold-start-sft")
    rng = random.Random(seed)
    batches_per_epoch = (len(records) + accumulation - 1) // accumulation
    start_step = 0
    if resume:
        state = torch.load(Path(resume) / "training-state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step = state["step"]
    for epoch in range(epochs):
        order = list(records)
        random.Random(seed + epoch).shuffle(order)
        for offset in range(0, len(records), accumulation):
            step = epoch * batches_per_epoch + offset // accumulation + 1
            if step <= start_step:
                continue
            microbatch = order[offset:offset + accumulation]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for row in microbatch:
                logps = token_logprobs(model, row["token_ids"], device)
                mask = torch.tensor(row["generated_mask"][1:], dtype=torch.bool, device=device)
                if not mask.any():
                    raise ValueError("empty action mask")
                loss = -logps[mask].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite SFT loss")
                scaler.scale(loss / len(microbatch)).backward()
                losses.append(float(loss.detach()))
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            append_jsonl(out / "train.jsonl", {"step": step, "epoch": epoch + 1, "offset": offset, "loss": sum(losses) / len(losses)})
            if step % 25 == 0 or step == epochs * batches_per_epoch:
                save_checkpoint(out / f"checkpoint-{step:04d}", model, tokenizer, optimizer, scaler, step, rng, meta)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    meta = metadata(config, trajectories_path, model)
    meta.update(seed=seed, epochs=epochs, trajectories=len(records), purpose="cold-start-sft")
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


def evaluate(config_path, data_path, out_path, condition, seed=42, adapter=None, batch_size=1):
    import torch
    conditions = {"base-direct", "base-tools", "e2-lora", "cold-sft", "grpo"}
    if condition not in conditions:
        raise ValueError("unknown condition")
    if (condition in {"e2-lora", "cold-sft", "grpo"}) != bool(adapter):
        raise ValueError("condition and adapter availability disagree")
    config = load_config(config_path)
    config["seed"] = seed
    seed_all(seed)
    tasks = read_tasks(data_path)
    if not tasks or any(task.split not in {"dev", "test"} for task in tasks) or len({task.split for task in tasks}) != 1:
        raise ValueError("evaluation accepts a nonempty, unmixed dev or test split")
    if config.get("python_tool") and not probe_python_sandbox():
        raise RuntimeError("Python tool sandbox probe failed")
    out = Path(out_path)
    done = {}
    if out.exists():
        done = {row["task_id"]: row for row in map(json.loads, out.read_text().splitlines())}
    model, tokenizer = model_and_tokenizer(config, adapter)
    meta = metadata(config, data_path, model)
    meta.update(condition=condition, seed=seed, adapter=adapter, split=tasks[0].split, tasks=len(tasks))
    meta_path = Path(str(out) + ".metadata.json")
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        for key in ["config", "data_sha256", "condition", "seed", "adapter", "split", "tasks", "model_revision_resolved"]:
            if previous[key] != meta[key]:
                raise ValueError(f"cannot append incompatible evaluation: {key}")
    else:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    if set(done) - {task.id for task in tasks}:
        raise ValueError("result includes unexpected task IDs")
    execution = {"created_at": meta["created_at"], "git_commit": meta["git_commit"],
                 "existing_rows": len(done), "requested_batch_size": batch_size,
                 "effective_batch_size": batch_size}
    append_jsonl(str(out) + ".executions.jsonl", execution)
    pending = [task for task in tasks if task.id not in done]
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        started = time.monotonic()
        if condition == "base-direct" and batch_size > 1:
            trajectories = batch_direct_trajectories(model, tokenizer, batch,
                max_new_tokens=config["max_new_tokens"], max_sequence_length=config["max_sequence_length"])
        elif batch_size > 1:
            trajectories = batch_tool_trajectories(model, tokenizer, batch,
                max_new_tokens=config["max_new_tokens"], max_sequence_length=config["max_sequence_length"],
                enable_python=config.get("python_tool", False))
        else:
            trajectories = [sample_trajectory(model, tokenizer, batch[0], temperature=0,
                tools_enabled=condition != "base-direct", max_new_tokens=config["max_new_tokens"],
                max_sequence_length=config["max_sequence_length"], enable_python=config.get("python_tool", False))]
        elapsed = time.monotonic() - started
        for task, trajectory in zip(batch, trajectories):
            append_jsonl(out, {**asdict(trajectory), "dataset": task.dataset, "split": task.split,
                "source": task.source, "seed": seed, "condition": condition,
                "seconds": elapsed / len(batch), "timing_mode": "batch_amortized" if len(batch) > 1 else "single",
                "batch_size": len(batch), "batch_seconds": elapsed, "execution_commit": meta["git_commit"]})
        print(json.dumps({"condition": condition, "completed": len(done) + offset + len(batch),
                          "total": len(tasks), "batch_size": len(batch), "batch_seconds": elapsed}), flush=True)
