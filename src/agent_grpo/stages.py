"""Cold-start rejection sampling, assistant-token SFT and held-out evaluation."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from .schema import append_jsonl, read_jsonl, read_tasks
from .training import (copy_checkpoint, copy_outputs, load_config, metadata, model_and_tokenizer,
                       recorded_steps, restore_checkpoint, resume_provenance, save_checkpoint, seed_all, token_logprobs)
from .rollout import sample_trajectories
from .tools import probe_python_sandbox


def collect(config_path, data_path, out_path, seed=42, target=500, attempts_per_question=2, batch_size=32):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    config = load_config(config_path)
    tasks = read_tasks(data_path)
    if not tasks or any(task.split != "train" for task in tasks):
        raise ValueError("cold-start collection requires only training data")
    if config.get("python_tool") and not probe_python_sandbox():
        raise RuntimeError("Python tool sandbox probe failed")
    out = Path(out_path)
    meta_path = Path(str(out) + ".metadata.json")
    attempts_path = Path(str(out) + ".attempts.jsonl")
    accepted_ids = set()
    if out.exists():
        if not meta_path.exists():
            raise ValueError("collection has no provenance; inspect it before resuming")
        for row in read_jsonl(out):
            if row["task_id"] in accepted_ids:
                raise ValueError("duplicate accepted collection rows")
            accepted_ids.add(row["task_id"])
    counts = Counter(row["task_id"] for row in read_jsonl(attempts_path)) if attempts_path.exists() else Counter()
    attempted = counts.total()
    seed_all(seed)
    random.Random(seed).shuffle(tasks)
    model, tokenizer = model_and_tokenizer(config)
    meta = metadata(config, data_path, model)
    meta.update(target=target, seed=seed, attempts_per_question=attempts_per_question, rollout_batch_size=batch_size)
    executions_path = Path(str(out) + ".executions.jsonl")
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        for key in ["config", "data_sha256", "model_revision_resolved", "seed", "target", "attempts_per_question"]:
            if previous[key] != meta[key]:
                raise ValueError(f"collection resume mismatch: {key}")
        meta["resume_count"] = previous.get("resume_count", 0) + 1
        if not executions_path.exists():
            append_jsonl(executions_path, {"git_commit": previous["git_commit"], "created_at": previous["created_at"],
                "batch_size": previous.get("rollout_batch_size", 32), "existing_rows": 0,
                "preserved_rows_at_transition": len(accepted_ids), "attempts_at_transition": attempted})
    append_jsonl(executions_path, {"git_commit": meta["git_commit"], "created_at": meta["created_at"],
        "batch_size": batch_size, "existing_rows": len(accepted_ids), "existing_attempts": attempted})
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    last_copy = time.monotonic()
    pending = [task for task in tasks if task.id not in accepted_ids and counts[task.id] < attempts_per_question]
    cursor = 0
    while len(accepted_ids) < target and cursor < len(pending):
        batch = pending[cursor:cursor + min(batch_size, target - len(accepted_ids))]
        cursor += len(batch)
        for _ in range(attempts_per_question):
            active = [task for task in batch if task.id not in accepted_ids and counts[task.id] < attempts_per_question]
            if not active:
                break
            trajectories = sample_trajectories(model, tokenizer, active, temperature=config["temperature"],
                max_new_tokens=config["max_new_tokens"], max_sequence_length=config["max_sequence_length"],
                enable_python=config.get("python_tool", False))
            for task, trajectory in zip(active, trajectories):
                attempted += 1
                counts[task.id] += 1
                success = trajectory.reward["correctness"] == 1 and not trajectory.truncated
                if success:
                    append_jsonl(out, {**asdict(trajectory), "source_split": "train", "seed": seed,
                                      "execution_commit": meta["git_commit"]})
                    accepted_ids.add(task.id)
                append_jsonl(attempts_path, {"task_id": task.id, "accepted": success, "reward": trajectory.reward,
                                           "execution_commit": meta["git_commit"]})
            accepted = len(accepted_ids)
            print(json.dumps({"stage": "cold-collection", "seed": seed, "accepted": accepted,
                              "target": target, "attempted": attempted}), flush=True)
            # Persist only after all rows from this generation batch are closed.
            # A reclaimed Colab runtime must not lose the whole collection stage.
            if time.monotonic() - last_copy >= 60:
                meta.update(accepted=accepted, attempted=attempted, reached_target=accepted >= target)
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")
                copy_outputs([out, attempts_path, executions_path, meta_path], "data")
                last_copy = time.monotonic()
                print(json.dumps({"stage": "cold-collection-copy", "seed": seed,
                                  "accepted": accepted, "attempted": attempted,
                                  "time_unix": time.time()}), flush=True)
    accepted = len(accepted_ids)
    meta.update(accepted=accepted, attempted=attempted, reached_target=accepted >= target)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    if accepted < target:
        raise RuntimeError(f"only {accepted}/{target} correct trajectories collected; inspect before changing protocol")
    copy_outputs([out, meta_path, attempts_path, executions_path], "data")


def assert_trajectory_source(collection_meta, config, model):
    """SFT may only train on trajectories from this model.

    Older collection sidecars stored the revision only under config.model_revision.
    A missing revision is acceptable when the loaded model also has none; a present
    revision must match the resolved model commit.
    """
    recorded_model = (collection_meta.get("config") or {}).get("model")
    recorded_revision = collection_meta.get("model_revision_resolved")
    if recorded_revision is None:
        recorded_revision = (collection_meta.get("config") or {}).get("model_revision")
    resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
    if not recorded_model or recorded_model != config["model"] or recorded_revision != resolved:
        raise ValueError("SFT trajectories came from another model/tokenizer revision")


def sft(config_path, trajectories_path, out_dir, seed=42, epochs=1, resume=None):
    import torch
    config = load_config(config_path)
    seed_all(seed)
    records = list(read_jsonl(trajectories_path))
    if not records or any(row.get("source_split") != "train" or row["reward"]["correctness"] != 1 for row in records):
        raise ValueError("SFT accepts only verified correct training trajectories")
    out = Path(out_dir)
    if out.exists() and not resume:
        raise ValueError("SFT output exists; use --resume checkpoint or a new run directory")
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = model_and_tokenizer(config, resume, trainable=True)
    collection_meta = json.loads(Path(str(trajectories_path) + ".metadata.json").read_text())
    assert_trajectory_source(collection_meta, config, model)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.get("sft_learning_rate", 2e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=config["dtype"] == "float16" and device.type == "cuda")
    accumulation = config.get("sft_gradient_accumulation", 8)
    meta = metadata(config, trajectories_path, model)
    meta.update(seed=seed, epochs=epochs, trajectories=len(records), purpose="cold-start-sft")
    rng = random.Random(seed)
    batches_per_epoch = (len(records) + accumulation - 1) // accumulation
    start_step = 0
    logged_steps = recorded_steps(out / "train.jsonl")
    if resume:
        resume_provenance(resume, out, meta, ("config", "data_sha256", "model_revision_resolved", "seed"), "SFT")
        start_step = restore_checkpoint(resume, optimizer, scaler)
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
            if step not in logged_steps:
                append_jsonl(out / "train.jsonl", {"step": step, "epoch": epoch + 1, "offset": offset, "loss": sum(losses) / len(losses)})
                logged_steps.add(step)
            if step % 25 == 0 or step == epochs * batches_per_epoch:
                save_checkpoint(out / f"checkpoint-{step:04d}", model, tokenizer, optimizer, scaler, step, rng, meta)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    meta = metadata(config, trajectories_path, model)
    meta.update(seed=seed, epochs=epochs, trajectories=len(records), purpose="cold-start-sft")
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    copy_checkpoint(out / "adapter")
    copy_outputs([out / "run.json", out / "train.jsonl"], out.name)


def _existing_evaluation_rows(out, meta_path, tasks, condition, seed):
    """Reject duplicate or foreign rows before loading the model."""
    done = set()
    if not out.exists():
        return done
    if not meta_path.exists():
        raise ValueError("evaluation output has no provenance; inspect it before resuming")
    expected = {"condition": condition, "seed": seed, "split": tasks[0].split}
    for row in read_jsonl(out):
        task_id = row.get("task_id")
        if task_id in done:
            raise ValueError("duplicate evaluation task IDs")
        done.add(task_id)
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"cannot append incompatible evaluation: {key}")
    if done - {task.id for task in tasks}:
        raise ValueError("result includes unexpected task IDs")
    return done


def evaluate(config_path, data_path, out_path, condition, seed=42, adapter=None, batch_size=1):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
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
    meta_path = Path(str(out) + ".metadata.json")
    done = _existing_evaluation_rows(out, meta_path, tasks, condition, seed)
    model, tokenizer = model_and_tokenizer(config, adapter)
    meta = metadata(config, data_path, model)
    meta.update(condition=condition, seed=seed, adapter=adapter, split=tasks[0].split, tasks=len(tasks))
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        for key in ["config", "data_sha256", "condition", "seed", "adapter", "split", "tasks", "model_revision_resolved"]:
            if previous[key] != meta[key]:
                raise ValueError(f"cannot append incompatible evaluation: {key}")
    else:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    execution = {"created_at": meta["created_at"], "git_commit": meta["git_commit"],
                 "existing_rows": len(done), "requested_batch_size": batch_size,
                 "effective_batch_size": batch_size}
    append_jsonl(str(out) + ".executions.jsonl", execution)
    pending = [task for task in tasks if task.id not in done]
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        started = time.monotonic()
        trajectories = sample_trajectories(model, tokenizer, batch, tools_enabled=condition != "base-direct",
            max_new_tokens=config["max_new_tokens"], max_sequence_length=config["max_sequence_length"],
            enable_python=config.get("python_tool", False))
        elapsed = time.monotonic() - started
        for task, trajectory in zip(batch, trajectories):
            append_jsonl(out, {**asdict(trajectory), "dataset": task.dataset, "split": task.split,
                "source": task.source, "seed": seed, "condition": condition,
                "seconds": elapsed / len(batch), "timing_mode": "batch_amortized" if len(batch) > 1 else "single",
                "batch_size": len(batch), "batch_seconds": elapsed, "execution_commit": meta["git_commit"]})
        print(json.dumps({"condition": condition, "completed": len(done) + offset + len(batch),
                          "total": len(tasks), "batch_size": len(batch), "batch_seconds": elapsed}), flush=True)
    copy_outputs([out, meta_path, Path(str(out) + ".executions.jsonl")],
                 "results/core" if out.parent.name == "core" else "results")
