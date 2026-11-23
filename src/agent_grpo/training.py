"""Hand-written rollout -> frozen logprobs -> clipped GRPO update loop."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from .objective import group_advantages, grpo_loss
from .rollout import batch_tool_trajectories
from .schema import append_jsonl, read_tasks
from .tools import probe_python_sandbox


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config["temperature"] != 1.0:
        raise ValueError("training temperature must be 1.0 so the rollout distribution matches old logprobs")
    if config["group_size"] < 2 or config["update_epochs"] < 1:
        raise ValueError("group_size>=2 and update_epochs>=1 required")
    return config


def seed_all(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_and_tokenizer(config, adapter=None, trainable=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required by config")
    dtype = getattr(torch, config["dtype"])
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 unsupported on this GPU; choose t4 profile with float16")
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config.get("model_revision", "main"),
                    torch_dtype=dtype, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config.get("model_revision", "main"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=2 * config["lora_rank"],
                  lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.eval()
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, tokenizer


def token_logprobs(model, token_ids, device):
    import torch
    model.train(torch.is_grad_enabled())  # checkpointing during updates, dropout explicitly zero
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits = model(inputs, use_cache=False).logits[0, :-1]
    labels = inputs[0, 1:]
    # Avoid materializing a second full [sequence,vocab] float32 log-softmax.
    selected = logits.gather(1, labels[:, None]).squeeze(1).float()
    from torch.utils.checkpoint import checkpoint
    def normalize(block):
        return torch.logsumexp(block.float(), dim=-1)
    normalizer = torch.cat([checkpoint(normalize, chunk, use_reentrant=False) if torch.is_grad_enabled()
                            else normalize(chunk) for chunk in logits.split(128)], dim=0)
    return selected - normalizer


def metadata(config, data_path, model=None):
    import torch
    from .rewards import SCORE_VERSION
    def git(args):
        p = subprocess.run(["git", *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else "unknown"
    commit = os.environ.get("EXP_COMMIT") or os.environ.get("EXPERIMENT_COMMIT") or git(["rev-parse", "HEAD"])
    dirty = os.environ.get("EXP_DIRTY", os.environ.get("EXPERIMENT_DIRTY"))
    runtime_revision = Path(__file__).resolve().parents[2] / ".runtime-revision"
    if runtime_revision.exists():
        commit, dirty = runtime_revision.read_text().strip(), "false"
    return {"created_at": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "score_version": SCORE_VERSION,
            "git_dirty": dirty.lower() == "true" if dirty is not None else bool(git(["status", "--porcelain"])), "config": config,
            "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
            "model_revision_resolved": getattr(getattr(model, "config", None), "_commit_hash", None)}


def _json_object(path):
    """A non-empty JSON object, or None when the file is missing or not verifiable."""
    path = Path(path)
    if not path.is_file():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and value else None


def resume_provenance(checkpoint_dir, out_dir, current, required, label, optional=()):
    """Match a checkpoint to this run.

    Keys stored on the checkpoint win. Empty or absent metadata is accepted
    only when run.json still proves the same identity, which is how older
    checkpoints without a complete sidecar remain resumable.
    """
    checkpoint = _json_object(Path(checkpoint_dir) / "metadata.json")
    history = _json_object(Path(out_dir) / "run.json")
    if checkpoint is None and history is None:
        raise ValueError(f"{label} provenance is not verifiable")
    merged = {}
    if history:
        merged.update(history)
    if checkpoint:
        merged.update(checkpoint)
    for key in required:
        if key not in merged:
            raise ValueError(f"{label} provenance is not verifiable: {key}")
        if merged[key] != current[key]:
            raise ValueError(f"{label} resume mismatch: {key}")
    for key in optional:
        if key in merged and merged[key] != current.get(key):
            raise ValueError(f"{label} resume mismatch: {key}")
    return merged


def recorded_steps(path):
    steps = set()
    path = Path(path)
    if not path.exists():
        return steps
    for line in path.read_text().splitlines():
        if line.strip():
            steps.add(json.loads(line).get("step"))
    return steps


def drop_uncommitted_rollouts(path, committed_steps, checkpoint_step):
    """Drop a torn rollout tail without removing rows already covered by a checkpoint."""
    path = Path(path)
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    kept = []
    for row in rows:
        step = row.get("step")
        if step in committed_steps or (isinstance(step, int) and step <= checkpoint_step):
            kept.append(row)
    if len(kept) != len(rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in kept))


def dev_datasets_missing(path, step, datasets):
    present = set()
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("step") == step:
                present.add(row.get("dataset"))
    return [dataset for dataset in datasets if dataset not in present]


def step_is_checkpoint(step, config):
    return step > 0 and (step % config["checkpoint_every"] == 0 or step == config["steps"])


def write_dev_metrics(model, tokenizer, config, dev_tasks, out, step):
    datasets = sorted({task.dataset for task in dev_tasks})
    missing = dev_datasets_missing(out / "dev.jsonl", step, datasets)
    if not missing:
        return
    eval_batch_size = config.get("eval_batch_size", 32)
    for dataset in missing:
        subset = [task for task in dev_tasks if task.dataset == dataset][:50]
        values = []
        for offset in range(0, len(subset), eval_batch_size):
            trajectories = batch_tool_trajectories(model, tokenizer, subset[offset:offset + eval_batch_size],
                temperature=0, max_new_tokens=config["max_new_tokens"],
                max_sequence_length=config["max_sequence_length"],
                enable_python=config.get("python_tool", False))
            values.extend(trajectory.reward["correctness"] for trajectory in trajectories)
        append_jsonl(out / "dev.jsonl", {"step": step, "dataset": dataset, "split": "dev",
            "n": len(values), "mean": sum(values) / len(values)})


def copy_checkpoint(directory):
    """Optionally copy a completed checkpoint using controller-supplied rclone auth."""
    remote = os.environ.get("EXP_CHECKPOINT_REMOTE")
    if remote:
        directory = Path(directory)
        destination = remote.rstrip("/") + "/" + directory.parent.name + "/" + directory.name
        subprocess.run(["rclone", "copy", str(directory), destination], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def copy_outputs(paths, subdirectory):
    """Copy closed stage outputs to the same configured Drive experiment directory."""
    remote = os.environ.get("EXP_CHECKPOINT_REMOTE")
    if remote:
        for path in map(Path, paths):
            destination = remote.rstrip("/") + "/" + subdirectory.strip("/") + "/" + path.name
            subprocess.run(["rclone", "copyto", str(path), destination], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def save_checkpoint(directory, model, tokenizer, optimizer, scaler, step, rng, meta):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": step,
                "sampling_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, directory / "training-state.pt")
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    copy_checkpoint(directory)


def training(config_path, data_path, out_dir, seed, adapter=None, resume=None, dev_path=None):
    import torch
    config = load_config(config_path)
    config["seed"] = seed
    config["initial_adapter"] = adapter
    tasks = read_tasks(data_path)
    if not tasks or any(task.split != "train" for task in tasks):
        raise ValueError("training accepts only nonempty train split")
    if config.get("python_tool") and not probe_python_sandbox():
        raise RuntimeError("Python sandbox probe failed")
    dev_tasks = read_tasks(dev_path) if dev_path else []
    if not adapter:
        raise ValueError("formal GRPO requires the cold-start SFT --adapter, including on resume")
    if not dev_tasks:
        raise ValueError("formal training requires --dev from held-out train split")
    if any(task.split != "dev" for task in dev_tasks):
        raise ValueError("checkpoint evaluation must use only dev, never final test")
    if {task.id for task in tasks} & {task.id for task in dev_tasks}:
        raise ValueError("training/dev overlap")
    seed_all(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists() and not resume:
        raise ValueError("run directory already exists; resume explicitly or choose a new one")
    model, tokenizer = model_and_tokenizer(config, resume or adapter, trainable=True)
    device = next(model.parameters()).device
    meta = metadata(config, data_path, model)
    meta["rollout_batch_size"] = config.get("rollout_batch_size", 64)
    meta["dev_sha256"] = hashlib.sha256(Path(dev_path).read_bytes()).hexdigest()
    meta["initial_adapter"] = adapter
    if resume:
        resume_provenance(resume, out, meta, ("config", "data_sha256", "model_revision_resolved"),
                          "training", ("dev_sha256", "initial_adapter"))
    if not (out / "run.json").exists():
        (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not (out / "env.txt").exists():
        (out / "env.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["dtype"] == "float16" and device.type == "cuda")
    rng = random.Random(seed)
    start = 0
    if resume:
        # Only load checkpoints created by this trusted experiment, never third-party pickles.
        state = torch.load(Path(resume) / "training-state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.setstate(state["sampling_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            if len(state["cuda_rng"]) != torch.cuda.device_count():
                raise RuntimeError("CUDA device count does not match checkpoint RNG state")
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start = state["step"]
    logged_steps = recorded_steps(out / "train.jsonl")
    if resume:
        drop_uncommitted_rollouts(out / "rollouts.jsonl", logged_steps, start)
        if step_is_checkpoint(start, config):
            write_dev_metrics(model, tokenizer, config, dev_tasks, out, start)
    for step in range(start, config["steps"]):
        started = time.monotonic()
        step_no = step + 1
        keep_log = step_no not in logged_steps
        sampled = rng.sample(tasks, min(config["questions_per_step"], len(tasks)))
        batch, zero_groups = [], 0
        group_size = config["group_size"]
        questions_per_batch = max(1, config.get("rollout_batch_size", 64) // group_size)
        sampled_trajectories = []
        for offset in range(0, len(sampled), questions_per_batch):
            rollout_tasks = [task for task in sampled[offset:offset + questions_per_batch] for _ in range(group_size)]
            sampled_trajectories.extend(batch_tool_trajectories(model, tokenizer, rollout_tasks,
                temperature=config["temperature"], max_new_tokens=config["max_new_tokens"],
                max_sequence_length=config["max_sequence_length"], enable_python=config.get("python_tool", False)))
        for group_index, task in enumerate(sampled):
            trajectories = sampled_trajectories[group_index * group_size:(group_index + 1) * group_size]
            advantages = group_advantages([t.reward["total"] for t in trajectories])
            zero_groups += int(all(a == 0 for a in advantages))
            for trajectory, advantage in zip(trajectories, advantages):
                if not any(trajectory.generated_mask[1:]):
                    raise RuntimeError("empty generated trajectory; fix context length before training")
                with torch.no_grad():
                    old = token_logprobs(model, trajectory.token_ids, device).cpu()
                    # Reference is frozen base model, not SFT adapter. Documented comparison choice.
                    with model.disable_adapter():
                        reference = token_logprobs(model, trajectory.token_ids, device).cpu()
                batch.append((trajectory, advantage, old, reference))
                if keep_log:
                    append_jsonl(out / "rollouts.jsonl", {"step": step_no, **asdict(trajectory)})
        metrics = []
        for epoch in range(config["update_epochs"]):
            optimizer.zero_grad(set_to_none=True)
            for trajectory, advantage, old, reference in batch:
                current = token_logprobs(model, trajectory.token_ids, device)
                mask = torch.tensor(trajectory.generated_mask[1:], dtype=torch.bool, device=device)
                loss, details = grpo_loss(current, old.to(device), reference.to(device), mask, advantage,
                                         config["kl_beta"], config["clip_epsilon"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss; refusing to update")
                scaler.scale(loss / len(batch)).backward()
                metrics.append({"loss": float(loss.detach()), **details})
                del current, loss
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        record = {"step": step_no, "seed": seed, "seconds": time.monotonic() - started,
                  "reward": sum(t.reward["total"] for t, *_ in batch) / len(batch),
                  "correctness": sum(t.reward["correctness"] for t, *_ in batch) / len(batch),
                  "tool_calls": sum(t.reward["tool_calls"] for t, *_ in batch) / len(batch),
                  "response_tokens": sum(sum(t.generated_mask) for t, *_ in batch) / len(batch),
                  "zero_advantage_group_fraction": zero_groups / len(sampled),
                  "truncated_fraction": sum(t.truncated for t, *_ in batch) / len(batch),
                  **{key: sum(m[key] for m in metrics) / len(metrics) for key in ["loss", "kl", "clip_fraction"]}}
        if keep_log:
            append_jsonl(out / "train.jsonl", record)
            logged_steps.add(step_no)
        print(json.dumps(record), flush=True)
        if step_no % config["checkpoint_every"] == 0 or step_no == config["steps"]:
            save_checkpoint(out / f"checkpoint-{step_no:04d}", model, tokenizer, optimizer, scaler, step_no, rng, meta)
            write_dev_metrics(model, tokenizer, config, dev_tasks, out, step_no)
            copy_outputs([p for p in (out / name for name in ["run.json", "env.txt", "train.jsonl", "rollouts.jsonl", "dev.jsonl"]) if p.exists()], out.name)
    copy_outputs([p for p in (out / name for name in ["run.json", "env.txt", "train.jsonl", "rollouts.jsonl", "dev.jsonl"]) if p.exists()], out.name)
    return out
