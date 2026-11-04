"""Hand-written rollout -> frozen logprobs -> clipped GRPO update loop."""
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from .objective import group_advantages, grpo_loss
from .rollout import sample_trajectory
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
        raise RuntimeError("CUDA required by config; use a dedicated CPU smoke config only")
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
    def git(args):
        p = subprocess.run(["git", *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else "unknown"
    return {"created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git(["rev-parse", "HEAD"]),
            "git_dirty": bool(git(["status", "--porcelain"])), "config": config,
            "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
            "model_revision_resolved": getattr(getattr(model, "config", None), "_commit_hash", None)}


def save_checkpoint(directory, model, tokenizer, optimizer, scaler, step, rng, meta):
    import torch
    directory = Path(directory)
    temporary = directory.with_name(directory.name + ".partial")
    if temporary.exists():
        raise RuntimeError(f"incomplete checkpoint exists: {temporary}; inspect before retry")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": step,
                "sampling_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, temporary / "training-state.pt")
    (temporary / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    temporary.rename(directory)


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
    if not config.get("smoke") and not adapter:
        raise ValueError("formal GRPO requires the cold-start SFT --adapter, including on resume")
    if not config.get("smoke") and not dev_tasks:
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
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "env.txt").write_text(subprocess.check_output([__import__('sys').executable, "-m", "pip", "freeze"], text=True))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["dtype"] == "float16" and device.type == "cuda")
    rng = random.Random(seed)
    start = 0
    if resume:
        # Only load checkpoints created by this trusted experiment, never third-party pickles.
        state = torch.load(Path(resume) / "training-state.pt", map_location="cpu", weights_only=False)
        saved = json.loads((Path(resume) / "metadata.json").read_text())
        if saved["data_sha256"] != meta["data_sha256"] or saved["config"] != config or saved["model_revision_resolved"] != meta["model_revision_resolved"]:
            raise ValueError("checkpoint config/data differs from requested run")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.setstate(state["sampling_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start = state["step"]
    for step in range(start, config["steps"]):
        started = time.monotonic()
        sampled = rng.sample(tasks, min(config["questions_per_step"], len(tasks)))
        batch, zero_groups = [], 0
        for task in sampled:
            trajectories = [sample_trajectory(model, tokenizer, task, temperature=config["temperature"],
                            max_new_tokens=config["max_new_tokens"], max_sequence_length=config["max_sequence_length"],
                            enable_python=config.get("python_tool", False)) for _ in range(config["group_size"])]
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
                append_jsonl(out / "rollouts.jsonl", {"step": step + 1, **asdict(trajectory)})
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
        record = {"step": step + 1, "seed": seed, "seconds": time.monotonic() - started,
                  "reward": sum(t.reward["total"] for t, *_ in batch) / len(batch),
                  "correctness": sum(t.reward["correctness"] for t, *_ in batch) / len(batch),
                  "tool_calls": sum(t.reward["tool_calls"] for t, *_ in batch) / len(batch),
                  "response_tokens": sum(sum(t.generated_mask) for t, *_ in batch) / len(batch),
                  "zero_advantage_group_fraction": zero_groups / len(sampled),
                  "truncated_fraction": sum(t.truncated for t, *_ in batch) / len(batch),
                  **{key: sum(m[key] for m in metrics) / len(metrics) for key in ["loss", "kl", "clip_fraction"]}}
        append_jsonl(out / "train.jsonl", record)
        print(json.dumps(record), flush=True)
        if (step + 1) % config["checkpoint_every"] == 0 or step + 1 == config["steps"]:
            save_checkpoint(out / f"checkpoint-{step+1:04d}", model, tokenizer, optimizer, scaler, step + 1, rng, meta)
            for dataset in sorted({task.dataset for task in dev_tasks}):
                subset = [task for task in dev_tasks if task.dataset == dataset][:50]
                values = [sample_trajectory(model, tokenizer, task, temperature=0,
                    max_new_tokens=config["max_new_tokens"], max_sequence_length=config["max_sequence_length"],
                    enable_python=config.get("python_tool", False)).reward["correctness"] for task in subset]
                append_jsonl(out / "dev.jsonl", {"step": step + 1, "dataset": dataset, "split": "dev",
                    "n": len(values), "mean": sum(values) / len(values)})
    return out
