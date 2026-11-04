"""Atomically saved checkpoint directories with argv-only, receipt-verified sync."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import uuid


class CheckpointPersistenceError(RuntimeError):
    """The local checkpoint remains intact but its remote copy is not verified."""


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checkpoint_manifest(directory):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint symlinks are not allowed")
        if path.is_file() and path != directory / "checkpoint-manifest.json":
            files[str(path.relative_to(directory))] = file_sha256(path)
    if not files:
        raise ValueError("empty checkpoint")
    atomic_json(directory / "checkpoint-manifest.json", {"version": 1, "files": files})


def verify_checkpoint_files(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint must be a real directory")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("checkpoint must not contain symlinks")
    manifest = json.loads((directory / "checkpoint-manifest.json").read_text())
    if manifest.get("version") != 1 or not manifest.get("files"):
        raise ValueError("invalid checkpoint manifest")
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file() and path != directory / "checkpoint-manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("checkpoint file inventory differs from manifest")
    for filename, checksum in manifest["files"].items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or (directory / relative).is_symlink():
            raise ValueError("unsafe checkpoint path")
        if file_sha256(directory / relative) != checksum:
            raise ValueError("checkpoint checksum mismatch")
    return file_sha256(directory / "checkpoint-manifest.json")


def parse_sync_hook(value=None):
    value = os.environ.get("EXP_CHECKPOINT_SYNC") if value is None else value
    if value is None:
        return None
    try:
        argv = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("EXP_CHECKPOINT_SYNC must be a JSON argv array") from exc
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
        raise ValueError("checkpoint hook needs a nonempty argv array of strings")
    for names in [("{checkpoint_dir}", "{source}"), ("{checkpoint_id}", "{run_id}"), ("{receipt}",)]:
        if not any(name in arg for name in names for arg in argv):
            raise ValueError("checkpoint hook missing " + " or ".join(names))
    return argv


def require_checkpoint_sync(directory, *, formal):
    hook = parse_sync_hook()
    if hook is None and (formal or Path(directory).resolve().is_relative_to(Path('/content'))):
        raise CheckpointPersistenceError("Formal or Colab training requires EXP_CHECKPOINT_SYNC; configure verified rclone persistence before training")
    return hook


def persist_checkpoint(directory, *, timeout=3700):
    """Called only after a complete .partial generation has been atomically renamed.

    Returns local-only when optional local smoke has no hook. A hook is successful
    only if a fresh common-helper receipt identifies this exact immutable source.
    """
    directory = Path(directory).resolve()
    checksum = verify_checkpoint_files(directory)
    checkpoint_id = "E3-" + re.sub(r"[^A-Za-z0-9_.-]", "-", directory.parent.name)[:30] + "-" + directory.name[:25] + "-" + checksum[:12]
    state = {"checkpoint": str(directory), "checkpoint_id": checkpoint_id, "manifest_sha256": checksum, "state": "local_only"}
    status_path = directory.parent / "checkpoint-status.json"
    atomic_json(status_path, state)
    hook = parse_sync_hook()
    if hook is None:
        return state
    # Fresh attempt paths avoid stale proof and preserve the prior durable pointer
    # even when retrying the same checkpoint after a previously verified upload.
    receipt = directory.parent / "checkpoint-receipts" / (checkpoint_id + "-" + uuid.uuid4().hex[:12] + ".json")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    substitutions = {"{checkpoint_dir}": str(directory), "{source}": str(directory),
                     "{checkpoint_id}": checkpoint_id, "{run_id}": checkpoint_id,
                     "{receipt}": str(receipt.resolve()), "{experiment}": "E3"}
    argv = []
    for arg in hook:
        for key, value in substitutions.items():
            arg = arg.replace(key, value)
        argv.append(arg)
    try:
        process = subprocess.Popen(argv, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise CheckpointPersistenceError("checkpoint sync timed out") from None
        if code:
            raise CheckpointPersistenceError(f"checkpoint sync exited {code}")
        proof = json.loads(receipt.read_text())
        if (proof.get("state") != "verified" or proof.get("experiment") != "E3"
                or proof.get("run_id") != checkpoint_id or Path(proof.get("source", "")).resolve() != directory
                or not proof.get("destination") or not proof.get("verified_at")):
            raise CheckpointPersistenceError("checkpoint sync returned no matching verified receipt")
        if verify_checkpoint_files(directory) != checksum:
            raise CheckpointPersistenceError("checkpoint changed during upload")
        state.update(state="verified", receipt=str(receipt), destination=proof["destination"], verified_at=proof["verified_at"])
        # A prior durable pointer is preserved until verification of the new copy.
        atomic_json(directory.parent / "durable-checkpoint.json", state)
        atomic_json(status_path, state)
        return state
    except (OSError, ValueError, subprocess.SubprocessError, CheckpointPersistenceError) as exc:
        state.update(state="sync_failed", error_type=type(exc).__name__)
        atomic_json(status_path, state)
        raise CheckpointPersistenceError("Checkpoint persistence is not verified; local checkpoint and prior durable receipt retained. Do not release the runtime.") from None
