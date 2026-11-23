"""Bounded deterministic tools; Python is fail-closed unless bwrap is usable."""
import ast
from collections import Counter
import math
import operator
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def calculator(expr):
    if not isinstance(expr, str) or len(expr) > 256:
        raise ValueError("expression too long")
    tree = ast.parse(expr, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("expression too complex")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            result = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            result = ops[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        else:
            raise ValueError("only numeric constants and + - * / allowed")
        if not math.isfinite(result) or abs(result) > 1e15:
            raise ValueError("result outside supported finite range")
        return result
    return str(visit(tree.body))


def search(query, documents, k=2):
    if not isinstance(query, str) or len(query) > 1024:
        raise ValueError("invalid search query")
    corpus = [_tokenize(doc["title"] + " " + doc["text"]) for doc in documents]
    if not corpus:
        return []
    query_terms = set(_tokenize(query))
    if not query_terms:
        return []
    frequencies = [Counter(tokens) for tokens in corpus]
    avg = sum(map(len, corpus)) / len(corpus) or 1
    idfs = {
        term: math.log(1 + (len(corpus) - df + 0.5) / (df + 0.5))
        for term in query_terms
        for df in (sum(term in counts for counts in frequencies),)
    }
    scores = []
    for index, tokens in enumerate(corpus):
        counts = frequencies[index]
        norm = 1.5 * (0.25 + 0.75 * len(tokens) / avg)
        score = sum(idfs[term] * counts[term] * 2.5 / (counts[term] + norm) for term in query_terms if counts[term])
        scores.append((score, index))
    return [documents[index] for score, index in sorted(scores, key=lambda pair: (-pair[0], pair[1]))[:k] if score > 0]


def python_sandbox(code, timeout=5.0):
    """Read-only minimal Linux mount namespace, no network, bounded output/time.

    This never falls back to executing the code without bubblewrap. Colab often
    disables user namespaces: callers must omit Python when probe fails.
    """
    if sys.platform != "linux" or not shutil.which("bwrap"):
        raise RuntimeError("Python tool unavailable: Linux bubblewrap required")
    if not isinstance(code, str) or len(code) > 8192:
        raise ValueError("Python source exceeds 8192 characters")
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
    for directory in ("/usr", "/lib", "/lib64", "/bin"):
        if Path(directory).exists():
            cmd += ["--ro-bind", directory, directory]
    # No /home, /srv, /content, /tmp, /proc, devices, credentials, or writable mount.
    cmd += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"] if Path("/etc/ld.so.cache").exists() else []
    cmd += ["--remount-ro", "/", "--", "/usr/bin/python3", "-I", "-S", "-B", "-c", code]
    def limits():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        resource.setrlimit(resource.RLIMIT_NPROC, (8, 8))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True, preexec_fn=limits)
    chunks = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    started = time.monotonic()
    try:
        while selector.get_map():
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Python tool exceeded deadline")
            for key, _ in selector.select(min(remaining, 0.1)):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    selector.unregister(key.fileobj)
                chunks.extend(chunk)
                if len(chunks) > 8192:
                    raise ValueError("Python tool output exceeded 8192 bytes")
        proc.wait(timeout=max(0.01, timeout - (time.monotonic() - started)))
        if proc.returncode:
            raise RuntimeError("Python sandbox failed: " + chunks.decode(errors="replace")[:1000])
        return chunks.decode(errors="replace")
    finally:
        selector.close()
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        proc.stdout.close()


def probe_python_sandbox():
    probe = """import os,socket
try:
 open('/probe','w').write('x')
 raise AssertionError('write succeeded')
except OSError: pass
try:
 socket.create_connection(('1.1.1.1',443),timeout=.2)
 raise AssertionError('network succeeded')
except OSError: pass
assert not os.path.exists('/home') and not os.path.exists('/content')
print('SANDBOX_OK')"""
    try:
        return python_sandbox(probe).strip() == "SANDBOX_OK"
    except (RuntimeError, ValueError, TimeoutError, OSError):
        return False


def tool_schemas(enable_python=False):
    definitions = [("calculator", "Evaluate bounded arithmetic using + - * /.", "expr"),
                   ("search", "Search only documents attached to this question; return top 2.", "query")]
    if enable_python:
        definitions.append(("python", "Run Python in read-only Linux isolation without network, max 5 seconds.", "code"))
    return [{"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": {arg: {"type": "string"}}, "required": [arg], "additionalProperties": False}}}
        for name, description, arg in definitions]


def execute(call, documents, enable_python=False):
    name, args = call["name"], call["arguments"]
    try:
        if name == "calculator" and set(args) == {"expr"}:
            return {"ok": True, "result": calculator(args["expr"])}
        if name == "search" and set(args) == {"query"}:
            return {"ok": True, "result": search(args["query"], documents)}
        if name == "python" and enable_python and set(args) == {"code"}:
            return {"ok": True, "result": python_sandbox(args["code"])}
        return {"ok": False, "error": "unavailable tool or invalid arguments"}
    except (ValueError, SyntaxError, ZeroDivisionError, RuntimeError, TimeoutError, OverflowError) as exc:
        return {"ok": False, "error": str(exc)[:1000]}
