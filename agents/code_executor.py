"""Code Executor Agent: validate experiment code, then run it in a restricted subprocess.

Layers of defence (best-effort, NOT a hard security boundary):
  1. AST validation: syntax check, import allow-list, banned builtins, no dunder tricks.
  2. Separate interpreter process in isolated mode (-I) with a scrubbed environment and a
     throw-away working directory.
  3. Runtime import guard inside the child (allow-list applied to the user's own imports).
  4. POSIX resource limits (CPU time, address space, file size, open files) plus a wall-clock timeout.
  5. Output truncation.
For untrusted code in production, run this inside a container / gVisor / Firecracker VM instead.
Resource limits are skipped on Windows (only the timeout applies).
"""
from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ALLOWED_IMPORTS = frozenset({
    "math", "random", "statistics", "itertools", "collections", "functools", "json", "re", "string",
    "datetime", "decimal", "fractions", "dataclasses", "typing", "numpy", "pandas", "scipy", "sklearn"})
BANNED_CALLS = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__", "breakpoint", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "help", "exit", "quit", "memoryview"})
MAX_CODE_CHARS = 50_000

_RUNNER = r'''
import builtins, os, sys
code_path, cpu, mem_mb, fsize_mb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
allowed = set(os.environ.get("SANDBOX_ALLOWED", "").split(","))
try:
    import resource
    for name, val in (("RLIMIT_CPU", cpu), ("RLIMIT_AS", mem_mb * 1024 * 1024),
                      ("RLIMIT_FSIZE", fsize_mb * 1024 * 1024), ("RLIMIT_NOFILE", 64), ("RLIMIT_CORE", 0)):
        try:
            resource.setrlimit(getattr(resource, name), (val, val))
        except (ValueError, OSError, AttributeError):
            pass
except ImportError:
    pass
_real_import = builtins.__import__
def _guard(name, globals=None, locals=None, fromlist=(), level=0):
    if globals is not None and globals.get("__sandbox_user__"):
        if level != 0 or name.split(".")[0] not in allowed:
            raise ImportError("import of %r is not permitted in the sandbox" % name)
    return _real_import(name, globals, locals, fromlist, level)
builtins.__import__ = _guard
with open(code_path, encoding="utf-8") as fh:
    source = fh.read()
import traceback
try:
    exec(compile(source, "<experiment>", "exec"), {"__name__": "__main__", "__sandbox_user__": True})
except SystemExit:
    raise
except BaseException:
    etype, err, tb = sys.exc_info()
    traceback.print_exception(etype, err, tb.tb_next)  # hide the runner frame
    sys.exit(1)
'''


class ExecutionResult(BaseModel):
    status: Literal["success", "error", "timeout", "rejected"]
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    duration_seconds: float = 0.0
    validation_errors: list[str] = Field(default_factory=list)
    output_truncated: bool = False
    note: str = ("Executed in a restricted subprocess; this is defence-in-depth, not a hard security "
                 "boundary. Use a container for untrusted code.")


class _Validator(ast.NodeVisitor):
    def __init__(self, allowed: frozenset[str]) -> None:
        self.allowed, self.errors = allowed, []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] not in self.allowed:
                self.errors.append(f"line {node.lineno}: import of '{alias.name}' is not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or (node.module or "").split(".")[0] not in self.allowed:
            self.errors.append(f"line {node.lineno}: import from '{node.module or '.'}' is not allowed")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            self.errors.append(f"line {node.lineno}: call to '{node.func.id}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.errors.append(f"line {node.lineno}: access to dunder attribute '{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") and node.id not in ("__name__",):
            self.errors.append(f"line {node.lineno}: use of '{node.id}' is not allowed")


class CodeExecutorAgent:
    def __init__(self, timeout_seconds: int = 10, memory_mb: int = 1024, max_output_chars: int = 20000,
                 allowed_imports: frozenset[str] = ALLOWED_IMPORTS) -> None:
        self.timeout, self.memory_mb = timeout_seconds, memory_mb
        self.max_output_chars = max_output_chars
        self.allowed_imports = allowed_imports

    def validate(self, code: str) -> list[str]:
        """Return a list of validation errors (empty list means the code may be executed)."""
        if not isinstance(code, str) or not code.strip():
            return ["code is empty"]
        if len(code) > MAX_CODE_CHARS:
            return [f"code exceeds {MAX_CODE_CHARS} characters"]
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [f"syntax error at line {exc.lineno}: {exc.msg}"]
        v = _Validator(self.allowed_imports)
        v.visit(tree)
        return v.errors

    def _truncate(self, text: str) -> tuple[str, bool]:
        if len(text) > self.max_output_chars:
            return text[: self.max_output_chars] + "\n...[output truncated]", True
        return text, False

    def execute(self, code: str) -> ExecutionResult:
        errors = self.validate(code)
        if errors:
            return ExecutionResult(status="rejected", validation_errors=errors)
        env = {"PATH": os.defpath, "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg",
               "SANDBOX_ALLOWED": ",".join(sorted(self.allowed_imports))}
        for keep in ("SYSTEMROOT", "LD_LIBRARY_PATH"):
            if keep in os.environ:
                env[keep] = os.environ[keep]
        with tempfile.TemporaryDirectory(prefix="sandbox_") as tmp:
            env["HOME"] = tmp
            code_file, runner = Path(tmp, "experiment.py"), Path(tmp, "_runner.py")
            code_file.write_text(code, encoding="utf-8")
            runner.write_text(_RUNNER, encoding="utf-8")
            cmd = [sys.executable, "-I", str(runner), str(code_file), str(self.timeout + 2),
                   str(self.memory_mb), "10"]
            start = time.monotonic()
            try:
                proc = subprocess.run(cmd, cwd=tmp, env=env, stdin=subprocess.DEVNULL,
                                      capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                out, t1 = self._truncate(out)
                err, t2 = self._truncate(err)
                return ExecutionResult(status="timeout", stdout=out, stderr=err, duration_seconds=round(time.monotonic() - start, 3),
                                       output_truncated=t1 or t2, note=f"Killed after exceeding the {self.timeout}s wall-clock limit.")
            except OSError as exc:
                return ExecutionResult(status="error", stderr=f"could not start sandbox: {exc}")
            duration = round(time.monotonic() - start, 3)
        out, t1 = self._truncate(proc.stdout)
        err, t2 = self._truncate(proc.stderr)
        result = ExecutionResult(status="success" if proc.returncode == 0 else "error", stdout=out, stderr=err,
                                 return_code=proc.returncode, duration_seconds=duration, output_truncated=t1 or t2)
        if proc.returncode < 0:
            result.note = f"Process terminated by signal {-proc.returncode} (possibly a resource limit)."
        return result
