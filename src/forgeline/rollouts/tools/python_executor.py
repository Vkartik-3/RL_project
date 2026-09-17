"""Sandboxed-by-subprocess Python executor with a hard timeout and output cap."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import dataclass

TIMEOUT_SEC = 5.0
MAX_OUTPUT_CHARS = 500


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    timed_out: bool
    error: bool

    @property
    def output(self) -> str:
        if self.timed_out:
            return f"ERROR: execution timed out ({TIMEOUT_SEC:.0f}s limit)"
        if self.error and not self.stdout:
            return f"ERROR: {self.stderr[:200]}"
        out = self.stdout[:MAX_OUTPUT_CHARS]
        if self.stderr and not self.error:
            out = out.rstrip() + f"\n# stderr: {self.stderr[:100]}"
        return out


def execute_python(code: str, timeout: float = TIMEOUT_SEC) -> ExecutionResult:
    """Run ``code`` in a fresh interpreter process; never raises."""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", textwrap.dedent(code)],
            capture_output=True, text=True, timeout=timeout,
        )
        return ExecutionResult(stdout=result.stdout.strip(), stderr=result.stderr.strip(),
                               timed_out=False, error=result.returncode != 0)
    except subprocess.TimeoutExpired:
        return ExecutionResult("", "", timed_out=True, error=True)
    except Exception as exc:  # pragma: no cover - defensive
        return ExecutionResult("", str(exc), timed_out=False, error=True)


class PythonExecutorTool:
    name = "python_executor"
    description = "Executes Python code and returns stdout."

    def __init__(self, timeout: float = TIMEOUT_SEC):
        self.timeout = timeout

    def __call__(self, args: dict) -> str:
        code = args.get("code", "print('')") if isinstance(args, dict) else str(args)
        return execute_python(code, timeout=self.timeout).output
