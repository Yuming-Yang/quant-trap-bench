"""The one tool an agent gets: run Python in its working directory.

Each call starts a fresh interpreter (the one running the harness, so numpy and pandas
are available) with the working directory as its current directory and a minimal
environment. Nothing persists between calls except files; the tool description says so.

This isolates *files the agent is meant to see*, not the machine. The process can read
anything the user running the harness can read and can open network connections. It is
not a security sandbox (no Docker on the development machine; SPEC decision D2). The
environment is stripped so that API keys in the harness's environment are not visible to
the code the model writes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_CHARS = 8_000
DEFAULT_TIMEOUT_S = 120

PYTHON_TOOL = {
    "name": "python",
    "description": (
        "Run Python 3 code in the working directory and return its standard output and "
        "standard error. numpy and pandas are installed. Each call runs in a new process: "
        "variables do not persist between calls, so save anything you need later to a "
        f"file. Output longer than {MAX_OUTPUT_CHARS} characters is truncated, and a call "
        f"is stopped after {DEFAULT_TIMEOUT_S} seconds."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to run."}},
        "required": ["code"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class ToolResult:
    output: str
    is_error: bool
    exit_code: int | None
    timed_out: bool = False


def _environment(workdir: Path) -> dict[str, str]:
    """Only what an interpreter needs. No inherited secrets."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(workdir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "MPLBACKEND": "Agg",
        "LANG": "C.UTF-8",
    }


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    dropped = len(text) - limit
    return f"{text[:head]}\n... [{dropped} characters truncated] ...\n{text[-(limit - head) :]}"


def run_python(code: str, workdir: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> ToolResult:
    try:
        done = subprocess.run(
            [sys.executable, "-c", code],
            cwd=workdir,
            env=_environment(workdir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return ToolResult(
            truncate(partial + f"\n[stopped after {timeout_s:g} seconds]"), True, None, True
        )
    output = done.stdout
    if done.stderr:
        output += ("\n" if output else "") + done.stderr
    if not output:
        output = "(no output)"
    return ToolResult(truncate(output), done.returncode != 0, done.returncode)


def execute(name: str, arguments: dict, workdir: Path, timeout_s: float) -> ToolResult:
    """Dispatch a tool call; unknown tools and bad arguments come back as errors."""
    if name != PYTHON_TOOL["name"]:
        return ToolResult(f"unknown tool {name!r}; the only tool is 'python'", True, None)
    code = arguments.get("code") if isinstance(arguments, dict) else None
    if not isinstance(code, str):
        return ToolResult("the python tool needs a string argument 'code'", True, None)
    return run_python(code, workdir, timeout_s)
