"""Run an agent on a task in a fresh temporary working directory, then score it."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from qtb.agents import Agent
from qtb.scorer import Score, score_workdir
from qtb.task import Task


@dataclass(frozen=True)
class RunResult:
    task_id: str
    agent: str
    score: Score
    meta: dict


def run_task(agent: Agent, task: Task) -> RunResult:
    """Copy the public files to a new temporary directory, let the agent work, score.

    The directory is removed afterwards. It isolates files only; see the README on what
    the harness does not sandbox.
    """
    with tempfile.TemporaryDirectory(prefix="qtb-") as tmp:
        workdir = Path(tmp) / "work"
        shutil.copytree(task.public_dir, workdir)
        meta = agent.solve(workdir, task)
        score = score_workdir(workdir, task.key)
    return RunResult(task.task_id, agent.name, score, meta)
