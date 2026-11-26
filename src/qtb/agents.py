"""Scripted reference agents.

They validate the benchmark, not a model: the naive agent runs the textbook analysis on
the files as given, the oracle applies the reference correction. Both read only the
public files in their working directory. They are told which trap module the task came
from (a model under test is not), because their job is to show that the materials are
enough to reach each key value.

On a control task there is nothing to correct, so both agents run the plain analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from qtb.catalog import TRAPS
from qtb.task import ANSWER_FILE, Kind, Task


class Agent(Protocol):
    name: str

    def solve(self, workdir: Path, task: Task) -> dict:
        """Work in ``workdir`` (a copy of the task's public files) and leave an answer
        file there. Returns run metadata for the transcript."""
        ...


class ScriptedAgent:
    def __init__(self, name: str, corrects: bool) -> None:
        self.name = name
        self.corrects = corrects

    def solve(self, workdir: Path, task: Task) -> dict:
        module = TRAPS[task.key.mirrors]
        use_oracle = self.corrects and task.key.kind is Kind.TRAP
        solver = module.oracle if use_oracle else module.naive
        answer = solver(workdir, task.key.params)
        (workdir / ANSWER_FILE).write_text(json.dumps(answer))
        return {"agent": self.name, "solver": f"{module.__name__}.{solver.__name__}"}


NAIVE = ScriptedAgent("naive", corrects=False)
ORACLE = ScriptedAgent("oracle", corrects=True)
SCRIPTED = {agent.name: agent for agent in (NAIVE, ORACLE)}
