"""Saved runs and their summary.

Each run is one JSON file: ``<out>/<agent>/<task_id>/run-<k>.json``, holding the score,
the agent's metadata and, for model agents, the full transcript and token counts. The
summary is computed from these files only, so every reported number can be traced to a
transcript.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qtb.agents import Agent
from qtb.runner import RunResult, run_task
from qtb.scorer import Outcome
from qtb.task import Task


def run_path(out_dir: Path, agent: str, task_id: str, repeat: int) -> Path:
    return Path(out_dir) / agent / task_id / f"run-{repeat}.json"


def run_and_record(agent: Agent, task: Task, out_dir: Path, repeat: int) -> RunResult:
    """Run once and save the record. An existing record is never overwritten."""
    path = run_path(out_dir, agent.name, task.task_id, repeat)
    if path.exists():
        raise FileExistsError(f"{path} exists; runs are not overwritten")
    result = run_task(agent, task)
    record = {
        "task_id": task.task_id,
        "kind": str(task.key.kind),
        "family": str(task.key.family),
        "trap": task.key.trap,
        "mirrors": task.key.mirrors,
        "repeat": repeat,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "score": result.score.to_dict(),
        "meta": result.meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str) + "\n")
    return result


def load_records(agent_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(Path(agent_dir).glob("*/run-*.json"))]


@dataclass(frozen=True)
class Summary:
    agent: str
    n_runs: int
    trap_outcomes: Counter
    control_outcomes: Counter
    by_trap: dict[str, Counter]
    usage: dict[str, int]
    steps: float

    def share(self, counter: Counter, outcome: Outcome) -> float:
        total = sum(counter.values())
        return counter[str(outcome)] / total if total else float("nan")

    def lines(self) -> list[str]:
        trap_n, control_n = sum(self.trap_outcomes.values()), sum(self.control_outcomes.values())
        t = self.trap_outcomes
        failed = self.share(t, Outcome.MALFORMED) + self.share(t, Outcome.MISSING)
        out = [
            f"agent: {self.agent}  runs: {self.n_runs}  mean steps: {self.steps:.1f}",
            f"trap runs ({trap_n}): corrected {self.share(t, Outcome.CORRECTED):.0%}"
            f"  naive {self.share(t, Outcome.NAIVE):.0%}"
            f"  other {self.share(t, Outcome.OTHER):.0%}"
            f"  malformed or missing {failed:.0%}",
            f"control runs ({control_n}): correct "
            f"{self.share(self.control_outcomes, Outcome.CORRECTED):.0%}",
            "by trap (corrected / naive / other / total):",
        ]
        for trap, counter in sorted(self.by_trap.items()):
            out.append(
                f"  {trap:<18}{counter['corrected']:>3} /{counter['naive']:>3} /"
                f"{counter['other'] + counter['malformed'] + counter['missing']:>3} /"
                f"{sum(counter.values()):>3}"
            )
        out.append("tokens: " + ", ".join(f"{k} {v:,}" for k, v in self.usage.items()))
        return out


def summarize(agent_dir: Path) -> Summary:
    records = load_records(agent_dir)
    trap, control = Counter(), Counter()
    by_trap: dict[str, Counter] = defaultdict(Counter)
    usage: Counter = Counter()
    steps = []
    for r in records:
        outcome = r["score"]["outcome"]
        if r["kind"] == "trap":
            trap[outcome] += 1
            by_trap[r["trap"]][outcome] += 1
        else:
            control[outcome] += 1
        usage.update(r["meta"].get("usage", {}))
        if "steps" in r["meta"]:
            steps.append(r["meta"]["steps"])
    return Summary(
        agent=Path(agent_dir).name,
        n_runs=len(records),
        trap_outcomes=trap,
        control_outcomes=control,
        by_trap=dict(by_trap),
        usage=dict(usage),
        steps=sum(steps) / len(steps) if steps else 0.0,
    )
