"""Command-line entry point.

qtb generate [--out tasks] [--force]   build every task from its seed
qtb check [--tasks tasks]              run the naive and oracle agents and validate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qtb import __version__
from qtb.agents import NAIVE, ORACLE
from qtb.catalog import generate
from qtb.runner import run_task
from qtb.scorer import Outcome
from qtb.task import Kind, load_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qtb", description="QuantTrapBench")
    parser.add_argument("--version", action="version", version=f"qtb {__version__}")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="build every task from its seed")
    gen.add_argument("--out", type=Path, default=Path("tasks"))
    gen.add_argument("--force", action="store_true", help="replace existing task folders")

    check = sub.add_parser("check", help="validate the tasks with the scripted agents")
    check.add_argument("--tasks", type=Path, default=Path("tasks"))
    return parser


def cmd_generate(out: Path, force: bool) -> int:
    tasks = generate(out, force=force)
    n_control = sum(t.key.kind is Kind.CONTROL for t in tasks)
    print(f"wrote {len(tasks)} tasks ({n_control} controls) to {out}")
    return 0


def expected_outcome(agent: str, kind: Kind) -> Outcome:
    if agent == "naive" and kind is Kind.TRAP:
        return Outcome.NAIVE
    return Outcome.CORRECTED


def cmd_check(tasks_dir: Path) -> int:
    """Naive must be naive on every trap and correct on every control; oracle always correct."""
    tasks = load_tasks(tasks_dir)
    if not tasks:
        print(f"no tasks under {tasks_dir}; run `qtb generate` first")
        return 1
    failures = 0
    print(f"{'task':<26}{'kind':<9}{'naive':<11}{'oracle':<11}separation")
    for task in tasks:
        outcomes = {}
        for agent in (NAIVE, ORACLE):
            outcome = run_task(agent, task).score.outcome
            outcomes[agent.name] = outcome
            failures += outcome is not expected_outcome(agent.name, task.key.kind)
        separation = task.key.field(task.key.primary).separation
        sep = "-" if separation is None else f"{separation:.1f} tol"
        print(
            f"{task.task_id:<26}{task.key.kind:<9}{outcomes['naive']:<11}"
            f"{outcomes['oracle']:<11}{sep}"
        )
    print("ok" if failures == 0 else f"{failures} unexpected outcomes")
    return int(failures > 0)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return cmd_generate(args.out, args.force)
    if args.command == "check":
        return cmd_check(args.tasks)
    return 0
