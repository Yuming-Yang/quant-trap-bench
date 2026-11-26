"""Command-line entry point.

qtb generate [--out tasks] [--force]   build every task from its seed
qtb check [--tasks tasks]              run the naive and oracle agents and validate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qtb import __version__
from qtb.agents import NAIVE, ORACLE, SCRIPTED
from qtb.catalog import generate
from qtb.harness import SYSTEM_PROMPTS, AgentConfig, LLMAgent
from qtb.records import run_and_record, summarize
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

    run = sub.add_parser("run", help="run an agent on tasks and save the records")
    run.add_argument("--agent", required=True, choices=[*SCRIPTED, "claude"])
    run.add_argument("--tasks", type=Path, default=Path("tasks"))
    run.add_argument("--task", action="append", dest="task_ids", help="only these task ids")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--out", type=Path, default=Path("runs"))
    run.add_argument("--model", default="claude-opus-5-5")
    run.add_argument("--effort", default="high")
    run.add_argument("--prompt", default="plain", choices=sorted(SYSTEM_PROMPTS))
    run.add_argument("--max-steps", type=int, default=30)
    run.add_argument("--allow-paid", action="store_true", help="required for --agent claude")

    summary = sub.add_parser("summarize", help="summarize the saved runs of one agent")
    summary.add_argument("agent_dir", type=Path)
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


def cmd_run(args: argparse.Namespace) -> int:
    if args.agent == "claude":
        if not args.allow_paid:
            print("--agent claude calls a paid API; add --allow-paid once the spend is approved")
            return 2
        from qtb.models.anthropic import AnthropicModel

        model = AnthropicModel(args.model, effort=args.effort)
        agent = LLMAgent(AgentConfig(model, prompt=args.prompt, max_steps=args.max_steps))
    else:
        agent = SCRIPTED[args.agent]
    tasks = load_tasks(args.tasks)
    if args.task_ids:
        wanted = set(args.task_ids)
        tasks = [t for t in tasks if t.task_id in wanted]
        missing = wanted - {t.task_id for t in tasks}
        if missing:
            print(f"unknown task ids: {', '.join(sorted(missing))}")
            return 1
    for repeat in range(1, args.repeats + 1):
        for task in tasks:
            result = run_and_record(agent, task, args.out, repeat)
            print(f"{task.task_id:<26} run {repeat}: {result.score.outcome}")
    return 0


def cmd_summarize(agent_dir: Path) -> int:
    summary = summarize(agent_dir)
    if summary.n_runs == 0:
        print(f"no runs under {agent_dir}")
        return 1
    print("\n".join(summary.lines()))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return cmd_generate(args.out, args.force)
    if args.command == "check":
        return cmd_check(args.tasks)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "summarize":
        return cmd_summarize(args.agent_dir)
    return 0
