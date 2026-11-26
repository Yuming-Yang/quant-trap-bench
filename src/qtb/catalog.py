"""The task list, and generation of every task from its seed.

Each entry names a trap module, a base seed and the request parameters. Generation tries
the base seed first. If the trap does not move the answer by more than two tolerances
for that draw (the key refuses such a task), it tries the next seed in a fixed sequence
and records how many draws it took. A control that records a likely over-correction
(``false_alarm`` in the key) is held to the same rule against that value. The sequence
is fixed, so the same catalog always gives the same tasks.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any

from qtb.task import Kind, Task, TaskError, write_task
from qtb.traps import (
    costs,
    ic_alignment,
    lookahead,
    multiple_testing,
    point_in_time,
    selection,
    splits,
    survivorship,
)
from qtb.traps.base import Built

TRAPS: dict[str, ModuleType] = {
    module.TRAP: module
    for module in (
        survivorship,
        point_in_time,
        splits,
        lookahead,
        costs,
        selection,
        multiple_testing,
        ic_alignment,
    )
}
MAX_DRAWS = 20
SEED_STEP = 1000


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    trap: str
    kind: Kind
    seed: int
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def module(self) -> ModuleType:
        return TRAPS[self.trap]


def _trap(task_id: str, trap: str, seed: int, **params: Any) -> TaskSpec:
    return TaskSpec(task_id, trap, Kind.TRAP, seed, params)


def _control(task_id: str, trap: str, seed: int, **params: Any) -> TaskSpec:
    return TaskSpec(task_id, trap, Kind.CONTROL, seed, params)


CATALOG: tuple[TaskSpec, ...] = (
    _trap("survivorship_1", "survivorship", 101, statistic="mean"),
    _trap("survivorship_2", "survivorship", 102, statistic="sharpe", n_assets=160),
    _control("survivorship_control", "survivorship", 103, statistic="mean"),
    _trap("point_in_time_1", "point_in_time", 201),
    _trap(
        "point_in_time_2", "point_in_time", 202, n_assets=250, lag_months=(2, 3), announcement=0.35
    ),
    _control("point_in_time_control", "point_in_time", 203, ic=0.06),
    _trap("splits_1", "splits", 301, statistic="sharpe"),
    _trap("splits_2", "splits", 302, statistic="vol", n_assets=10),
    _control("splits_control", "splits", 303, statistic="sharpe"),
    _trap("lookahead_1", "lookahead", 401),
    _trap("lookahead_2", "lookahead", 402, n_assets=60, k=12),
    _control("lookahead_control", "lookahead", 403),
    _trap("costs_1", "costs", 501, phrasing="net"),
    _trap("costs_2", "costs", 502, phrasing="investor", k=6),
    # Method traps, on factor-research-pipeline's synthetic panel.
    _trap("selection_1", "selection", 601),
    _trap("selection_2", "selection", 602, n_assets=250, n_groups=10),
    _control("selection_control", "selection", 603, lookback=6),
    _trap("multiple_testing_1", "multiple_testing", 701),
    _trap("multiple_testing_2", "multiple_testing", 702, n_variants=100, n_days=504),
    _trap("ic_alignment_1", "ic_alignment", 801, signal="momentum"),
    _trap("ic_alignment_2", "ic_alignment", 802, signal="planted", persistence=0.3, ic=0.06),
    _control(
        "ic_alignment_control", "ic_alignment", 803, signal="planted", persistence=0.3, ic=0.06
    ),
)


def build(spec: TaskSpec) -> Built:
    """Build one task, moving to the next seed if the trap does not bite on this draw."""
    control = spec.kind is Kind.CONTROL
    last_error: TaskError | None = None
    for draw in range(MAX_DRAWS):
        seed = spec.seed + draw * SEED_STEP
        try:
            built = spec.module.build(spec.task_id, seed, control, spec.params)
        except TaskError as exc:
            last_error = exc
            continue
        if control and not _detects_false_alarm(built):
            continue
        extra = dict(built.key.extra) | {"draws": float(draw + 1)}
        return replace(built, key=replace(built.key, extra=extra))
    raise TaskError(f"{spec.task_id}: no usable draw in {MAX_DRAWS} seeds") from last_error


def _detects_false_alarm(built: Built) -> bool:
    """A control that records a likely over-correction must be able to tell it apart.

    The same two-tolerance rule as for trap tasks, applied to the false-alarm value.
    """
    if "false_alarm" not in built.key.extra:
        return True
    spec = built.key.field(built.key.primary)
    return abs(built.key.extra["false_alarm"] - spec.corrected) > 2 * spec.tolerance


def generate(
    out_dir: Path, specs: tuple[TaskSpec, ...] = CATALOG, force: bool = False
) -> list[Task]:
    """Write every task under ``out_dir``. Existing folders are replaced only with ``force``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for spec in specs:
        root = out_dir / spec.task_id
        if root.exists():
            if not force:
                raise FileExistsError(f"{root} exists; pass force=True to rebuild it")
            shutil.rmtree(root)
        built = build(spec)
        tasks.append(write_task(root, built.key, built.request, built.dictionary, built.files))
    return tasks


def spec_for(task_id: str) -> TaskSpec:
    for spec in CATALOG:
        if spec.task_id == task_id:
            return spec
    raise KeyError(task_id)
