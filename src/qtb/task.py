"""The task format: what an agent sees, and the answer key it is scored against.

A task is a directory::

    <task_id>/
        public/            copied into the agent's working directory, nothing else is
            REQUEST.md     the research request and the answer format
            DATA_DICTIONARY.md
            *.csv
        key.json           the hidden answer key; never shown to the agent

The key holds, for every requested number, the **corrected** value (what the analysis
gives when the trap is handled), the **naive** value (what it gives when the data is
taken at face value) and an absolute tolerance. Control tasks have no trap, so their
naive value is ``None``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

PUBLIC_DIR = "public"
KEY_FILE = "key.json"
REQUEST_FILE = "REQUEST.md"
DICTIONARY_FILE = "DATA_DICTIONARY.md"
ANSWER_FILE = "answer.json"


class TaskError(ValueError):
    """A task or answer key is malformed."""


class Kind(StrEnum):
    TRAP = "trap"
    CONTROL = "control"


class Family(StrEnum):
    DATA = "data"
    METHOD = "method"


@dataclass(frozen=True)
class AnswerField:
    """One number the request asks for."""

    name: str
    description: str
    corrected: float
    naive: float | None
    tolerance: float

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise TaskError(f"field name {self.name!r} must be a valid identifier")
        if not (math.isfinite(self.tolerance) and self.tolerance > 0):
            raise TaskError(f"{self.name}: tolerance must be positive and finite")
        if not math.isfinite(self.corrected):
            raise TaskError(f"{self.name}: corrected value must be finite")
        if self.naive is not None and not math.isfinite(self.naive):
            raise TaskError(f"{self.name}: naive value must be finite")

    @property
    def separation(self) -> float | None:
        """Distance between the corrected and naive values, in tolerances."""
        if self.naive is None:
            return None
        return abs(self.corrected - self.naive) / self.tolerance


@dataclass(frozen=True)
class AnswerKey:
    """Hidden key of one task.

    ``trap`` names the trap the task plants. A control task has none, but records in
    ``mirrors`` which trap task it was built to resemble, so that false alarms can be
    reported next to the trap they would come from.

    ``primary`` is the field that decides the outcome. Other fields are scored and
    recorded but do not change the credit.

    ``params`` holds the parameters the request was written from (window, portfolio
    size, which statistic). Only the scripted reference agents read them; a model under
    test sees the request text and nothing from the key.
    """

    task_id: str
    kind: Kind
    family: Family
    trap: str | None
    mirrors: str
    primary: str
    fields: tuple[AnswerField, ...]
    seed: int
    notes: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", Kind(self.kind))
        object.__setattr__(self, "family", Family(self.family))
        names = [f.name for f in self.fields]
        if not names:
            raise TaskError(f"{self.task_id}: a key needs at least one field")
        if len(set(names)) != len(names):
            raise TaskError(f"{self.task_id}: field names must be unique")
        if self.primary not in names:
            raise TaskError(f"{self.task_id}: primary field {self.primary!r} is not a field")
        if self.kind is Kind.TRAP:
            if not self.trap:
                raise TaskError(f"{self.task_id}: a trap task must name its trap")
            primary = self.field(self.primary)
            if primary.naive is None:
                raise TaskError(f"{self.task_id}: a trap task's primary field needs a naive value")
            # Acceptance criterion: an answer cannot be within tolerance of both values,
            # and a small slip cannot move a corrected answer onto the naive one.
            if primary.separation <= 2.0:
                raise TaskError(
                    f"{self.task_id}: corrected and naive values are only "
                    f"{primary.separation:.2f} tolerances apart; at least 2 are required"
                )
        else:
            if self.trap is not None:
                raise TaskError(f"{self.task_id}: a control task has no trap")
            if any(f.naive is not None for f in self.fields):
                raise TaskError(f"{self.task_id}: control fields have no naive value")

    def field(self, name: str) -> AnswerField:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def to_json(self) -> str:
        data = asdict(self)
        data["kind"] = str(self.kind)
        data["family"] = str(self.family)
        data["params"] = dict(self.params)
        data["extra"] = dict(self.extra)
        return json.dumps(data, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> AnswerKey:
        data = json.loads(text)
        data["fields"] = tuple(AnswerField(**f) for f in data["fields"])
        return cls(**data)


@dataclass(frozen=True)
class Task:
    """A task on disk."""

    root: Path
    key: AnswerKey

    @property
    def task_id(self) -> str:
        return self.key.task_id

    @property
    def public_dir(self) -> Path:
        return self.root / PUBLIC_DIR

    @property
    def request(self) -> str:
        return (self.public_dir / REQUEST_FILE).read_text()


def answer_format(fields: tuple[AnswerField, ...]) -> str:
    """The part of every request that says what to write and where."""
    lines = [
        "## Answer format",
        "",
        f"Write a JSON object to `{ANSWER_FILE}` in the working directory with these keys,",
        "each a plain number (not a string, not a percentage sign):",
        "",
    ]
    lines += [f"- `{f.name}`: {f.description}" for f in fields]
    lines += ["", "Then give a short written report of what you did and why."]
    return "\n".join(lines) + "\n"


def write_task(
    root: Path,
    key: AnswerKey,
    request: str,
    dictionary: str,
    files: Mapping[str, pd.DataFrame | str],
) -> Task:
    """Write a task directory. ``request`` gets the answer format appended."""
    public = root / PUBLIC_DIR
    public.mkdir(parents=True, exist_ok=False)
    reserved = {REQUEST_FILE, DICTIONARY_FILE, ANSWER_FILE}
    for name, content in files.items():
        if name in reserved or "/" in name or name.startswith("."):
            raise TaskError(f"{key.task_id}: invalid public file name {name!r}")
        path = public / name
        if isinstance(content, pd.DataFrame):
            content.to_csv(path, index=False, float_format="%.10g")
        else:
            path.write_text(content)
    (public / REQUEST_FILE).write_text(request.rstrip() + "\n\n" + answer_format(key.fields))
    (public / DICTIONARY_FILE).write_text(dictionary.rstrip() + "\n")
    (root / KEY_FILE).write_text(key.to_json())
    return Task(root=root, key=key)


def load_task(root: Path) -> Task:
    root = Path(root)
    if not (root / PUBLIC_DIR / REQUEST_FILE).is_file():
        raise TaskError(f"{root}: no {PUBLIC_DIR}/{REQUEST_FILE}")
    key = AnswerKey.from_json((root / KEY_FILE).read_text())
    if key.task_id != root.name:
        raise TaskError(f"{root}: directory name does not match task id {key.task_id!r}")
    return Task(root=root, key=key)


def load_tasks(directory: Path) -> list[Task]:
    """Every task directly under ``directory``, sorted by id."""
    roots = sorted(p for p in Path(directory).iterdir() if (p / KEY_FILE).is_file())
    return [load_task(p) for p in roots]
