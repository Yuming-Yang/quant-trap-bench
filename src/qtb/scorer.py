"""Deterministic scoring of one answer against its key.

No judge model is involved. Each requested number is classified as

- ``corrected``: within tolerance of the corrected value;
- ``naive``: within tolerance of the naive value (trap tasks only);
- ``other``: neither. A wrong answer for an unrelated reason is not evidence about the
  trap, so it is counted separately from ``naive``.

The task outcome is the outcome of the key's primary field, unless the answer file is
absent (``missing``) or unusable (``malformed``). Only ``corrected`` earns credit.

An answer is malformed if it is not a JSON object, or if any requested field is absent
or is not a finite number. Numbers given as strings are rejected on purpose: the request
asks for plain numbers, and parsing "12%" or "0.12" leniently would let the scorer guess
units on the agent's behalf. Extra keys are ignored.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from qtb.task import ANSWER_FILE, AnswerField, AnswerKey, Kind


class Outcome(StrEnum):
    CORRECTED = "corrected"
    NAIVE = "naive"
    OTHER = "other"
    MALFORMED = "malformed"
    MISSING = "missing"


@dataclass(frozen=True)
class Score:
    task_id: str
    kind: Kind
    outcome: Outcome
    #: Outcome of every requested field; empty when the answer is missing or malformed.
    fields: dict[str, Outcome] = field(default_factory=dict)
    #: The numbers as given, for the same fields.
    values: dict[str, float] = field(default_factory=dict)
    detail: str = ""

    @property
    def credit(self) -> int:
        return int(self.outcome is Outcome.CORRECTED)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "kind": str(self.kind),
            "outcome": str(self.outcome),
            "credit": self.credit,
            "fields": {k: str(v) for k, v in self.fields.items()},
            "values": self.values,
            "detail": self.detail,
        }


# Tolerances are inclusive. The slack keeps a value exactly one tolerance away inside
# despite floating-point rounding (0.40 + 0.05 - 0.40 is slightly above 0.05).
_SLACK = 1.0 + 1e-9


def classify(value: float, spec: AnswerField) -> Outcome:
    """Classify one finite number."""
    if abs(value - spec.corrected) <= spec.tolerance * _SLACK:
        return Outcome.CORRECTED
    if spec.naive is not None and abs(value - spec.naive) <= spec.tolerance * _SLACK:
        return Outcome.NAIVE
    return Outcome.OTHER


def score_text(text: str | None, key: AnswerKey) -> Score:
    """Score the contents of an answer file; ``None`` means there was no file."""
    if text is None:
        return Score(key.task_id, key.kind, Outcome.MISSING, detail=f"no {ANSWER_FILE}")
    try:
        answer = json.loads(text)
    except json.JSONDecodeError as exc:
        return _malformed(key, f"not valid JSON: {exc.msg}")
    if not isinstance(answer, dict):
        return _malformed(key, f"expected a JSON object, got {type(answer).__name__}")

    values: dict[str, float] = {}
    for spec in key.fields:
        if spec.name not in answer:
            return _malformed(key, f"missing field {spec.name!r}")
        raw = answer[spec.name]
        # bool is a subclass of int; true/false is not a number here.
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            return _malformed(key, f"field {spec.name!r} is not a number: {raw!r}")
        if not math.isfinite(raw):
            return _malformed(key, f"field {spec.name!r} is not finite: {raw!r}")
        values[spec.name] = float(raw)

    outcomes = {spec.name: classify(values[spec.name], spec) for spec in key.fields}
    return Score(key.task_id, key.kind, outcomes[key.primary], fields=outcomes, values=values)


def score_workdir(workdir: Path, key: AnswerKey) -> Score:
    """Score the answer an agent left in its working directory."""
    path = Path(workdir) / ANSWER_FILE
    if not path.is_file():
        return score_text(None, key)
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return _malformed(key, "answer file is not UTF-8 text")
    return score_text(text, key)


def _malformed(key: AnswerKey, detail: str) -> Score:
    return Score(key.task_id, key.kind, Outcome.MALFORMED, detail=detail)
