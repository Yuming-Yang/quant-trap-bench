import json

import pandas as pd
import pytest

from qtb.task import (
    DICTIONARY_FILE,
    KEY_FILE,
    PUBLIC_DIR,
    REQUEST_FILE,
    AnswerField,
    AnswerKey,
    TaskError,
    load_task,
    load_tasks,
    write_task,
)


def make_key(**overrides) -> AnswerKey:
    base = dict(
        task_id="t1",
        kind="trap",
        family="data",
        trap="lookahead",
        mirrors="lookahead",
        primary="x",
        fields=(AnswerField("x", "a number", 1.0, 2.0, 0.1),),
        seed=0,
    )
    base.update(overrides)
    return AnswerKey(**base)


def test_key_round_trips_through_json():
    key = make_key(extra={"n_delisted": 12.0}, notes="toy")
    again = AnswerKey.from_json(key.to_json())
    assert again == key
    assert json.loads(key.to_json())["kind"] == "trap"


def test_trap_key_requires_corrected_and_naive_more_than_two_tolerances_apart():
    with pytest.raises(TaskError, match="tolerances apart"):
        make_key(fields=(AnswerField("x", "a number", 1.0, 1.15, 0.1),))
    with pytest.raises(TaskError, match="tolerances apart"):
        make_key(fields=(AnswerField("x", "a number", 1.0, 0.8, 0.1),))
    assert make_key(fields=(AnswerField("x", "a number", 1.0, 1.21, 0.1),)).fields


def test_trap_key_needs_a_trap_and_a_naive_primary():
    with pytest.raises(TaskError, match="must name its trap"):
        make_key(trap=None)
    with pytest.raises(TaskError, match="naive value"):
        make_key(fields=(AnswerField("x", "a number", 1.0, None, 0.1),))


def test_control_key_has_no_trap_and_no_naive_values():
    with pytest.raises(TaskError, match="no trap"):
        make_key(kind="control", fields=(AnswerField("x", "n", 1.0, None, 0.1),))
    with pytest.raises(TaskError, match="no naive"):
        make_key(kind="control", trap=None)
    key = make_key(kind="control", trap=None, fields=(AnswerField("x", "n", 1.0, None, 0.1),))
    assert key.field("x").separation is None


@pytest.mark.parametrize(
    ("fields", "primary", "match"),
    [
        ((), "x", "at least one field"),
        ((AnswerField("x", "n", 1.0, 2.0, 0.1),) * 2, "x", "unique"),
        ((AnswerField("x", "n", 1.0, 2.0, 0.1),), "y", "not a field"),
    ],
)
def test_key_field_checks(fields, primary, match):
    with pytest.raises(TaskError, match=match):
        make_key(fields=fields, primary=primary)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="not valid"),
        dict(tolerance=0.0),
        dict(tolerance=float("nan")),
        dict(corrected=float("inf")),
        dict(naive=float("nan")),
    ],
)
def test_answer_field_checks(kwargs):
    base = dict(name="x", description="n", corrected=1.0, naive=2.0, tolerance=0.1)
    base.update(kwargs)
    with pytest.raises(TaskError):
        AnswerField(**base)


def test_write_and_load_a_task(tmp_path):
    key = make_key()
    frame = pd.DataFrame({"date": ["2020-01-31"], "ret": [0.0123456789012]})
    write_task(tmp_path / "t1", key, "Compute x.", "## Files", {"returns.csv": frame})
    task = load_task(tmp_path / "t1")
    assert task.key == key
    public = sorted(p.name for p in (tmp_path / "t1" / PUBLIC_DIR).iterdir())
    assert public == [DICTIONARY_FILE, REQUEST_FILE, "returns.csv"]
    # The key is next to the public folder, not inside it.
    assert (tmp_path / "t1" / KEY_FILE).is_file()
    assert "`x`: a number" in task.request
    assert task.request.startswith("Compute x.")
    assert pd.read_csv(task.public_dir / "returns.csv")["ret"].iloc[0] == pytest.approx(
        0.0123456789012, abs=1e-11
    )


def test_load_tasks_sorts_by_id(tmp_path):
    for task_id in ["b", "a"]:
        write_task(tmp_path / task_id, make_key(task_id=task_id), "r", "d", {})
    (tmp_path / "not_a_task").mkdir()
    assert [t.task_id for t in load_tasks(tmp_path)] == ["a", "b"]


def test_task_id_must_match_directory(tmp_path):
    write_task(tmp_path / "other", make_key(), "r", "d", {})
    with pytest.raises(TaskError, match="does not match"):
        load_task(tmp_path / "other")


@pytest.mark.parametrize("name", [REQUEST_FILE, "answer.json", "../x.csv", ".hidden"])
def test_public_file_names_are_checked(tmp_path, name):
    with pytest.raises(TaskError, match="invalid public file name"):
        write_task(tmp_path / "t1", make_key(), "r", "d", {name: "x"})


def test_refuses_to_overwrite_an_existing_task(tmp_path):
    write_task(tmp_path / "t1", make_key(), "r", "d", {})
    with pytest.raises(FileExistsError):
        write_task(tmp_path / "t1", make_key(), "r", "d", {})
