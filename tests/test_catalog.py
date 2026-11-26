from types import SimpleNamespace

import pytest

from qtb import catalog
from qtb.catalog import CATALOG, TaskSpec, build, generate, spec_for
from qtb.task import Kind, TaskError


def test_task_ids_are_unique():
    ids = [spec.task_id for spec in CATALOG]
    assert len(ids) == len(set(ids))


def test_spec_for_finds_a_task():
    assert spec_for("costs_2").params["phrasing"] == "investor"
    with pytest.raises(KeyError):
        spec_for("nope")


def test_build_moves_to_the_next_seed_when_the_trap_does_not_bite(monkeypatch):
    seen = []
    real = CATALOG[0]

    def fake_build(task_id, seed, control, params):
        seen.append(seed)
        if len(seen) < 3:
            raise TaskError("only 1.2 tolerances apart")
        return real.module.build(task_id, seed, control, params)

    monkeypatch.setitem(catalog.TRAPS, "fake", SimpleNamespace(build=fake_build))
    built = build(TaskSpec("fake_1", "fake", Kind.TRAP, 7, real.params))
    assert seen == [7, 1007, 2007]
    assert built.key.seed == 2007
    assert built.key.extra["draws"] == 3.0


def test_build_gives_up_after_max_draws(monkeypatch):
    def never(task_id, seed, control, params):
        raise TaskError("never")

    monkeypatch.setitem(catalog.TRAPS, "fake", SimpleNamespace(build=never))
    with pytest.raises(TaskError, match="no usable draw"):
        build(TaskSpec("fake_1", "fake", Kind.TRAP, 7))


def test_generate_refuses_to_overwrite_without_force(tmp_path):
    specs = (CATALOG[0],)
    generate(tmp_path, specs)
    with pytest.raises(FileExistsError):
        generate(tmp_path, specs)
    assert len(generate(tmp_path, specs, force=True)) == 1
