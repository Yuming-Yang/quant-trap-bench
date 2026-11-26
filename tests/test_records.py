import json

import pytest

from qtb.agents import NAIVE, ORACLE
from qtb.cli import main
from qtb.records import run_and_record, run_path, summarize


def test_records_are_saved_and_never_overwritten(tasks_by_id, tmp_path):
    task = tasks_by_id["costs_1"]
    run_and_record(NAIVE, task, tmp_path, repeat=1)
    record = json.loads(run_path(tmp_path, "naive", "costs_1", 1).read_text())
    assert record["score"]["outcome"] == "naive"
    assert record["trap"] == "costs"
    with pytest.raises(FileExistsError):
        run_and_record(NAIVE, task, tmp_path, repeat=1)


def test_summary_counts_outcomes(tasks_by_id, tmp_path):
    for task_id in ["costs_1", "splits_1", "splits_control"]:
        run_and_record(NAIVE, tasks_by_id[task_id], tmp_path, repeat=1)
        run_and_record(ORACLE, tasks_by_id[task_id], tmp_path, repeat=1)
    naive = summarize(tmp_path / "naive")
    assert naive.n_runs == 3
    assert naive.trap_outcomes == {"naive": 2}
    assert naive.control_outcomes == {"corrected": 1}
    assert naive.by_trap["costs"] == {"naive": 1}
    oracle = summarize(tmp_path / "oracle")
    assert oracle.trap_outcomes == {"corrected": 2}
    assert "naive 0%" in "\n".join(oracle.lines())


def test_cli_run_and_summarize(generated, tmp_path, capsys):
    tasks_dir = generated[0].root.parent
    args = ["run", "--agent", "oracle", "--tasks", str(tasks_dir), "--out", str(tmp_path)]
    assert main([*args, "--task", "lookahead_1", "--task", "lookahead_control"]) == 0
    assert main(["summarize", str(tmp_path / "oracle")]) == 0
    out = capsys.readouterr().out
    assert "trap runs (1): corrected 100%" in out
    assert "control runs (1): correct 100%" in out


def test_cli_refuses_a_paid_run_without_explicit_permission(generated, tmp_path, capsys):
    tasks_dir = generated[0].root.parent
    code = main(["run", "--agent", "claude", "--tasks", str(tasks_dir), "--out", str(tmp_path)])
    assert code == 2
    assert "--allow-paid" in capsys.readouterr().out
    assert not any(tmp_path.iterdir())


def test_cli_rejects_unknown_task_ids(generated, tmp_path, capsys):
    tasks_dir = generated[0].root.parent
    args = ["run", "--agent", "naive", "--tasks", str(tasks_dir), "--out", str(tmp_path)]
    assert main([*args, "--task", "nope"]) == 1
    assert "unknown task ids: nope" in capsys.readouterr().out


def test_summarize_empty_directory(tmp_path):
    assert main(["summarize", str(tmp_path)]) == 1
