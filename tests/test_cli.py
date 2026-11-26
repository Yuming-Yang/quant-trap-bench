import pytest

from qtb import catalog
from qtb.cli import main


def test_version_flag_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "qtb" in capsys.readouterr().out


def test_no_arguments_is_a_no_op():
    assert main([]) == 0


def test_generate_then_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(catalog, "CATALOG", catalog.CATALOG[:3])
    monkeypatch.setattr(
        "qtb.cli.generate", lambda out, force: catalog.generate(out, catalog.CATALOG, force)
    )
    assert main(["generate", "--out", str(tmp_path)]) == 0
    assert "wrote 3 tasks (1 controls)" in capsys.readouterr().out
    assert main(["check", "--tasks", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "survivorship_control" in out
    assert out.strip().endswith("ok")


def test_check_without_tasks_fails(tmp_path, capsys):
    assert main(["check", "--tasks", str(tmp_path)]) == 1
    assert "qtb generate" in capsys.readouterr().out
