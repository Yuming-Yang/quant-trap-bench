import pytest

from qtb.cli import main


def test_version_flag_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "qtb" in capsys.readouterr().out


def test_no_arguments_is_a_no_op():
    assert main([]) == 0
