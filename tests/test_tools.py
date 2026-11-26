import os

from qtb.tools import MAX_OUTPUT_CHARS, execute, run_python, truncate


def test_runs_in_the_working_directory(tmp_path):
    (tmp_path / "data.csv").write_text("x\n1\n")
    result = run_python("import os; print(sorted(os.listdir('.')))", tmp_path)
    assert result.output.strip() == "['data.csv']"
    assert not result.is_error
    assert result.exit_code == 0


def test_errors_are_returned_not_raised(tmp_path):
    result = run_python("raise ValueError('boom')", tmp_path)
    assert result.is_error
    assert result.exit_code == 1
    assert "ValueError: boom" in result.output


def test_files_persist_between_calls_but_variables_do_not(tmp_path):
    run_python("x = 1\nopen('note.txt', 'w').write('kept')", tmp_path)
    assert "NameError" in run_python("print(x)", tmp_path).output
    assert run_python("print(open('note.txt').read())", tmp_path).output.strip() == "kept"


def test_secrets_in_the_harness_environment_are_not_passed_on(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    result = run_python("import os; print('ANTHROPIC_API_KEY' in os.environ)", tmp_path)
    assert result.output.strip() == "False"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-not-real"


def test_timeout(tmp_path):
    result = run_python("import time; time.sleep(5)", tmp_path, timeout_s=0.5)
    assert result.timed_out
    assert result.is_error
    assert "stopped after 0.5 seconds" in result.output


def test_long_output_is_truncated_in_the_middle():
    text = "a" * 10 + "b" * (MAX_OUTPUT_CHARS * 2) + "z" * 10
    out = truncate(text)
    assert len(out) < MAX_OUTPUT_CHARS + 100
    assert out.startswith("a" * 10)
    assert out.endswith("z" * 10)
    assert "characters truncated" in out


def test_empty_output_says_so(tmp_path):
    assert run_python("pass", tmp_path).output == "(no output)"


def test_unknown_tool_and_bad_arguments(tmp_path):
    assert execute("bash", {"code": "ls"}, tmp_path, 5).is_error
    assert "string argument" in execute("python", {"source": "1"}, tmp_path, 5).output
    assert execute("python", "print(1)", tmp_path, 5).is_error
