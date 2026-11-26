import json

import pytest

from qtb.scorer import Outcome, classify, score_text, score_workdir
from qtb.task import ANSWER_FILE


def answer(**values) -> str:
    return json.dumps(values)


def test_corrected_value_earns_credit(trap_key):
    score = score_text(answer(sharpe=0.42, n_assets=120), trap_key)
    assert score.outcome is Outcome.CORRECTED
    assert score.credit == 1
    assert score.values == {"sharpe": 0.42, "n_assets": 120.0}


def test_naive_value_is_recorded_as_naive_with_no_credit(trap_key):
    score = score_text(answer(sharpe=0.88, n_assets=100), trap_key)
    assert score.outcome is Outcome.NAIVE
    assert score.credit == 0
    assert score.fields == {"sharpe": Outcome.NAIVE, "n_assets": Outcome.NAIVE}


def test_unrelated_wrong_answer_is_other_not_naive(trap_key):
    score = score_text(answer(sharpe=0.65, n_assets=120), trap_key)
    assert score.outcome is Outcome.OTHER
    assert score.credit == 0


def test_outcome_follows_the_primary_field_only(trap_key):
    # A secondary field that is off does not take away the credit.
    score = score_text(answer(sharpe=0.40, n_assets=7), trap_key)
    assert score.outcome is Outcome.CORRECTED
    assert score.fields["n_assets"] is Outcome.OTHER


def test_tolerance_is_inclusive_on_both_sides(trap_key):
    spec = trap_key.field("sharpe")
    assert classify(spec.corrected + spec.tolerance, spec) is Outcome.CORRECTED
    assert classify(spec.corrected - spec.tolerance, spec) is Outcome.CORRECTED
    assert classify(spec.corrected + spec.tolerance * 1.01, spec) is Outcome.OTHER
    assert classify(spec.naive - spec.tolerance, spec) is Outcome.NAIVE


def test_missing_answer_file(trap_key, tmp_path):
    score = score_workdir(tmp_path, trap_key)
    assert score.outcome is Outcome.MISSING
    assert score.credit == 0


def test_answer_file_is_read_from_the_working_directory(trap_key, tmp_path):
    (tmp_path / ANSWER_FILE).write_text(answer(sharpe=0.4, n_assets=120))
    assert score_workdir(tmp_path, trap_key).outcome is Outcome.CORRECTED


def test_binary_answer_file_is_malformed(trap_key, tmp_path):
    (tmp_path / ANSWER_FILE).write_bytes(b"\xff\xfe\x00")
    assert score_workdir(tmp_path, trap_key).outcome is Outcome.MALFORMED


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("{sharpe: 0.4}", "not valid JSON"),
        ("", "not valid JSON"),
        ("[0.4, 120]", "JSON object"),
        ("0.4", "JSON object"),
        (answer(sharpe=0.4), "missing field 'n_assets'"),
        (answer(sharpe="0.4", n_assets=120), "not a number"),
        (answer(sharpe="40%", n_assets=120), "not a number"),
        (answer(sharpe=None, n_assets=120), "not a number"),
        (answer(sharpe=True, n_assets=120), "not a number"),
        ('{"sharpe": NaN, "n_assets": 120}', "not finite"),
        ('{"sharpe": Infinity, "n_assets": 120}', "not finite"),
    ],
)
def test_malformed_answers(trap_key, text, reason):
    score = score_text(text, trap_key)
    assert score.outcome is Outcome.MALFORMED
    assert reason in score.detail
    assert score.credit == 0
    assert score.fields == {}


def test_extra_keys_are_ignored(trap_key):
    text = json.dumps({"sharpe": 0.4, "n_assets": 120, "notes": "used the archive file"})
    assert score_text(text, trap_key).outcome is Outcome.CORRECTED


def test_integer_answers_are_numbers(control_key):
    text = json.dumps({"sharpe": 0})
    assert score_text(text, control_key).outcome is Outcome.OTHER


def test_control_task_has_no_naive_outcome(control_key):
    assert score_text(answer(sharpe=0.41), control_key).outcome is Outcome.CORRECTED
    # A value far off is a false alarm or a plain error, never "naive".
    assert score_text(answer(sharpe=0.9), control_key).outcome is Outcome.OTHER


def test_score_serializes(trap_key):
    data = score_text(answer(sharpe=0.88, n_assets=100), trap_key).to_dict()
    assert data["outcome"] == "naive"
    assert data["credit"] == 0
    assert json.loads(json.dumps(data)) == data
