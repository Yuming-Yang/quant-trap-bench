"""Benchmark validity: every task behaves as the SPEC requires."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from qtb.agents import NAIVE, ORACLE
from qtb.catalog import CATALOG, TRAPS, build
from qtb.runner import run_task
from qtb.scorer import Outcome, score_text
from qtb.stats import sharpe
from qtb.task import KEY_FILE, Kind
from qtb.traps.base import read_wide

TASK_IDS = [spec.task_id for spec in CATALOG]


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_naive_agent_falls_for_every_trap_and_passes_every_control(tasks_by_id, task_id):
    task = tasks_by_id[task_id]
    outcome = run_task(NAIVE, task).score.outcome
    expected = Outcome.NAIVE if task.key.kind is Kind.TRAP else Outcome.CORRECTED
    assert outcome is expected


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_oracle_reproduces_the_key_from_the_public_files(tasks_by_id, task_id):
    task = tasks_by_id[task_id]
    result = run_task(ORACLE, task)
    assert result.score.outcome is Outcome.CORRECTED
    # The key comes from the generator's arrays, the oracle from the CSV files. They
    # should agree to rounding, so the tolerance is not what makes the oracle pass.
    spec = task.key.field(task.key.primary)
    assert result.score.values[spec.name] == pytest.approx(spec.corrected, abs=spec.tolerance / 100)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_trap_moves_the_answer_by_more_than_two_tolerances(tasks_by_id, task_id):
    key = tasks_by_id[task_id].key
    spec = key.field(key.primary)
    if key.kind is Kind.TRAP:
        assert spec.separation > 2.0
    else:
        assert spec.naive is None


@pytest.mark.parametrize("spec", CATALOG[:6], ids=lambda s: s.task_id)
def test_generation_is_deterministic(spec):
    first, second = build(spec), build(spec)
    assert first.key == second.key
    for name, content in first.files.items():
        if isinstance(content, pd.DataFrame):
            pd.testing.assert_frame_equal(content, second.files[name])


def test_answer_key_is_not_in_the_public_folder(generated):
    for task in generated:
        assert not (task.public_dir / KEY_FILE).exists()
        assert (task.root / KEY_FILE).is_file()


# Words that would give the trap away if they appeared in what the agent reads.
GIVEAWAYS = ["trap", "survivorship", "lookahead", "look-ahead", "point-in-time", "bias", "careful"]


def test_public_text_does_not_name_the_trap(generated):
    for task in generated:
        for path in task.public_dir.glob("*.md"):
            text = path.read_text().lower()
            for word in GIVEAWAYS:
                assert word not in text, f"{task.task_id}/{path.name} mentions {word!r}"


def test_every_trap_has_two_tasks(generated):
    counts = {}
    for task in generated:
        if task.key.kind is Kind.TRAP:
            counts[task.key.trap] = counts.get(task.key.trap, 0) + 1
    assert counts == dict.fromkeys(TRAPS, 2)


def _overcorrected(task, tmp_path, edit):
    """Run the plain analysis on a control after applying a needless 'fix' to its files."""
    workdir = tmp_path / "work"
    shutil.copytree(task.public_dir, workdir)
    edit(workdir)
    answer = TRAPS[task.key.mirrors].naive(workdir, task.key.params)
    return score_text(json.dumps(answer), task.key).outcome


def test_control_catches_lagging_a_signal_that_was_already_tradable(tasks_by_id, tmp_path):
    def lag_signal(workdir):
        signals = pd.read_csv(workdir / "signals.csv")
        signals["mom5"] = signals.groupby("ticker")["mom5"].shift(1)
        signals.to_csv(workdir / "signals.csv", index=False)

    assert _overcorrected(tasks_by_id["lookahead_control"], tmp_path, lag_signal) is Outcome.OTHER


def test_control_catches_applying_splits_twice(tasks_by_id, tmp_path):
    def reapply_splits(workdir):
        prices = pd.read_csv(workdir / "prices.csv")
        splits = pd.read_csv(workdir / "splits.csv")
        for row in splits.itertuples():
            later = (prices["ticker"] == row.ticker) & (prices["date"] >= row.date)
            prices.loc[later, "close"] *= row.ratio
        prices.to_csv(workdir / "prices.csv", index=False)

    assert _overcorrected(tasks_by_id["splits_control"], tmp_path, reapply_splits) is Outcome.OTHER


def test_control_catches_an_extra_reporting_lag(tasks_by_id, tmp_path):
    def add_a_quarter(workdir):
        reports = pd.read_csv(workdir / "fundamentals.csv", parse_dates=["date"])
        reports["date"] = reports["date"] + pd.DateOffset(months=3)
        reports.to_csv(workdir / "fundamentals.csv", index=False)

    task = tasks_by_id["point_in_time_control"]
    assert _overcorrected(task, tmp_path, add_a_quarter) is Outcome.OTHER


def test_long_files_have_no_rows_for_unlisted_months(generated):
    # The one documented exception: a forward return is empty when there is no next month.
    for task in generated:
        for path in task.public_dir.glob("*.csv"):
            frame = pd.read_csv(path).drop(columns=["fwd_ret"], errors="ignore")
            assert not frame.isna().any().any(), f"{task.task_id}/{path.name}"


def test_control_catches_reporting_only_the_last_five_years(tasks_by_id):
    task = tasks_by_id["selection_control"]
    selection = TRAPS["selection"]
    returns = read_wide(task.public_dir / "returns.csv", "ret")
    series = selection.momentum_returns(returns, task.key.params["lookback"], 5)
    late = sharpe(series[series.index >= selection.HOLDOUT_START], 12)
    assert late == pytest.approx(task.key.extra["false_alarm"], abs=1e-9)
    assert score_text(json.dumps({"sharpe": late}), task.key).outcome is Outcome.OTHER


def test_control_catches_shifting_a_forward_return_again(tasks_by_id, tmp_path):
    def shift_again(workdir):
        panel = pd.read_csv(workdir / "panel.csv")
        panel["fwd_ret"] = panel.groupby("ticker")["fwd_ret"].shift(-1)
        panel.to_csv(workdir / "panel.csv", index=False)

    task = tasks_by_id["ic_alignment_control"]
    assert _overcorrected(task, tmp_path, shift_again) is Outcome.OTHER


def test_catalog_size_and_control_share(generated):
    n_control = sum(task.key.kind is Kind.CONTROL for task in generated)
    assert 20 <= len(generated) <= 30
    assert 0.2 <= n_control / len(generated) <= 1 / 3


def test_both_families_are_covered(generated):
    families = {(task.key.family, task.key.kind) for task in generated}
    assert families == {
        (family, kind) for family in ("data", "method") for kind in (Kind.TRAP, Kind.CONTROL)
    }


def test_multiple_testing_key_matches_a_separate_implementation(tasks_by_id):
    # The key uses factor-research-pipeline's deflated Sharpe ratio; the oracle its own.
    task = tasks_by_id["multiple_testing_1"]
    answer = TRAPS["multiple_testing"].oracle(task.public_dir, task.key.params)
    assert answer["prob_skill"] == pytest.approx(task.key.field("prob_skill").corrected, abs=1e-6)
    assert task.key.field("prob_skill").naive > 0.9


def test_survivorship_trap_hides_only_delisted_stocks(tasks_by_id):
    public = tasks_by_id["survivorship_1"].public_dir
    main = pd.read_csv(public / "returns.csv")
    archive = pd.read_csv(public / "delisted_returns.csv")
    assert set(main["ticker"]).isdisjoint(archive["ticker"])
    last_month = main["date"].max()
    # Every stock in the main file is still listed at the end; none in the archive is.
    assert set(main.loc[main["date"] == last_month, "ticker"]) == set(main["ticker"])
    assert archive["date"].max() < last_month


def test_split_days_look_like_crashes_in_raw_prices(tasks_by_id):
    prices = pd.read_csv(tasks_by_id["splits_1"].public_dir / "prices.csv")
    prices["raw"] = prices.groupby("ticker")["close"].pct_change()
    prices["adjusted"] = (
        (prices["close"] * prices["adj_factor"]).groupby(prices["ticker"]).pct_change()
    )
    split_days = (
        prices["adj_factor"].ne(prices.groupby("ticker")["adj_factor"].shift(1))
        & prices["raw"].notna()
    )
    assert split_days.sum() == tasks_by_id["splits_1"].key.extra["n_splits"]
    assert (prices.loc[split_days, "raw"] < -0.45).all()
    assert prices["adjusted"].abs().max() < 0.2


def test_lookahead_signal_contains_the_next_day_return(tasks_by_id):
    public = tasks_by_id["lookahead_1"].public_dir
    close = read_wide(public / "prices.csv", "close")
    signal = read_wide(public / "signals.csv", "mom5").reindex(close.index)
    expected = close.shift(-1) / close.shift(4) - 1
    mask = (signal.notna() & expected.notna()).to_numpy()
    assert mask.sum() > 20_000
    np.testing.assert_allclose(
        signal.to_numpy()[mask], expected.to_numpy()[mask], rtol=0, atol=1e-8
    )


def test_point_in_time_reports_are_published_after_the_quarter(tasks_by_id):
    reports = pd.read_csv(
        tasks_by_id["point_in_time_1"].public_dir / "fundamentals.csv",
        parse_dates=["period_end", "publish_date"],
    )
    lag_days = (reports["publish_date"] - reports["period_end"]).dt.days
    assert lag_days.min() > 0
    assert lag_days.max() < 120
