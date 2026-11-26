"""Price lookahead: a signal column that includes the next day's close.

Market. Daily closes for 50 stocks over two years. Returns carry a small genuine
five-day momentum effect: the cross-sectional rank of the past five-day return predicts
the next day's return.

Trap. ``signals.csv`` holds a five-day momentum value whose window ends at the close of
the trading day *after* the row's date, and the data dictionary says so. Trading on it
at the row's close uses the next day's price, so the backtest earns the very return the
signal contains. The corrected backtest uses the value from the previous row, whose
window ends at the current close (equivalently, momentum recomputed from prices).

Control. The signal's window ends at the close of the row's date, so it can be traded
at that close. Lagging it again is a false alarm.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtb.stats import sharpe
from qtb.task import AnswerField, AnswerKey, Family, Kind
from qtb.traps.base import Answer, Built, business_days, long_frame, read_wide, tickers

TRAP = "lookahead"
FAMILY = Family.DATA
TOLERANCE = 0.1
FIELD = "long_short_sharpe"
WINDOW = 5


def long_short(signal: pd.DataFrame, forward: pd.DataFrame, k: int) -> pd.Series:
    """Top-``k`` minus bottom-``k`` equal-weighted return, per row with a full signal."""
    out = {}
    for date in signal.index:
        s, r = signal.loc[date], forward.loc[date]
        ok = s.notna() & r.notna()
        if ok.sum() < 2 * k:
            continue
        ranked = s[ok].sort_values(kind="mergesort")
        out[date] = r[ranked.index[-k:]].mean() - r[ranked.index[:k]].mean()
    return pd.Series(out, dtype=float)


def build(task_id: str, seed: int, control: bool, params: Mapping[str, Any]) -> Built:
    rng = np.random.default_rng(seed)
    n_assets, n_days, k = params.get("n_assets", 50), 510, params.get("k", 10)
    strength = params.get("momentum_ic", 0.015)
    dates = business_days("2023-01-02", n_days)
    names = tickers(n_assets, rng)

    log_price = np.empty((n_days, n_assets))
    log_price[: WINDOW + 1] = np.log(rng.uniform(20, 200, size=n_assets)) + np.cumsum(
        0.015 * rng.standard_normal((WINDOW + 1, n_assets)), axis=0
    )
    for t in range(WINDOW, n_days - 1):
        momentum = log_price[t] - log_price[t - WINDOW]
        z = (momentum - momentum.mean()) / momentum.std()
        shock = strength * z + np.sqrt(1 - strength**2) * rng.standard_normal(n_assets)
        log_price[t + 1] = log_price[t] + 0.0003 + 0.008 * rng.standard_normal() + 0.015 * shock
    close = pd.DataFrame(np.exp(log_price), index=dates, columns=names)

    past = close / close.shift(WINDOW) - 1.0  # window ends at this row's close
    forward = close.shift(-1) / close - 1.0  # return from this close to the next
    leaked = past.shift(-1)  # window ends at the next row's close
    signal_file = past if control else leaked

    corrected = sharpe(long_short(past, forward, k), 252)
    naive = sharpe(long_short(leaked, forward, k), 252)
    spec = AnswerField(
        FIELD,
        "the annualized Sharpe ratio of the daily long-short returns: their mean divided "
        "by their sample standard deviation (n - 1 denominator), times the square root of "
        "252; no risk-free rate",
        corrected,
        None if control else naive,
        TOLERANCE,
    )
    key = AnswerKey(
        task_id=task_id,
        kind=Kind.CONTROL if control else Kind.TRAP,
        family=FAMILY,
        trap=None if control else TRAP,
        mirrors=TRAP,
        primary=FIELD,
        fields=(spec,),
        seed=seed,
        params=dict(params),
        extra={"naive": naive},
    )
    request = (
        "# Request\n\n"
        "Backtest a daily momentum strategy on the `mom5` signal. At each close, buy the "
        f"{k} stocks with the highest `mom5` and sell short the {k} with the lowest, "
        "equal-weighted within each side, and hold until the next close. The strategy's "
        "daily return is the average return of the long stocks minus the average return "
        "of the short stocks over that holding day. Ignore trading costs. Run it over "
        "every day the data allows and report its annualized Sharpe ratio.\n\n"
        "The data is described in `DATA_DICTIONARY.md`.\n"
    )
    if control:
        window_text = (
            "five-day price return from the close five trading days before `date` to the "
            "close of `date`. Available after the close of `date`."
        )
    else:
        window_text = (
            "five-day price return from the close four trading days before `date` to the "
            "close of the trading day after `date`. The vendor labels each value with the "
            "day on which its final one-day return begins, not with the day it was "
            "computed."
        )
    dictionary = (
        "# Data dictionary\n\n## prices.csv\n\n"
        "- `date`: trading day, `YYYY-MM-DD`\n"
        "- `ticker`: stock identifier\n"
        "- `close`: split-adjusted closing price\n\n"
        "## signals.csv\n\n"
        "- `date`: trading day, `YYYY-MM-DD`\n"
        "- `ticker`: stock identifier\n"
        f"- `mom5`: {window_text}\n"
    )
    files = {
        "prices.csv": long_frame(close, "close"),
        "signals.csv": long_frame(signal_file, "mom5"),
    }
    return Built(key, request, dictionary, files)


def _answer(workdir: Path, params: Mapping[str, Any], lag: int) -> Answer:
    close = read_wide(workdir / "prices.csv", "close")
    signal = read_wide(workdir / "signals.csv", "mom5").reindex(close.index)
    forward = close.shift(-1) / close - 1.0
    returns = long_short(signal.shift(lag), forward, params.get("k", 10))
    return {FIELD: sharpe(returns, 252)}


def naive(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Trade each row's signal at that row's close."""
    return _answer(workdir, params, lag=0)


def oracle(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Trade at each close on the last value whose window had ended by then."""
    return _answer(workdir, params, lag=1)
