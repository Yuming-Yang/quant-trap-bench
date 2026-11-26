"""Fundamentals dated by fiscal period end instead of publication date.

Market. Monthly returns for a few hundred stocks from 2014 to 2024. Each stock reports a
quarterly earnings yield, published one to three months after the quarter ends (on a
day inside that month). Returns have two parts that involve the reports:

- a small genuine premium: the last *published* earnings yield predicts next month's
  return with a modest rank IC;
- an announcement effect: in the month a report comes out, the stock moves with the
  news in it (the part of the new value not predictable from the previous one).

Trap. ``fundamentals.csv`` has both ``period_end`` and ``publish_date``. Lining values up
by ``period_end`` uses each report one to three months before anyone could see it, and
the month before publication it picks up the announcement return. The naive rank IC is
therefore far higher than the IC an investor could have earned.

Control. The file has one date column, the publication date, and says so. Lining up by
it is correct; lagging it further would be a false alarm.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtb.stats import rank_ic_series
from qtb.task import AnswerField, AnswerKey, Family, Kind
from qtb.traps.base import Answer, Built, long_frame, month_ends, read_wide, tickers

TRAP = "point_in_time"
FAMILY = Family.DATA
TOLERANCE = 0.005
FIELD = "mean_rank_ic"
FIRST_SIGNAL, LAST_SIGNAL = "2015-01-31", "2024-11-30"


def build(task_id: str, seed: int, control: bool, params: Mapping[str, Any]) -> Built:
    rng = np.random.default_rng(seed)
    n_assets, n_months = params.get("n_assets", 300), 132
    lag_lo, lag_hi = params.get("lag_months", (1, 3))
    ic, announce = params.get("ic", 0.03), params.get("announcement", 0.6)
    dates = month_ends("2014-01-31", n_months)
    names = tickers(n_assets, rng)

    # Quarterly earnings yield per stock: stationary AR(1) with unit variance.
    quarter_months = np.flatnonzero(dates.month % 3 == 0)
    n_q = len(quarter_months)
    phi = 0.8
    value = np.empty((n_q, n_assets))
    news = rng.standard_normal((n_q, n_assets)) * np.sqrt(1 - phi**2)
    value[0] = rng.standard_normal(n_assets)
    news[0] = 0.0  # the first report has no "previous" to surprise against
    for q in range(1, n_q):
        value[q] = phi * value[q - 1] + news[q]
    lag = rng.integers(lag_lo, lag_hi + 1, size=(n_q, n_assets))
    publish_month = quarter_months[:, None] + lag  # index of the month it comes out
    publish_day = rng.integers(5, 26, size=(n_q, n_assets))

    # Signals on the monthly grid: latest value by period end, and by publication.
    by_period = np.full((n_months, n_assets), np.nan)
    by_publish = np.full((n_months, n_assets), np.nan)
    surprise = np.zeros((n_months, n_assets))
    for q, m in enumerate(quarter_months):
        by_period[m:] = value[q]
        for i in range(n_assets):
            p = publish_month[q, i]
            if p < n_months:
                by_publish[p:, i] = value[q, i]
                surprise[p, i] += news[q, i] / np.sqrt(1 - phi**2)

    # Returns of month t+1 load on the published value at the end of month t, and on
    # the news released during month t+1.
    known = np.nan_to_num(by_publish)
    noise_scale = np.sqrt(1 - ic**2 - announce**2 / 3)
    standardized = np.zeros((n_months, n_assets))
    noise = rng.standard_normal((n_months - 1, n_assets))
    standardized[1:] = ic * known[:-1] + announce * surprise[1:] + noise_scale * noise
    market = 0.007 + 0.04 * rng.standard_normal(n_months)
    returns = np.maximum(market[:, None] + 0.09 * standardized, -0.95)
    ret = pd.DataFrame(returns, index=dates, columns=names)

    def mean_ic(signal: np.ndarray) -> float:
        sig = pd.DataFrame(signal, index=dates, columns=names)
        forward = ret.shift(-1)
        window = (dates >= FIRST_SIGNAL) & (dates <= LAST_SIGNAL)
        return float(rank_ic_series(sig[window], forward[window]).mean())

    corrected, naive = mean_ic(by_publish), mean_ic(by_period)

    spec = AnswerField(
        FIELD,
        "the mean monthly Spearman rank IC, as a decimal (0.05, not 5)",
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
        extra={"naive": naive, "planted_ic": ic},
    )

    rows = []
    for q, m in enumerate(quarter_months):
        for i, name in enumerate(names):
            p = publish_month[q, i]
            if p >= n_months:
                continue  # not yet published when the data was pulled
            published = dates[p].replace(day=int(publish_day[q, i]))
            rows.append((name, dates[m], published, value[q, i]))
    reports = pd.DataFrame(rows, columns=["ticker", "period_end", "publish_date", "earnings_yield"])
    reports = reports.sort_values(["ticker", "period_end"], ignore_index=True)
    for column in ["period_end", "publish_date"]:
        reports[column] = reports[column].dt.strftime("%Y-%m-%d")
    returns_long = long_frame(ret, "ret")

    returns_text = (
        "## returns.csv\n\n"
        "- `date`: month end, `YYYY-MM-DD`\n"
        "- `ticker`: stock identifier\n"
        "- `ret`: total return over the month ending on `date`, as a decimal\n"
    )
    if control:
        reports = reports[["ticker", "publish_date", "earnings_yield"]].rename(
            columns={"publish_date": "date"}
        )
        fundamentals_text = (
            "## fundamentals.csv\n\n"
            "Quarterly earnings yield, each row stored under the date on which the "
            "report was filed and became public.\n\n"
            "- `ticker`: stock identifier\n"
            "- `date`: filing date, `YYYY-MM-DD`; the value was public from this date on\n"
            "- `earnings_yield`: trailing earnings over market value, standardized\n"
        )
    else:
        fundamentals_text = (
            "## fundamentals.csv\n\n"
            "Quarterly earnings yield, one row per stock and fiscal quarter.\n\n"
            "- `ticker`: stock identifier\n"
            "- `period_end`: last day of the fiscal quarter the value refers to\n"
            "- `publish_date`: date the quarterly report was filed and became public\n"
            "- `earnings_yield`: trailing earnings over market value, standardized\n"
        )
    request = (
        "# Request\n\n"
        "Test whether earnings yield predicts stock returns in this market. At each month "
        "end from January 2015 to November 2024, take each stock's most recent earnings "
        "yield and compute the Spearman rank correlation, across stocks, between it and "
        "the stock's return over the following month. Report the mean of these monthly "
        "rank correlations (the mean rank IC).\n\n"
        "The data is described in `DATA_DICTIONARY.md`.\n"
    )
    dictionary = "# Data dictionary\n\n" + returns_text + "\n" + fundamentals_text
    return Built(
        key, request, dictionary, {"returns.csv": returns_long, "fundamentals.csv": reports}
    )


def _mean_ic(workdir: Path, date_column: str) -> Answer:
    ret = read_wide(workdir / "returns.csv", "ret")
    reports = pd.read_csv(workdir / "fundamentals.csv", parse_dates=[date_column])
    # A value dated on any day of a month is usable at that month's end.
    reports["month"] = reports[date_column] + pd.offsets.MonthEnd(0)
    reports = reports.sort_values(date_column)
    signal = (
        reports.groupby(["month", "ticker"])["earnings_yield"]
        .last()
        .unstack()
        .reindex(index=ret.index, columns=ret.columns)
        .ffill()
    )
    window = (ret.index >= FIRST_SIGNAL) & (ret.index <= LAST_SIGNAL)
    ics = rank_ic_series(signal[window], ret.shift(-1)[window])
    return {FIELD: float(ics.mean())}


def naive(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Use the date column that labels the value: the fiscal period end if there is one."""
    columns = pd.read_csv(workdir / "fundamentals.csv", nrows=0).columns
    return _mean_ic(workdir, "period_end" if "period_end" in columns else "date")


def oracle(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Use each value only from the date it became public."""
    columns = pd.read_csv(workdir / "fundamentals.csv", nrows=0).columns
    return _mean_ic(workdir, "publish_date" if "publish_date" in columns else "date")
