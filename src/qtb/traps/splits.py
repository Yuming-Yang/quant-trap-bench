"""Unadjusted corporate actions: a split shows up as a price crash.

Market. Daily closes for a dozen stocks over three years, from geometric random walks
with a common market factor. Several stocks split two-for-one or three-for-one.

Trap. ``prices.csv`` holds the raw close as traded, with a cumulative split factor next
to it. Daily returns computed from the raw close show a -50% or -67% day at every split.
The request asks for a statistic of the equal-weighted portfolio, which those fake
crashes drag down.

Control. ``prices.csv`` is already split-adjusted, and a separate ``splits.csv`` lists
the events for reference. Applying the splits a second time is a false alarm.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtb.stats import statistic
from qtb.task import AnswerField, AnswerKey, Family, Kind
from qtb.traps.base import (
    Answer,
    Built,
    business_days,
    long_frame,
    read_wide,
    statistic_text,
    tickers,
)

TRAP = "splits"
FAMILY = Family.DATA
TOLERANCE = {"mean": 0.01, "sharpe": 0.05, "vol": 0.005}


def build(task_id: str, seed: int, control: bool, params: Mapping[str, Any]) -> Built:
    stat = params["statistic"]
    rng = np.random.default_rng(seed)
    n_assets, n_days = params.get("n_assets", 12), 756
    dates = business_days("2021-01-04", n_days)
    names = tickers(n_assets, rng)

    market = 0.0004 + 0.01 * rng.standard_normal(n_days)
    log_ret = market[:, None] + 0.015 * rng.standard_normal((n_days, n_assets))
    log_ret[0] = 0.0
    start = rng.uniform(40, 250, size=n_assets)
    adjusted = start * np.exp(np.cumsum(log_ret, axis=0))  # in today's-share units at t=0

    factor = np.ones((n_days, n_assets))
    events = []
    splitters = rng.choice(n_assets, size=params.get("n_splitters", 6), replace=False)
    for i in splitters:
        n_events = rng.integers(1, 3)
        days = np.sort(rng.choice(np.arange(20, n_days - 20), size=n_events, replace=False))
        for day in days:
            ratio = int(rng.choice([2, 3], p=[0.7, 0.3]))
            factor[day:, i] *= ratio
            events.append((names[i], dates[day], ratio))
    raw = adjusted / factor

    def portfolio(prices: np.ndarray) -> pd.Series:
        daily = prices[1:] / prices[:-1] - 1.0
        return pd.Series(daily.mean(axis=1), index=dates[1:])

    corrected = statistic(stat, portfolio(adjusted), 252)
    naive = statistic(stat, portfolio(raw), 252)

    label, definition = statistic_text(stat, "daily", 252)
    field_name = f"ew_{stat}"
    spec = AnswerField(
        field_name,
        f"the {label} of the equal-weighted portfolio, {definition}",
        corrected,
        None if control else naive,
        TOLERANCE[stat],
    )
    key = AnswerKey(
        task_id=task_id,
        kind=Kind.CONTROL if control else Kind.TRAP,
        family=FAMILY,
        trap=None if control else TRAP,
        mirrors=TRAP,
        primary=field_name,
        fields=(spec,),
        seed=seed,
        params=dict(params),
        extra={"naive": naive, "n_splits": float(len(events))},
    )
    request = (
        "# Request\n\n"
        "Report the "
        f"{label} of an equal-weighted portfolio of these stocks, rebalanced every day: "
        "each day's portfolio return is the simple average of the stocks' daily returns "
        "from the previous close to that day's close. Use every trading day in the data. "
        "There are no dividends.\n\n"
        "The data is described in `DATA_DICTIONARY.md`.\n"
    )
    splits = pd.DataFrame(events, columns=["ticker", "date", "ratio"]).sort_values(
        ["date", "ticker"], ignore_index=True
    )
    splits["date"] = splits["date"].dt.strftime("%Y-%m-%d")
    if control:
        prices = long_frame(pd.DataFrame(adjusted, index=dates, columns=names), "close")
        dictionary = (
            "# Data dictionary\n\n## prices.csv\n\n"
            "- `date`: trading day, `YYYY-MM-DD`\n"
            "- `ticker`: stock identifier\n"
            "- `close`: closing price, adjusted for every split in `splits.csv` so that "
            "it is comparable across dates (in units of a share on the first day)\n\n"
            "## splits.csv\n\n"
            "Stock splits during the period, for reference. They are already reflected "
            "in `close`.\n\n"
            "- `ticker`: stock identifier\n"
            "- `date`: first trading day on the new share count\n"
            "- `ratio`: new shares per old share\n"
        )
        files = {"prices.csv": prices, "splits.csv": splits}
    else:
        prices = long_frame(pd.DataFrame(raw, index=dates, columns=names), "close")
        prices["adj_factor"] = long_frame(
            pd.DataFrame(factor, index=dates, columns=names), "adj_factor"
        )["adj_factor"]
        dictionary = (
            "# Data dictionary\n\n## prices.csv\n\n"
            "- `date`: trading day, `YYYY-MM-DD`\n"
            "- `ticker`: stock identifier\n"
            "- `close`: closing price as traded that day, per share outstanding that day\n"
            "- `adj_factor`: cumulative split factor, the number of shares that one share "
            "held on the first day of the file had become by `date`. `close * adj_factor` "
            "is comparable across dates.\n"
        )
        files = {"prices.csv": prices}
    return Built(key, request, dictionary, files)


def _answer(prices: pd.DataFrame, params: Mapping[str, Any]) -> Answer:
    stat = params["statistic"]
    daily = prices.pct_change().iloc[1:]
    return {f"ew_{stat}": statistic(stat, daily.mean(axis=1), 252)}


def naive(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Daily returns from the close column as it is."""
    return _answer(read_wide(workdir / "prices.csv", "close"), params)


def oracle(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Multiply by the split factor first, when the file has one."""
    frame = pd.read_csv(workdir / "prices.csv", parse_dates=["date"])
    if "adj_factor" in frame:
        frame["close"] = frame["close"] * frame["adj_factor"]
    prices = frame.pivot(index="date", columns="ticker", values="close").sort_index()
    return _answer(prices, params)
