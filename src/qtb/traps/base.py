"""Pieces shared by the trap generators."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtb.task import AnswerKey

Answer = dict[str, float]
Solver = Callable[[Path, Mapping[str, Any]], Answer]


@dataclass(frozen=True)
class Built:
    """A generated task before it is written to disk."""

    key: AnswerKey
    request: str
    dictionary: str
    files: Mapping[str, pd.DataFrame | str]


STATISTIC_WORDS = {
    "mean": (
        "annualized mean return",
        "the mean of the {freq} portfolio returns times {periods}, as a decimal "
        "(0.05 means 5% a year)",
    ),
    "vol": (
        "annualized volatility",
        "the sample standard deviation (n - 1 denominator) of the {freq} portfolio "
        "returns times the square root of {periods}, as a decimal",
    ),
    "sharpe": (
        "annualized Sharpe ratio",
        "the mean of the {freq} portfolio returns divided by their sample standard "
        "deviation (n - 1 denominator), times the square root of {periods}; "
        "do not subtract a risk-free rate",
    ),
}


def statistic_text(statistic: str, freq: str, periods: int) -> tuple[str, str]:
    """Short name and exact definition of a statistic, for a request."""
    name, definition = STATISTIC_WORDS[statistic]
    return name, definition.format(freq=freq, periods=periods)


def month_ends(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def business_days(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def tickers(n: int, rng: np.random.Generator) -> list[str]:
    """Distinct random four-letter tickers, sorted. Random so that names carry no hint."""
    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    names: set[str] = set()
    while len(names) < n:
        names.add("".join(rng.choice(letters, size=4)))
    return sorted(names)


def long_frame(wide: pd.DataFrame, value: str, date: str = "date") -> pd.DataFrame:
    """Dates-by-tickers frame to a long table with one row per listed (date, ticker)."""
    # Missing cells mean "not listed" and get no row. (pandas 3 keeps NaN in stack.)
    out = wide.stack().dropna().rename(value).reset_index()
    out.columns = [date, "ticker", value]
    out[date] = out[date].dt.strftime("%Y-%m-%d")
    return out.sort_values([date, "ticker"], ignore_index=True)


def read_wide(path: Path, value: str, date: str = "date") -> pd.DataFrame:
    """Long CSV back to a dates-by-tickers frame. Used by the solvers."""
    frame = pd.read_csv(path, parse_dates=[date])
    return frame.pivot(index=date, columns="ticker", values=value).sort_index()
