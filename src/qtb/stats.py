"""Statistics shared by the reference answers.

These are the definitions the requests spell out, so that the only thing separating a
corrected answer from a naive one is the trap: sample standard deviation (n - 1),
annualization by the stated number of periods, Spearman rank correlation computed date
by date on the assets that have both values.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def annualized_mean(returns, periods_per_year: int) -> float:
    r = _finite(returns)
    return float(r.mean() * periods_per_year)


def annualized_vol(returns, periods_per_year: int) -> float:
    r = _finite(returns)
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe(returns, periods_per_year: int) -> float:
    """Mean over sample standard deviation, scaled by the square root of periods per year.

    No risk-free rate is subtracted; requests say so.
    """
    r = _finite(returns)
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))


STATISTICS = {"mean": annualized_mean, "vol": annualized_vol, "sharpe": sharpe}


def statistic(name: str, returns, periods_per_year: int) -> float:
    return STATISTICS[name](returns, periods_per_year)


def rank_ic_series(signal: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman correlation per row, on assets with both values.

    ``signal`` and ``target`` are dates-by-assets frames on the same grid. Rows with fewer
    than three usable assets are dropped.
    """
    if not signal.index.equals(target.index) or not signal.columns.equals(target.columns):
        raise ValueError("signal and target must share the same grid")
    out = {}
    for date in signal.index:
        s, t = signal.loc[date], target.loc[date]
        ok = s.notna() & t.notna()
        if ok.sum() < 3:
            continue
        out[date] = float(np.corrcoef(s[ok].rank(), t[ok].rank())[0, 1])
    return pd.Series(out, dtype=float)


def _finite(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError("need at least two finite returns")
    return r
