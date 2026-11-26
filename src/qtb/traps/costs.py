"""Zero costs: a high-turnover strategy whose edge is smaller than its trading costs.

Market. Daily closes for 40 stocks over three years, with a planted one-day reversal:
the cross-sectional rank of today's return predicts tomorrow's return with the opposite
sign.

Trap. The request asks for performance net of costs (or, in the subtler phrasing, for
what an investor would have earned). A per-stock cost schedule sits in a side file, and
the data dictionary says how to charge it. The strategy trades most of its book every
day, so the gross Sharpe ratio is high and the net one is not.

There is no control for this trap: a request that asks for performance before costs
would state it, which is not a test of noticing anything.
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

TRAP = "costs"
FAMILY = Family.DATA
TOLERANCE = 0.1
FIELD = "net_sharpe"
PHRASING = {
    "net": "Report the annualized Sharpe ratio of the strategy's daily returns, net of "
    "trading costs.",
    "investor": "Report the annualized Sharpe ratio that an investor running this strategy "
    "would have earned.",
}


def weights(signal: pd.DataFrame, k: int) -> pd.DataFrame:
    """Target weights at each close: +1/k on the k lowest values, -1/k on the k highest."""
    out = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for date in signal.index:
        s = signal.loc[date].dropna()
        if len(s) < 2 * k:
            continue
        ranked = s.sort_values(kind="mergesort")
        out.loc[date, ranked.index[:k]] = 1.0 / k
        out.loc[date, ranked.index[-k:]] = -1.0 / k
    return out


def strategy_returns(close: pd.DataFrame, cost_bps: pd.Series | None, k: int) -> pd.Series:
    """Daily returns of the reversal strategy, from each close to the next.

    Holding periods start at every close that has a one-day return behind it and a
    next close ahead. Costs, if given, are charged on the change in target weights at
    the close that starts the period.
    """
    past = close / close.shift(1) - 1.0
    forward = close.shift(-1) / close - 1.0
    active = past.notna().all(axis=1) & forward.notna().all(axis=1)
    w = weights(past[active], k)
    gross = (w * forward[active]).sum(axis=1)
    if cost_bps is None:
        return gross
    traded = w.diff().abs()
    traded.iloc[0] = w.iloc[0].abs()  # the first close builds the book from cash
    return gross - (traded * (cost_bps[w.columns] / 1e4)).sum(axis=1)


def build(task_id: str, seed: int, control: bool, params: Mapping[str, Any]) -> Built:
    if control:
        raise ValueError("the costs trap has no control task")
    rng = np.random.default_rng(seed)
    n_assets, n_days, k = params.get("n_assets", 40), 757, params.get("k", 8)
    strength = params.get("reversal_ic", 0.05)
    dates = business_days("2022-01-03", n_days)
    names = tickers(n_assets, rng)

    log_price = np.empty((n_days, n_assets))
    log_price[0] = np.log(rng.uniform(20, 200, size=n_assets))
    last = 0.015 * rng.standard_normal(n_assets)
    log_price[1] = log_price[0] + last
    for t in range(1, n_days - 1):
        z = (last - last.mean()) / last.std()
        shock = -strength * z + np.sqrt(1 - strength**2) * rng.standard_normal(n_assets)
        last = 0.0003 + 0.008 * rng.standard_normal() + 0.015 * shock
        log_price[t + 1] = log_price[t] + last
    close = pd.DataFrame(np.exp(log_price), index=dates, columns=names)
    cost_bps = pd.Series(np.round(rng.uniform(3, 12, size=n_assets), 1), index=names)

    corrected = sharpe(strategy_returns(close, cost_bps, k), 252)
    naive = sharpe(strategy_returns(close, None, k), 252)
    spec = AnswerField(
        FIELD,
        "the annualized Sharpe ratio: the mean of the daily strategy returns divided by "
        "their sample standard deviation (n - 1 denominator), times the square root of "
        "252; no risk-free rate",
        corrected,
        naive,
        TOLERANCE,
    )
    key = AnswerKey(
        task_id=task_id,
        kind=Kind.TRAP,
        family=FAMILY,
        trap=TRAP,
        mirrors=TRAP,
        primary=FIELD,
        fields=(spec,),
        seed=seed,
        params=dict(params),
        extra={"naive": naive},
    )
    request = (
        "# Request\n\n"
        "Evaluate a one-day reversal strategy. At each close, compute every stock's return "
        f"over that day, buy the {k} stocks with the lowest return and sell short the {k} "
        f"with the highest, each with weight 1/{k} of capital (so the long side is +1 and "
        "the short side is -1), and hold until the next close, when the book is rebuilt "
        "from scratch. Run it over every day the data allows. "
        f"{PHRASING[params.get('phrasing', 'net')]}\n\n"
        "The data is described in `DATA_DICTIONARY.md`.\n"
    )
    dictionary = (
        "# Data dictionary\n\n## prices.csv\n\n"
        "- `date`: trading day, `YYYY-MM-DD`\n"
        "- `ticker`: stock identifier\n"
        "- `close`: split-adjusted closing price; there are no dividends\n\n"
        "## costs.csv\n\n"
        "Estimated trading cost of each stock, from the execution desk.\n\n"
        "- `ticker`: stock identifier\n"
        "- `cost_bps`: one-way cost in basis points of the value traded\n\n"
        "To charge it to a strategy that is rebalanced to target weights at a close: for "
        "each stock, multiply `cost_bps / 10000` by the absolute change in its target "
        "weight (weights as fractions of capital; the weight before the first trade is "
        "zero, and positions are taken to stay at their targets between closes), sum over "
        "stocks, and subtract the total from the return of the holding period that starts "
        "at that close.\n"
    )
    costs = cost_bps.rename("cost_bps").rename_axis("ticker").reset_index()
    files = {"prices.csv": long_frame(close, "close"), "costs.csv": costs}
    return Built(key, request, dictionary, files)


def naive(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Gross returns: the cost file is never opened."""
    close = read_wide(workdir / "prices.csv", "close")
    return {FIELD: sharpe(strategy_returns(close, None, params.get("k", 8)), 252)}


def oracle(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Charge the cost schedule on the change in target weights."""
    close = read_wide(workdir / "prices.csv", "close")
    costs = pd.read_csv(workdir / "costs.csv").set_index("ticker")["cost_bps"]
    return {FIELD: sharpe(strategy_returns(close, costs, params.get("k", 8)), 252)}
