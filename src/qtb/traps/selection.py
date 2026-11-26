"""Full-sample selection: pick the best lookback on all the data, report its Sharpe.

Market. Twenty years of monthly returns from the factor pipeline's synthetic panel, with
one weak persistent characteristic, so past returns carry a little momentum.

Trap. The request asks for the best momentum lookback (1 to 12 months) and the Sharpe
ratio to expect from it. The data dictionary states the team's protocol: choices are
made on the research period (2001-2014) and the last five years are a holdout, used
once to measure the chosen strategy. The corrected answer chooses the lookback on the
research period and reports its holdout Sharpe ratio. The naive answer chooses on the
full sample and reports the full-sample Sharpe ratio of the winner, which is biased up
by the selection.

Control. The lookback is fixed in the request and there is no holdout
protocol: the full-sample Sharpe ratio of that one strategy is the answer. Reporting
the holdout alone, or searching over lookbacks, is a false alarm.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from factorpipe.synthetic import PlantedFactor

from qtb.stats import sharpe
from qtb.task import AnswerField, AnswerKey, Family, Kind
from qtb.traps.base import Answer, Built, long_frame, read_wide
from qtb.traps.panel import renamed, synthetic_panel

TRAP = "selection"
FAMILY = Family.METHOD
TOLERANCE = 0.05
LOOKBACKS = tuple(range(1, 13))
FIRST_HOLD, LAST_HOLD = "2001-01-31", "2019-12-31"
HOLDOUT_START = "2015-01-31"
QUANTILE_WORDS = {5: "quintile", 10: "decile"}


def momentum_returns(returns: pd.DataFrame, lookback: int, n_groups: int) -> pd.Series:
    """Monthly long-short momentum returns, indexed by holding month.

    At each month end ``t``, stocks with returns in all ``lookback`` months ending at
    ``t`` are ranked on their compounded return over those months. The top
    ``1/n_groups`` (rounded down) are bought and the bottom ones sold, equal-weighted.
    The return of month ``t+1`` is the long average minus the short average, over the
    selected stocks that have a return that month.
    """
    growth = np.log1p(returns)
    past = growth.rolling(lookback, min_periods=lookback).sum()
    forward = returns.shift(-1)
    out = {}
    for i, date in enumerate(returns.index[:-1]):
        score = past.loc[date].dropna()
        size = len(score) // n_groups
        if size == 0:
            continue
        ranked = score.sort_values(kind="mergesort")
        nxt = forward.loc[date]
        out[returns.index[i + 1]] = (
            nxt[ranked.index[-size:]].mean() - nxt[ranked.index[:size]].mean()
        )
    series = pd.Series(out, dtype=float)
    return series[(series.index >= FIRST_HOLD) & (series.index <= LAST_HOLD)]


def all_lookbacks(returns: pd.DataFrame, n_groups: int) -> dict[int, pd.Series]:
    return {lookback: momentum_returns(returns, lookback, n_groups) for lookback in LOOKBACKS}


def choose(series: dict[int, pd.Series], end: str | None = None) -> tuple[int, float]:
    """Lookback with the highest Sharpe ratio over holding months up to ``end``."""
    scores = {
        lookback: sharpe(s if end is None else s[s.index < end], 12)
        for lookback, s in series.items()
    }
    best = max(scores, key=lambda lookback: (scores[lookback], -lookback))
    return best, scores[best]


def corrected_and_naive(series: dict[int, pd.Series]) -> tuple[int, float, int, float]:
    research_best, _ = choose(series, end=HOLDOUT_START)
    holdout = series[research_best]
    corrected = sharpe(holdout[holdout.index >= HOLDOUT_START], 12)
    full_best, naive = choose(series)
    return research_best, corrected, full_best, naive


def build(task_id: str, seed: int, control: bool, params: Mapping[str, Any]) -> Built:
    n_groups = params.get("n_groups", 5)
    synthetic, names = synthetic_panel(
        seed,
        n_assets=params.get("n_assets", 300),
        n_months=240,
        start="2000-01-31",
        planted=(PlantedFactor("quality", ic=params.get("ic", 0.03), persistence=0.9),),
    )
    returns = renamed(synthetic.panel.returns, names)
    group_word = QUANTILE_WORDS[n_groups]
    strategy = (
        f"At each month end, rank the stocks that have returns for all of the past L months "
        f"on their compounded return over those L months. Buy the top {group_word} and sell "
        f"short the bottom {group_word} (each {group_word} is 1/{n_groups} of the ranked "
        "stocks, rounded down), equal-weighted within each side, and hold for the next "
        "month. The monthly strategy return is the average return of the long stocks minus "
        "the average return of the short stocks; a stock with no return in the holding "
        "month is left out of its side's average. Holding months run from January 2001 to "
        "December 2019."
    )
    sharpe_text = (
        "the annualized Sharpe ratio of the monthly long-short returns: their mean divided "
        "by their sample standard deviation (n - 1 denominator), times the square root of "
        "12; no risk-free rate"
    )
    if control:
        lookback = params["lookback"]
        series = momentum_returns(returns, lookback, n_groups)
        value = sharpe(series, 12)
        fields = (AnswerField("sharpe", sharpe_text, value, None, TOLERANCE),)
        # A likely over-correction: treating the last five years as a holdout and
        # reporting only those, although nothing was chosen.
        extra: dict[str, float] = {"false_alarm": sharpe(series[series.index >= HOLDOUT_START], 12)}
        request = (
            "# Request\n\n"
            "Evaluate a cross-sectional momentum strategy with a lookback of L = "
            f"{lookback} months. {strategy} Report its annualized Sharpe ratio.\n\n"
            "The data is described in `DATA_DICTIONARY.md`.\n"
        )
        protocol = ""
    else:
        research_best, corrected, full_best, naive = corrected_and_naive(
            all_lookbacks(returns, n_groups)
        )
        fields = (
            AnswerField(
                "sharpe",
                "the annualized Sharpe ratio to expect from the chosen strategy, as "
                + sharpe_text.removeprefix("the "),
                corrected,
                naive,
                TOLERANCE,
            ),
            AnswerField(
                "lookback",
                "the chosen lookback L, in months",
                float(research_best),
                float(full_best),
                0.5,
            ),
        )
        extra = {"naive": naive, "full_sample_best": float(full_best)}
        request = (
            "# Request\n\n"
            "We want to trade cross-sectional momentum in this market and need to pick the "
            f"lookback L, from 1 to 12 months. {strategy} Find the best lookback and report "
            "the annualized Sharpe ratio we should expect from the strategy you pick.\n\n"
            "The data is described in `DATA_DICTIONARY.md`.\n"
        )
        protocol = (
            "\n## Research protocol\n\n"
            "The team splits every dataset in two. Months up to December 2014 are the "
            "research period: all choices (signals, parameters) are made on it. Months from "
            "January 2015 on are the holdout, reserved for measuring the performance of the "
            "single strategy chosen on the research period. A holding month belongs to the "
            "period that contains it.\n"
        )
    key = AnswerKey(
        task_id=task_id,
        kind=Kind.CONTROL if control else Kind.TRAP,
        family=FAMILY,
        trap=None if control else TRAP,
        mirrors=TRAP,
        primary="sharpe",
        fields=fields,
        seed=seed,
        params=dict(params),
        extra=extra,
    )
    dictionary = (
        "# Data dictionary\n\n## returns.csv\n\n"
        "Monthly total returns, January 2000 to December 2019. A stock appears only in the "
        "months it was listed; the last month of a delisted stock includes its delisting "
        "return.\n\n"
        "- `date`: month end, `YYYY-MM-DD`\n"
        "- `ticker`: stock identifier\n"
        "- `ret`: total return over the month ending on `date`, as a decimal\n" + protocol
    )
    return Built(key, request, dictionary, {"returns.csv": long_frame(returns, "ret")})


def naive(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Search on all months and report the winner's in-sample Sharpe ratio.

    When the request fixes the lookback (the control), there is nothing to search.
    """
    returns = read_wide(workdir / "returns.csv", "ret")
    n_groups = params.get("n_groups", 5)
    if "lookback" in params:
        return {"sharpe": sharpe(momentum_returns(returns, params["lookback"], n_groups), 12)}
    best, value = choose(all_lookbacks(returns, n_groups))
    return {"sharpe": value, "lookback": float(best)}


def oracle(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Choose on the research period, measure on the holdout."""
    returns = read_wide(workdir / "returns.csv", "ret")
    research_best, corrected, _, _ = corrected_and_naive(
        all_lookbacks(returns, params.get("n_groups", 5))
    )
    return {"sharpe": corrected, "lookback": float(research_best)}
