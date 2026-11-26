"""Best-of-many reporting: the best of many backtests, judged as if it were the only one.

Data. Daily returns of N variants of a strategy, none with any skill: each is drawn with
a true mean of zero. The data dictionary states that these are all the variants that
were backtested.

Trap. The request asks for the Sharpe ratio of the best variant and for the probability
that its true Sharpe ratio is above zero, in the probabilistic Sharpe ratio framework
(Bailey and Lopez de Prado). The naive answer evaluates the winner on its own: its
probabilistic Sharpe ratio against zero is close to one. The corrected answer accounts
for having picked the best of N: the deflated Sharpe ratio measures the winner against
the Sharpe ratio the best of N skill-less variants is expected to show.

The Sharpe ratio of the best variant is the same either way; it is recorded but does not
decide the outcome. The key's corrected value comes from factor-research-pipeline's
implementation of the deflated Sharpe ratio; the oracle below is a separate one.

There is no control: a request about a single pre-registered strategy has nothing to
deflate, and the selection trap's control already covers a fixed choice.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from factorpipe.sharpe import deflated_sharpe_ratio

from qtb.stats import sharpe
from qtb.task import AnswerField, AnswerKey, Family, Kind
from qtb.traps.base import Answer, Built, business_days

TRAP = "multiple_testing"
FAMILY = Family.METHOD
TOLERANCE = 0.1
FIELD = "prob_skill"


def build(task_id: str, seed: int, control: bool, params: Mapping[str, Any]) -> Built:
    if control:
        raise ValueError("the multiple-testing trap has no control task")
    rng = np.random.default_rng(seed)
    n_variants, n_days = params.get("n_variants", 40), params.get("n_days", 756)
    dates = business_days("2022-01-03", n_days)
    columns = [f"v{j:03d}" for j in range(1, n_variants + 1)]
    returns = pd.DataFrame(
        0.01 * rng.standard_normal((n_days, n_variants)), index=dates, columns=columns
    )

    per_period = returns.mean() / returns.std(ddof=1)
    best = per_period.idxmax()
    trial_variance = float(per_period.var(ddof=1))
    deflated = deflated_sharpe_ratio(returns[best].to_numpy(), n_variants, trial_variance)
    corrected, naive = deflated.dsr, deflated.psr
    best_sharpe = sharpe(returns[best], 252)

    fields = (
        AnswerField(
            FIELD,
            "the probability that the true Sharpe ratio of the variant you would launch is "
            "above zero, as a decimal between 0 and 1, in the probabilistic Sharpe ratio "
            "framework of Bailey and Lopez de Prado",
            corrected,
            naive,
            TOLERANCE,
        ),
        AnswerField(
            "best_sharpe",
            "the annualized Sharpe ratio of the best variant: mean daily return over its "
            "sample standard deviation (n - 1 denominator), times the square root of 252",
            best_sharpe,
            best_sharpe,
            0.05,
        ),
    )
    key = AnswerKey(
        task_id=task_id,
        kind=Kind.TRAP,
        family=FAMILY,
        trap=TRAP,
        mirrors=TRAP,
        primary=FIELD,
        fields=fields,
        seed=seed,
        params=dict(params),
        extra={"naive": naive, "hurdle": deflated.hurdle, "trial_variance": trial_variance},
    )
    years = n_days / 252
    request = (
        "# Request\n\n"
        f"A colleague backtested variants of a daily trading strategy over about {years:.0f} "
        "years and wants to launch the one with the highest Sharpe ratio. Their backtests "
        "are in `variants.csv`. Report the annualized Sharpe ratio of that best variant, "
        "and the probability that its true Sharpe ratio is above zero. Use the "
        "probabilistic Sharpe ratio framework of Bailey and Lopez de Prado, working with "
        "the daily (not annualized) Sharpe ratio, and the skewness and kurtosis of the "
        "daily returns.\n\n"
        "The data is described in `DATA_DICTIONARY.md`.\n"
    )
    dictionary = (
        "# Data dictionary\n\n## variants.csv\n\n"
        f"Daily returns of every variant of the strategy that was backtested: {n_variants} "
        "parameter combinations, in the order they were tried. No other variant was run.\n\n"
        "- `date`: trading day, `YYYY-MM-DD`\n"
        f"- `v001` to `v{n_variants:03d}`: the variant's return on that day, as a decimal, "
        "after costs\n"
    )
    variants = returns.rename_axis("date").reset_index()
    variants["date"] = variants["date"].dt.strftime("%Y-%m-%d")
    return Built(key, request, dictionary, {"variants.csv": variants})


_NORMAL = NormalDist()
EULER_GAMMA = 0.5772156649015329


def _psr(r: np.ndarray, benchmark: float) -> float:
    """Probabilistic Sharpe ratio of daily returns ``r`` against a daily benchmark."""
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std(ddof=0)
    skew, kurt = float(np.mean(z**3)), float(np.mean(z**4))
    se = math.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (len(r) - 1))
    return _NORMAL.cdf((sr - benchmark) / se)


def _best(workdir: Path) -> tuple[pd.DataFrame, str, pd.Series]:
    variants = pd.read_csv(workdir / "variants.csv", index_col="date")
    per_period = variants.mean() / variants.std(ddof=1)
    return variants, str(per_period.idxmax()), per_period


def naive(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Judge the winner as if it had been the only backtest."""
    variants, best, _ = _best(workdir)
    r = variants[best].to_numpy()
    return {FIELD: _psr(r, 0.0), "best_sharpe": sharpe(r, 252)}


def oracle(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Measure the winner against the expected maximum Sharpe ratio of N skill-less trials."""
    variants, best, per_period = _best(workdir)
    n = variants.shape[1]
    sd = float(per_period.std(ddof=1))
    hurdle = sd * (
        (1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n)
        + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n * math.e))
    )
    r = variants[best].to_numpy()
    return {FIELD: _psr(r, hurdle), "best_sharpe": sharpe(r, 252)}
