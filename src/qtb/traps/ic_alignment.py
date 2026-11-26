"""Misaligned IC: the signal and the return sit on the same row, but the return is
contemporaneous.

Market. Monthly returns from the factor pipeline's synthetic panel, which plants a
characteristic with a known IC against next month's return.

Trap. ``panel.csv`` has one row per stock and month with a ``signal`` and a ``ret``
column. The data dictionary says that ``ret`` is the return over the month ending on the
row's date, the same month the signal is measured at the end of. The request asks for
the usual predictive rank IC: the signal at a month end against the return over the
following month. Correlating the two columns row by row measures something else, and
the error can go either way:

- ``signal="momentum"``: the signal is the 12-month return including the current month,
  so it is mechanically correlated with the same-row return. The naive IC is large; the
  predictive one is small.
- ``signal="planted"``: the signal is the planted characteristic, with low persistence.
  Its same-row correlation is the true IC times its persistence, so the naive IC
  understates the predictive one.

Control. The return column is the following month's return (``fwd_ret``), and the
dictionary says so. Row-by-row correlation is correct; shifting it again is a false
alarm.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from factorpipe.synthetic import PlantedFactor

from qtb.stats import rank_ic_series
from qtb.task import AnswerField, AnswerKey, Family, Kind
from qtb.traps.base import Answer, Built
from qtb.traps.panel import renamed, synthetic_panel

TRAP = "ic_alignment"
FAMILY = Family.METHOD
TOLERANCE = 0.005
FIELD = "mean_rank_ic"
# The file covers these month ends. The last one has no following month in the file, so
# the predictive IC uses one month fewer than a row-by-row correlation does.
FIRST_DATE, LAST_DATE = "2006-01-31", "2023-12-31"
SIGNAL_TEXT = {
    "momentum": "the stock's compounded return over the 12 months ending on `date`, "
    "including the month ending on `date`",
    "planted": "a proprietary quality score computed from information available at the "
    "close on `date`",
}


def build(task_id: str, seed: int, control: bool, params: Mapping[str, Any]) -> Built:
    kind = params.get("signal", "momentum")
    persistence = params.get("persistence", 0.9)
    synthetic, names = synthetic_panel(
        seed,
        n_assets=params.get("n_assets", 400),
        n_months=228,
        start="2005-01-31",
        planted=(PlantedFactor("quality", ic=params.get("ic", 0.04), persistence=persistence),),
    )
    returns = renamed(synthetic.panel.returns, names)
    if kind == "momentum":
        signal = np.expm1(np.log1p(returns).rolling(12, min_periods=12).sum())
    else:
        signal = renamed(synthetic.panel.characteristics["quality"], names)
    window = (returns.index >= FIRST_DATE) & (returns.index <= LAST_DATE)
    forward = returns.shift(-1)  # all NaN on the last date: the panel ends there
    corrected = float(rank_ic_series(signal[window], forward[window]).mean())
    naive = float(rank_ic_series(signal[window], returns[window]).mean())
    extra = {"naive": naive, "planted_ic": synthetic.truth.ic["quality"]}
    if control:
        # A likely over-correction: shifting a return that is already forward-looking.
        extra["false_alarm"] = float(
            rank_ic_series(signal[window], forward.shift(-1)[window]).mean()
        )

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
        extra=extra,
    )
    return_column = "fwd_ret" if control else "ret"
    long = pd.DataFrame(
        {
            "signal": signal[window].stack(),
            return_column: (forward if control else returns)[window].stack(),
        }
    ).dropna(subset=["signal"])
    long = long.rename_axis(["date", "ticker"]).reset_index()
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")
    if control:
        return_text = (
            "- `fwd_ret`: total return over the month *following* `date` (from the end of "
            "the month ending on `date` to the end of the next month), as a decimal; empty "
            "if the stock was not listed in that month\n"
        )
    else:
        return_text = (
            "- `ret`: total return over the month ending on `date`, as a decimal; empty if "
            "the stock was not listed in that month\n"
        )
    request = (
        "# Request\n\n"
        "How well does `signal` predict stock returns? At each month end, compute the "
        "Spearman rank correlation, across stocks, between the signal and the stock's "
        "return over the following month, using stocks that have both. Report the mean of "
        "these monthly rank correlations (the mean rank IC) over every month end the data "
        "allows.\n\n"
        "The data is described in `DATA_DICTIONARY.md`.\n"
    )
    dictionary = (
        "# Data dictionary\n\n## panel.csv\n\n"
        f"One row per stock and month end, {long['date'].min()} to {long['date'].max()}.\n\n"
        "- `date`: month end, `YYYY-MM-DD`\n"
        "- `ticker`: stock identifier\n"
        f"- `signal`: {SIGNAL_TEXT[kind]}\n" + return_text
    )
    return Built(key, request, dictionary, {"panel.csv": long})


def _mean_ic(workdir: Path, shift: bool) -> Answer:
    panel = pd.read_csv(workdir / "panel.csv", parse_dates=["date"])
    column = "fwd_ret" if "fwd_ret" in panel else "ret"
    signal = panel.pivot(index="date", columns="ticker", values="signal").sort_index()
    target = panel.pivot(index="date", columns="ticker", values=column).reindex_like(signal)
    if shift:
        target = target.shift(-1)
    return {FIELD: float(rank_ic_series(signal, target).mean())}


def naive(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Correlate the two columns row by row."""
    return _mean_ic(workdir, shift=False)


def oracle(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Pair each signal with the next row's return."""
    return _mean_ic(workdir, shift=True)
