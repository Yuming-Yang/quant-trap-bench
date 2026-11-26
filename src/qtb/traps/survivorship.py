"""Survivorship: delisted stocks are missing from the main returns file.

Market. Monthly returns from 2015 to 2024 for about 200 stocks: a common market factor
with dispersed betas plus idiosyncratic noise. Some stocks list late. About a third
delist before the end. Most delistings follow a year of decline and carry a large
negative delisting return; the rest are takeovers at a premium.

Trap. ``returns.csv`` is a current-constituents file: only stocks still listed at the end
appear, with their whole history. The delisted stocks are in a separate archive file
that the data dictionary describes. The request asks for the equal-weighted market
return, which must include every stock listed in each month.

Control. ``returns.csv`` holds every stock ever listed, and the dictionary says so.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtb.stats import statistic
from qtb.task import AnswerField, AnswerKey, Family, Kind
from qtb.traps.base import Answer, Built, long_frame, month_ends, read_wide, statistic_text, tickers

TRAP = "survivorship"
FAMILY = Family.DATA
TOLERANCE = {"mean": 0.004, "sharpe": 0.05}


def build(task_id: str, seed: int, control: bool, params: Mapping[str, Any]) -> Built:
    stat = params["statistic"]
    rng = np.random.default_rng(seed)
    n_assets, n_months = params.get("n_assets", 200), 120
    dates = month_ends("2015-01-31", n_months)
    names = tickers(n_assets, rng)

    market = 0.008 + 0.045 * rng.standard_normal(n_months)
    beta = 1.0 + 0.3 * rng.standard_normal(n_assets)
    returns = market[:, None] * beta[None, :] + 0.08 * rng.standard_normal((n_months, n_assets))

    first = np.zeros(n_assets, dtype=int)
    late = rng.random(n_assets) < 0.15
    first[late] = rng.integers(1, n_months - 24, size=int(late.sum()))
    last = np.full(n_assets, n_months - 1)
    delisted = rng.random(n_assets) < params.get("delist_share", 0.4)
    for i in np.flatnonzero(delisted):
        last[i] = rng.integers(first[i] + 12, n_months - 1)
        if rng.random() < 0.75:  # performance delisting: a year of decline, then -30%
            returns[last[i] - 11 : last[i] + 1, i] -= 0.04
            returns[last[i], i] = (1 + returns[last[i], i]) * (1 - 0.30) - 1
        else:  # takeover at a premium
            returns[last[i], i] = (1 + returns[last[i], i]) * (1 + 0.25) - 1
    returns = np.maximum(returns, -0.95)

    month = np.arange(n_months)[:, None]
    listed = (month >= first[None, :]) & (month <= last[None, :])
    wide = pd.DataFrame(np.where(listed, returns, np.nan), index=dates, columns=names)
    survivors = [name for name, d in zip(names, delisted, strict=True) if not d]
    gone = [name for name, d in zip(names, delisted, strict=True) if d]

    corrected = statistic(stat, wide.mean(axis=1), 12)
    naive = statistic(stat, wide[survivors].mean(axis=1), 12)

    label, definition = statistic_text(stat, "monthly", 12)
    field_name = f"ew_{stat}"
    spec = AnswerField(
        field_name,
        f"the {label} of the equal-weighted market portfolio, {definition}",
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
        extra={"n_assets": float(n_assets), "n_delisted": float(len(gone)), "naive": naive},
    )
    request = (
        "# Request\n\n"
        "We need a baseline for this equity market. Compute the monthly return of the "
        "equal-weighted market portfolio from January 2015 to December 2024: each month, "
        "the simple average of the returns of the stocks listed in that month. "
        f"Report its {label}.\n\n"
        "The data is described in `DATA_DICTIONARY.md`.\n"
    )
    columns = (
        "- `date`: month end, `YYYY-MM-DD`\n"
        "- `ticker`: stock identifier\n"
        "- `ret`: total return over the month ending on `date`, as a decimal\n"
    )
    if control:
        files = {"returns.csv": long_frame(wide, "ret")}
        dictionary = (
            "# Data dictionary\n\n## returns.csv\n\n"
            "Monthly total returns for every stock listed on the market at any time from "
            "January 2015 to December 2024, including stocks that later delisted. A stock "
            "appears only in the months it was listed. The return of a delisted stock's "
            "last month includes its delisting return.\n\n" + columns
        )
    else:
        files = {
            "returns.csv": long_frame(wide[survivors], "ret"),
            "delisted_returns.csv": long_frame(wide[gone], "ret"),
        }
        dictionary = (
            "# Data dictionary\n\n## returns.csv\n\n"
            "Monthly total returns from the vendor's constituent file, downloaded on "
            "2025-01-06. It covers the stocks listed on the market on 2024-12-31, with "
            "their history back to January 2015 or to their listing date.\n\n"
            + columns
            + "\n## delisted_returns.csv\n\n"
            "Monthly total returns from the vendor's archive for stocks that delisted "
            "between January 2015 and December 2024. Same columns as `returns.csv`. The "
            "return of a stock's last month includes its delisting return.\n"
        )
    return Built(key, request, dictionary, files)


def _answer(wide: pd.DataFrame, params: Mapping[str, Any]) -> Answer:
    stat = params["statistic"]
    return {f"ew_{stat}": statistic(stat, wide.mean(axis=1), 12)}


def naive(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Equal-weight whatever is in the main returns file."""
    return _answer(read_wide(workdir / "returns.csv", "ret"), params)


def oracle(workdir: Path, params: Mapping[str, Any]) -> Answer:
    """Add the delisted stocks back before averaging."""
    frames = [pd.read_csv(workdir / "returns.csv")]
    archive = workdir / "delisted_returns.csv"
    if archive.is_file():
        frames.append(pd.read_csv(archive))
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    return _answer(long.pivot(index="date", columns="ticker", values="ret").sort_index(), params)
