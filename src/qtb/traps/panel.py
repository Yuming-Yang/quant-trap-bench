"""Monthly panels for the method traps, from factor-research-pipeline's synthetic panel.

That generator plants characteristics with a known information coefficient and adds
late listings and delistings, so a method trap can be built on returns whose true
predictability is known. Asset names are replaced with random tickers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from factorpipe.synthetic import PlantedFactor, SyntheticConfig, SyntheticPanel, make_panel

from qtb.traps.base import tickers


def synthetic_panel(
    seed: int,
    n_assets: int,
    n_months: int,
    start: str,
    planted: tuple[PlantedFactor, ...],
) -> tuple[SyntheticPanel, list[str]]:
    """A factorpipe panel and the random tickers its assets are renamed to."""
    config = SyntheticConfig(
        n_assets=n_assets,
        n_months=n_months,
        start=start,
        planted=planted,
        n_noise=0,
        reported_factor=None,
        seed=seed,
    )
    synthetic = make_panel(config)
    names = tickers(n_assets, np.random.default_rng([seed, 7]))
    return synthetic, names


def renamed(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out.columns = names
    return out
