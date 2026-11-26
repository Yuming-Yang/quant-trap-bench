import math

import numpy as np
import pandas as pd
import pytest

from qtb.stats import annualized_mean, annualized_vol, rank_ic_series, sharpe, statistic


def test_annualized_statistics_by_hand():
    r = [0.01, 0.03, -0.01, 0.01]
    mean, sd = 0.01, math.sqrt(((0.0) ** 2 + 0.02**2 + 0.02**2 + 0.0) / 3)
    assert annualized_mean(r, 12) == pytest.approx(mean * 12)
    assert annualized_vol(r, 12) == pytest.approx(sd * math.sqrt(12))
    assert sharpe(r, 12) == pytest.approx(mean / sd * math.sqrt(12))
    assert statistic("sharpe", r, 12) == sharpe(r, 12)


def test_statistics_skip_missing_values_and_need_two_points():
    assert annualized_mean([0.01, np.nan, 0.03], 1) == pytest.approx(0.02)
    with pytest.raises(ValueError, match="two finite"):
        sharpe([0.01, np.nan], 252)


def test_rank_ic_uses_ranks_and_only_complete_pairs():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    signal = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, np.nan], [1.0, np.nan, np.nan, 2.0]],
        index=dates,
    )
    target = pd.DataFrame(
        [[10.0, 20.0, 30.0, 1000.0], [3.0, 2.0, 1.0, 5.0], [1.0, 2.0, 3.0, 4.0]], index=dates
    )
    ic = rank_ic_series(signal, target)
    assert ic.iloc[0] == pytest.approx(1.0)  # monotone, despite the outlier
    assert ic.iloc[1] == pytest.approx(-1.0)  # the fourth asset has no signal
    assert len(ic) == 2  # the third date has only two usable assets


def test_rank_ic_requires_the_same_grid():
    a = pd.DataFrame([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="same grid"):
        rank_ic_series(a, a.T)
