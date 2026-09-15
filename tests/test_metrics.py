"""Known-answer tests for the metrics and significance libraries."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from greenclose.evaluation import metrics
from greenclose.evaluation.significance import (
    deflated_sharpe,
    expected_max_sharpe,
    newey_west,
    paired_bootstrap_diff,
    stationary_bootstrap_ci,
)


def _panel(n_days: int = 50, n: int = 20, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(n)]], names=["date", "symbol"])
    realised = pd.Series(rng.normal(0, 0.02, len(idx)), index=idx)
    return realised, pd.Series(rng.normal(0, 1, len(idx)), index=idx)


def test_ic_of_perfect_and_reversed_ranking() -> None:
    realised, _ = _panel()
    assert metrics.daily_ic(realised, realised).round(12).eq(1.0).all()
    assert metrics.daily_ic(-realised, realised).round(12).eq(-1.0).all()


def test_ic_of_noise_is_near_zero() -> None:
    realised, noise = _panel(n_days=400)
    ic = metrics.daily_ic(noise, realised)
    assert abs(ic.mean()) < 0.03


def test_precision_lift_and_spread_on_a_constructed_day() -> None:
    dates = pd.DatetimeIndex(["2024-01-02"])
    syms = [f"S{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    realised = pd.Series([0.05, 0.04, -0.01, 0.03, 0.02, -0.02, -0.03, 0.01, -0.04, -0.05], index=idx)
    pred = pd.Series(np.arange(10, 0, -1, dtype=float), index=idx)  # predicts S0 best ... S9 worst
    assert metrics.precision_at_k(pred, realised, 3, "green").iloc[0] == pytest.approx(2 / 3)
    assert metrics.precision_at_k(pred, realised, 1, "green").iloc[0] == 1.0
    assert metrics.lift_at_k(pred, realised, 3).iloc[0] == pytest.approx(2 / 3 - 0.5)
    assert metrics.decile_spread(pred, realised).iloc[0] == pytest.approx(0.05 - (-0.05))
    assert metrics.ndcg_at_k(pred, realised, 5).iloc[0] <= 1.0
    assert metrics.mrr(pred, realised).iloc[0] == 1.0


def test_pesaran_timmermann_known_values() -> None:
    # 80 correct signs out of 100 with balanced marginals: P=0.8, P*=0.5 -> clearly significant
    rng = np.random.default_rng(1)
    y = np.where(rng.random(100) < 0.5, 1.0, -1.0)
    x = y.copy()
    flip = rng.choice(100, 20, replace=False)
    x[flip] = -x[flip]
    stat, p = metrics.pesaran_timmermann(pd.Series(x), pd.Series(y))
    assert stat > 4 and p < 1e-4
    stat0, p0 = metrics.pesaran_timmermann(pd.Series(rng.normal(size=500)), pd.Series(rng.normal(size=500)))
    assert abs(stat0) < 3


def test_newey_west_on_iid_noise_and_on_a_signal() -> None:
    rng = np.random.default_rng(2)
    nw = newey_west(pd.Series(rng.normal(0, 1, 500)))
    assert abs(nw["t"]) < 3
    nw2 = newey_west(pd.Series(rng.normal(0.5, 1, 500)))
    assert nw2["t"] > 5


def test_stationary_bootstrap_ci_contains_the_mean() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0.1, 1.0, 300)
    lo, hi = stationary_bootstrap_ci(x, n_boot=500, seed=3)
    assert lo < x.mean() < hi
    d = paired_bootstrap_diff(pd.Series(x + 0.5), pd.Series(x), n_boot=300)
    assert d["diff"] == pytest.approx(0.5) and d["lo"] > 0.4


def test_deflated_sharpe_ratio_behaviour() -> None:
    assert expected_max_sharpe([0.1]) == 0.0
    assert expected_max_sharpe([0.05] * 10) == 0.0  # no variance across trials -> no inflation
    sr0 = expected_max_sharpe(list(np.linspace(-0.1, 0.1, 50)))
    assert sr0 > 0.1
    one = deflated_sharpe(0.1, n_obs=1000, skew=0.0, kurt=3.0, sharpes_tried=[0.1])
    many = deflated_sharpe(
        0.1, n_obs=1000, skew=0.0, kurt=3.0, sharpes_tried=list(np.linspace(-0.1, 0.1, 50))
    )
    assert one["dsr"] > 0.99 and many["dsr"] < one["dsr"]
    assert math.isnan(deflated_sharpe(float("nan"), n_obs=10, skew=0.0, kurt=3.0, sharpes_tried=[])["dsr"])
