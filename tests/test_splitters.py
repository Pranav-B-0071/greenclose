"""Split integrity (brief section 13.6): no date on both sides, purge and embargo respected, no shuffle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greenclose.models.splitters import PurgedGroupTimeSeriesSplit, training_mask, walk_forward_schedule


def _date_idx(n_days: int = 300, per_day: int = 10) -> np.ndarray:
    return np.repeat(np.arange(n_days), per_day)


def test_no_date_appears_on_both_sides() -> None:
    idx = _date_idx()
    for train, val in PurgedGroupTimeSeriesSplit(4, purge_days=1, embargo_days=5).split(idx):
        assert not set(idx[train]) & set(idx[val])
        assert len(train) and len(val)


def test_purge_and_embargo_gaps() -> None:
    idx = _date_idx()
    sp = PurgedGroupTimeSeriesSplit(5, purge_days=1, embargo_days=5, horizon_days=1)
    for g in sp.gaps(idx):
        # before the block: t + h + purge < v_start  ->  v_start - t >= h + purge + 1 = 3
        assert g["gap_before"] == -1 or g["gap_before"] >= 3
        # after the block: t > v_end + embargo  ->  t - v_end >= 6
        assert g["gap_after"] == -1 or g["gap_after"] >= 6


def test_multi_day_horizon_widens_the_purge() -> None:
    idx = _date_idx()
    sp = PurgedGroupTimeSeriesSplit(3, purge_days=1, embargo_days=2, horizon_days=10)
    for g in sp.gaps(idx):
        assert g["gap_before"] == -1 or g["gap_before"] >= 12
        assert g["gap_after"] == -1 or g["gap_after"] >= 11  # max(embargo, horizon) + 1


def test_shuffle_is_impossible() -> None:
    with pytest.raises(TypeError):
        PurgedGroupTimeSeriesSplit(3, shuffle=True)  # type: ignore[call-arg]


def test_training_mask_requires_label_known_before_retrain() -> None:
    t = np.arange(100)
    avail = t + 1  # 1-day horizon
    m = training_mask(t, avail, retrain_idx=50, purge_days=1)
    assert m[:48].all() and not m[48:].any()  # t + 1 + 1 < 50  ->  t <= 47
    m2 = training_mask(t, avail, retrain_idx=50, purge_days=1, rolling_start_idx=20)
    assert not m2[:20].any() and m2[20:48].all()


def test_walk_forward_schedule_is_monthly_and_reselects_annually() -> None:
    dates = pd.bdate_range("2015-01-01", periods=1200)
    oos = dates[dates >= "2018-01-01"]
    pts = walk_forward_schedule(
        dates, oos_dates=oos, min_train_years=3, rolling_years=None, hparam_reselect_months=12
    )
    assert all(p.retrain_date.day <= 3 for p in pts)  # first trading day of each month
    assert sum(len(p.predict_dates) for p in pts) == len(oos)
    assert [p.reselect_hparams for p in pts][:13] == [True] + [False] * 11 + [True]
    assert all(p.rolling_start_idx is None for p in pts)
    rolling = walk_forward_schedule(dates, oos_dates=oos, min_train_years=3, rolling_years=2)
    assert all(p.rolling_start_idx is not None and p.rolling_start_idx < p.retrain_idx for p in rolling)
