"""Time-aware splitters (brief invariant 5, section 9.1).

- `PurgedGroupTimeSeriesSplit`: k contiguous blocks of *dates* as validation folds, with a purge of
  `purge_days` around each block and an `embargo_days` gap after it. Rows are grouped by date; no date
  ever appears on both sides; there is no shuffle argument and there never will be.
- `training_mask`: rows usable for a retrain at date index T: the label must be known strictly before
  T minus the purge (t + h + purge < T), optionally restricted to a rolling window.
- `walk_forward_schedule`: monthly retrain points after a minimum training window.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


class PurgedGroupTimeSeriesSplit:
    def __init__(
        self, n_splits: int = 3, *, purge_days: int = 1, embargo_days: int = 5, horizon_days: int = 1
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.horizon_days = horizon_days

    def split(self, date_idx: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """`date_idx`: integer trading-day ordinal per row. Yields (train_rows, val_rows) index arrays."""
        uniq = np.unique(date_idx)
        blocks = np.array_split(uniq, self.n_splits)
        h, p, e = self.horizon_days, self.purge_days, self.embargo_days
        for block in blocks:
            if len(block) == 0:
                continue
            v_start, v_end = int(block[0]), int(block[-1])
            val = np.flatnonzero((date_idx >= v_start) & (date_idx <= v_end))
            # before the block: the row's label (t + h) must be known before the block, minus the purge
            before = date_idx + h + p < v_start
            # after the block: skip the embargo so the block's labels (up to v_end + h) cannot leak in
            after = date_idx > v_end + max(e, h)
            train = np.flatnonzero(before | after)
            yield train, val

    def gaps(self, date_idx: np.ndarray) -> list[dict[str, int]]:
        """For tests: the minimum distance between validation and training dates on each side."""
        out = []
        for train, val in self.split(date_idx):
            tv, vv = date_idx[train], date_idx[val]
            before = tv[tv < vv.min()]
            after = tv[tv > vv.max()]
            out.append(
                {
                    "gap_before": int(vv.min() - before.max()) if len(before) else -1,
                    "gap_after": int(after.min() - vv.max()) if len(after) else -1,
                    "n_train": int(len(train)),
                    "n_val": int(len(val)),
                }
            )
        return out


def training_mask(
    t_idx: np.ndarray,
    label_available_idx: np.ndarray,
    *,
    retrain_idx: int,
    purge_days: int,
    rolling_start_idx: int | None = None,
) -> np.ndarray:
    """Rows whose label is known strictly before the retrain day, minus the purge."""
    mask = label_available_idx + purge_days < retrain_idx
    if rolling_start_idx is not None:
        mask &= t_idx >= rolling_start_idx
    return np.asarray(mask, dtype=bool)


@dataclass(frozen=True, slots=True)
class RetrainPoint:
    retrain_date: pd.Timestamp
    retrain_idx: int
    predict_dates: pd.DatetimeIndex  # this retrain's model scores these dates
    rolling_start_idx: int | None
    reselect_hparams: bool


def walk_forward_schedule(
    dates: pd.DatetimeIndex,
    *,
    oos_dates: pd.DatetimeIndex,
    min_train_years: int,
    rolling_years: int | None = None,
    hparam_reselect_months: int = 12,
) -> list[RetrainPoint]:
    """Monthly retrain points (first trading day of each month inside the OOS window)."""
    index_of = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    months = pd.Series(oos_dates).dt.to_period("M")
    firsts = pd.DatetimeIndex(pd.Series(oos_dates).groupby(months.to_numpy()).min().to_numpy())
    points: list[RetrainPoint] = []
    for n, first in enumerate(firsts):
        nxt = firsts[n + 1] if n + 1 < len(firsts) else oos_dates[-1] + pd.Timedelta(days=1)
        predict = oos_dates[(oos_dates >= first) & (oos_dates < nxt)]
        rolling_start = None
        if rolling_years is not None:
            start = first - pd.DateOffset(years=rolling_years)
            rolling_start = int(np.searchsorted(dates.to_numpy(), np.datetime64(start)))
        points.append(
            RetrainPoint(
                retrain_date=pd.Timestamp(first),
                retrain_idx=index_of[pd.Timestamp(first)],
                predict_dates=predict,
                rolling_start_idx=rolling_start,
                reselect_hparams=(n % hparam_reselect_months == 0),
            )
        )
    return points
