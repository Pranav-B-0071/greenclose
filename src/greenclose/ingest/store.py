"""`ParquetBarStore`: the only way code reads raw bars. Every read requires an explicit `cutoff`.

`get_bars(symbols, *, cutoff)` returns rows with date <= cutoff. `cutoff` is keyword-only with no default,
so omitting it is a `TypeError` (brief invariant 1). In `record_access` mode the store records the max
date it handed out for every call, which the leakage suite checks against the cutoff (section 13.5).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from greenclose.ingest.clean import remove_phantom_days
from greenclose.ingest.snapshots import read_bars, read_index, read_manifest, resolve_snapshot
from greenclose.ingest.yahoo import canonicalise_bars

DERIVED = ["adj_factor", "open_adj", "high_adj", "low_adj"]


def _check_cutoff(cutoff: object) -> pd.Timestamp:
    if not isinstance(cutoff, pd.Timestamp):
        raise TypeError(f"cutoff must be a pandas Timestamp, got {type(cutoff).__name__}")
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("Asia/Kolkata").tz_localize(None)
    return cutoff.normalize()


def add_derived(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out["adj_factor"] = out["adj_close"] / out["close"]
    out["open_adj"] = out["open"] * out["adj_factor"]
    out["high_adj"] = out["high"] * out["adj_factor"]
    out["low_adj"] = out["low"] * out["adj_factor"]
    return out


class ParquetBarStore:
    def __init__(
        self,
        bars: pd.DataFrame,
        index: pd.DataFrame,
        *,
        snapshot_id: str,
        record_access: bool = False,
    ) -> None:
        cleaned, self.phantom_days = remove_phantom_days(canonicalise_bars(bars))
        self._bars = add_derived(cleaned)
        if len(index):
            idx = canonicalise_bars(index)
            idx = idx.loc[~idx["date"].isin(self.phantom_days)].reset_index(drop=True)
            self._index = add_derived(idx)
        else:
            self._index = index
        self._snapshot_id = snapshot_id
        self._dates = np.sort(self._bars["date"].unique())
        self.record_access = record_access
        self.access_log: list[tuple[pd.Timestamp, pd.Timestamp | None]] = []

    # ---- construction

    @classmethod
    def from_snapshot(cls, snapshot_dir: Path, *, record_access: bool = False) -> ParquetBarStore:
        m = read_manifest(snapshot_dir)
        return cls(
            read_bars(snapshot_dir),
            read_index(snapshot_dir),
            snapshot_id=m.snapshot_id,
            record_access=record_access,
        )

    @classmethod
    def open(
        cls, snapshots_root: Path, snapshot_id: str | None = None, *, record_access: bool = False
    ) -> ParquetBarStore:
        return cls.from_snapshot(resolve_snapshot(snapshots_root, snapshot_id), record_access=record_access)

    # ---- protocol

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def symbols(self) -> list[str]:
        return sorted(self._bars["symbol"].unique().tolist())

    def trading_dates(self, *, cutoff: pd.Timestamp) -> pd.DatetimeIndex:
        c = _check_cutoff(cutoff)
        return pd.DatetimeIndex(self._dates[self._dates <= np.datetime64(c)])

    def get_bars(
        self,
        symbols: Sequence[str],
        *,
        cutoff: pd.Timestamp,
        lookback_days: int | None = None,
    ) -> pd.DataFrame:
        """Bars with date <= cutoff for the given symbols. `cutoff` is keyword-only and REQUIRED."""
        c = _check_cutoff(cutoff)
        mask = (self._bars["date"] <= c) & self._bars["symbol"].isin(list(symbols))
        if lookback_days is not None:
            dates = self.trading_dates(cutoff=c)
            if len(dates) > lookback_days:
                mask &= self._bars["date"] >= dates[-lookback_days]
        out = self._bars.loc[mask].reset_index(drop=True)
        if self.record_access:
            self.access_log.append((c, out["date"].max() if len(out) else None))
        return out

    def get_index(self, *, cutoff: pd.Timestamp) -> pd.DataFrame:
        c = _check_cutoff(cutoff)
        if not len(self._index):
            return self._index.copy()
        out = self._index.loc[self._index["date"] <= c].reset_index(drop=True)
        if self.record_access:
            self.access_log.append((c, out["date"].max() if len(out) else None))
        return out

    def max_date_touched(self) -> pd.Timestamp | None:
        seen = [d for _, d in self.access_log if d is not None]
        return max(seen) if seen else None

    def violations(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Calls whose returned max date exceeded the cutoff. Must always be empty."""
        return [(c, d) for c, d in self.access_log if d is not None and d > c]
