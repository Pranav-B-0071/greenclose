"""Wide (date x symbol) view of the bar panel that feature functions compute on.

Every frame here is indexed by the union trading calendar of the data handed in, which is already
truncated at the cutoff by the store. All feature functions must use *trailing* windows only, so
that truncating the panel at date T never changes any value dated <= T (brief section 13.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PanelContext:
    dates: pd.DatetimeIndex
    symbols: list[str]
    sector_of: dict[str, str]
    adj_close: pd.DataFrame
    open_adj: pd.DataFrame
    high_adj: pd.DataFrame
    low_adj: pd.DataFrame
    close: pd.DataFrame  # raw, split-adjusted only
    volume: pd.DataFrame
    ret: pd.DataFrame  # daily close-to-close return from adj_close
    nifty_close: pd.Series
    nifty_ret: pd.Series
    vix: pd.Series | None
    params: dict[str, Any] = field(default_factory=dict)
    _store: dict[str, Any] = field(default_factory=dict)

    # ---- feature results

    def set(self, name: str, value: Any) -> None:
        self._store[name] = value

    def get(self, name: str) -> Any:
        return self._store[name]

    def has(self, name: str) -> bool:
        return name in self._store

    # ---- helpers

    def sector_frame(self) -> pd.DataFrame:
        """date x symbol frame of sector labels (constant over time)."""
        return pd.DataFrame({s: self.sector_of.get(s, "unknown") for s in self.symbols}, index=self.dates)

    @classmethod
    def from_bars(
        cls,
        bars: pd.DataFrame,
        index: pd.DataFrame,
        *,
        sector_of: dict[str, str],
        benchmark: str,
        vix_symbol: str | None,
        params: dict[str, Any] | None = None,
    ) -> PanelContext:
        bars = bars.sort_values(["date", "symbol"])
        dates = pd.DatetimeIndex(np.sort(bars["date"].unique()))
        symbols = sorted(bars["symbol"].unique().tolist())

        def wide(col: str) -> pd.DataFrame:
            w = bars.pivot(index="date", columns="symbol", values=col)
            return w.reindex(index=dates, columns=symbols).astype("float64")

        adj_close = wide("adj_close")
        idx = index.sort_values("date") if len(index) else index
        nifty = (
            idx.loc[idx["symbol"] == benchmark].set_index("date")["close"].reindex(dates).astype("float64")
            if len(idx)
            else pd.Series(np.nan, index=dates, dtype="float64")
        )
        vix: pd.Series | None = None
        if vix_symbol and len(idx) and (idx["symbol"] == vix_symbol).any():
            vix = (
                idx.loc[idx["symbol"] == vix_symbol]
                .set_index("date")["close"]
                .reindex(dates)
                .astype("float64")
            )
        return cls(
            dates=dates,
            symbols=symbols,
            sector_of=dict(sector_of),
            adj_close=adj_close,
            open_adj=wide("open_adj"),
            high_adj=wide("high_adj"),
            low_adj=wide("low_adj"),
            close=wide("close"),
            volume=wide("volume"),
            ret=adj_close / adj_close.shift(1) - 1.0,
            nifty_close=nifty,
            nifty_ret=nifty / nifty.shift(1) - 1.0,
            vix=vix,
            params=dict(params or {}),
        )
