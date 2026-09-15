"""Deterministic synthetic NSE-like panel for offline tests. No network, fixed seed.

Returns follow a one-market-factor + sector-factor + idiosyncratic model with realistic scale
(daily sigma ~1.8%), so the leakage suite runs on data that has the right shape but, by
construction, no cross-sectional predictability unless a test injects one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from greenclose.config import UniverseCfg
from greenclose.ingest.store import ParquetBarStore

SECTORS = ["fin", "it", "energy", "auto", "pharma", "fmcg", "metals", "materials", "infra", "consumer"]


@dataclass
class SyntheticData:
    bars: pd.DataFrame
    index: pd.DataFrame
    universe: UniverseCfg
    dates: pd.DatetimeIndex

    def store(self, *, record_access: bool = False) -> ParquetBarStore:
        return ParquetBarStore(self.bars, self.index, snapshot_id="synthetic", record_access=record_access)


def synthetic_panel(
    n_symbols: int = 50,
    n_days: int = 800,
    seed: int = 0,
    start: str = "2018-01-01",
    n_sectors: int = 10,
    missing_frac: float = 0.001,
    late_listers: int = 2,
    late_listing_day: int = 150,
) -> SyntheticData:
    """`late_listers` symbols have no bars before `late_listing_day`, like HDFCLIFE/SBILIFE in the real
    universe, so a build truncated early has fewer symbols than the full build."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    sectors = {s: SECTORS[i % n_sectors] for i, s in enumerate(symbols)}

    beta = rng.normal(1.0, 0.25, n_symbols)
    idio_sd = rng.uniform(0.010, 0.020, n_symbols)
    mkt = rng.normal(0.0004, 0.010, n_days)
    sec = rng.normal(0.0, 0.006, (n_days, n_sectors))
    sec_idx = np.array([SECTORS.index(sectors[s]) for s in symbols])
    idio = rng.normal(0.0, 1.0, (n_days, n_symbols)) * idio_sd
    ret = mkt[:, None] * beta[None, :] + sec[:, sec_idx] + idio

    adj = 100.0 * np.exp(np.cumsum(ret, axis=0) + rng.uniform(-0.5, 0.5, n_symbols)[None, :])
    gap = rng.normal(0.0, 0.004, (n_days, n_symbols))
    open_ = np.vstack([adj[:1], adj[:-1]]) * (1 + gap)
    intraday = np.abs(rng.normal(0.0, 0.008, (n_days, n_symbols)))
    high = np.maximum(open_, adj) * (1 + intraday)
    low = np.minimum(open_, adj) * (1 - intraday)
    # raw close: a synthetic dividend-style adjustment factor that steps once a year
    factor = np.ones((n_days, n_symbols))
    for j in range(n_symbols):
        for k in range(1, n_days // 250 + 1):
            factor[: n_days - k * 250, j] *= 1 - rng.uniform(0.005, 0.02)
    close = adj / factor
    volume = np.exp(rng.normal(13.0, 0.6, (n_days, n_symbols))).astype("int64")

    rows = []
    for j, s in enumerate(symbols):
        df = pd.DataFrame(
            {
                "date": dates,
                "symbol": s,
                "open": open_[:, j] / factor[:, j],
                "high": high[:, j] / factor[:, j],
                "low": low[:, j] / factor[:, j],
                "close": close[:, j],
                "adj_close": adj[:, j],
                "volume": volume[:, j],
            }
        )
        rows.append(df)
    bars = pd.concat(rows, ignore_index=True)
    if late_listers > 0:
        late = set(symbols[-late_listers:])
        cut = dates[min(late_listing_day, n_days - 1)]
        bars = bars.loc[~(bars["symbol"].isin(late) & (bars["date"] < cut))].reset_index(drop=True)
    if missing_frac > 0:
        drop = rng.random(len(bars)) < missing_frac
        bars = bars.loc[~drop].reset_index(drop=True)

    level = 10000.0 * np.exp(np.cumsum(mkt))
    index = pd.DataFrame(
        {
            "date": dates,
            "symbol": "^NSEI",
            "open": level,
            "high": level * 1.005,
            "low": level * 0.995,
            "close": level,
            "adj_close": level,
            "volume": 0,
        }
    )
    vix_level = 15.0 + 5.0 * np.abs(np.cumsum(rng.normal(0, 0.05, n_days)))
    vix = index.copy()
    vix["symbol"] = "^INDIAVIX"
    for c in ("open", "high", "low", "close", "adj_close"):
        vix[c] = vix_level
    index = pd.concat([index, vix], ignore_index=True)

    by_sector: dict[str, list[str]] = {}
    for s in symbols:
        by_sector.setdefault(sectors[s], []).append(s)
    universe = UniverseCfg(fixed_as_of=start, benchmark="^NSEI", suffix="", universe=by_sector, dropped={})
    return SyntheticData(bars=bars, index=index, universe=universe, dates=dates)
