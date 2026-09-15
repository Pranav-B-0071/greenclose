"""`build(cutoff, store, universe, cfg)`: the pure feature builder (brief invariant 2).

No network, no global mutable state, no randomness. Output for every row dated <= T is bitwise
identical whether the store is cut at T or at any later date (invariant 3; tested in section 13.2).

Layout of the returned frame (MultiIndex date, symbol; sorted):
- Tier-1 raw values `<name>` and their cross-sectional ranks `<name>_rk` in [-1, 1]
- Tier-2 context columns broadcast to every symbol of the day
- non-model context: sector, cluster_id, staleness_flag, n_in_xs, close, adj_close, volume,
  traded_value_20d
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import greenclose.features.context  # noqa: F401  (registers Tier-2 features)
import greenclose.features.price  # noqa: F401  (registers Tier-1 features)
from greenclose.config import AppConfig, UniverseCfg
from greenclose.domain.protocols import BarStore
from greenclose.features.panel import PanelContext
from greenclose.features.registry import REGISTRY
from greenclose.ops.logging import get_logger

log = get_logger(__name__)

CONTEXT_COLUMNS = [
    "close",
    "adj_close",
    "volume",
    "traded_value_20d",
    "sector",
    "cluster_id",
    "staleness_flag",
    "n_in_xs",
]


@dataclass
class FeaturePanel:
    frame: pd.DataFrame
    tier1: list[str]
    tier2: list[str]
    skipped_days: list[pd.Timestamp] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)
    cutoff: pd.Timestamp | None = None
    snapshot_id: str = ""
    registry_version: str = ""

    @property
    def model_columns(self) -> list[str]:
        """What a model may see: ranked Tier-1 plus Tier-2 context."""
        return [f"{c}_rk" for c in self.tier1] + list(self.tier2)

    @property
    def n_model_features(self) -> int:
        return len(self.model_columns)


def rank_to_unit(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-date rank mapped to [-1, 1] via 2*(rank-1)/(n-1)-1. NaN stays NaN; a lone value maps to 0."""
    g = frame[cols].groupby(level="date")
    rank = g.rank(method="average")
    n = g.transform("count")
    denom = (n - 1).where(n > 1)
    unit = 2.0 * (rank - 1.0) / denom - 1.0
    return unit.where(n > 1, 0.0).where(frame[cols].notna())


def winsorise_xs(frame: pd.DataFrame, cols: list[str], lo: float, hi: float) -> pd.DataFrame:
    g = frame[cols].groupby(level="date")
    q_lo = g.quantile(lo)
    q_hi = g.quantile(hi)
    dates = frame.index.get_level_values("date")
    lo_rows = q_lo.reindex(dates).to_numpy()
    hi_rows = q_hi.reindex(dates).to_numpy()
    vals = frame[cols].to_numpy(dtype="float64")
    clipped = np.where(np.isnan(vals), np.nan, np.clip(vals, lo_rows, hi_rows))
    return pd.DataFrame(clipped, index=frame.index, columns=cols)


def build(
    cutoff: pd.Timestamp,
    store: BarStore,
    universe: UniverseCfg,
    cfg: AppConfig,
) -> FeaturePanel:
    bars = store.get_bars(universe.symbols, cutoff=cutoff)
    index = store.get_index(cutoff=cutoff)
    if bars.empty:
        raise ValueError(f"no bars at or before {cutoff.date()}")
    ctx = PanelContext.from_bars(
        bars,
        index,
        sector_of=universe.sector_of,
        benchmark=cfg.base.data.benchmark,
        vix_symbol=cfg.base.data.vix,
        params={"n_clusters": cfg.base.features.n_clusters, "corr_window": cfg.base.features.corr_window},
    )
    for spec in REGISTRY.ordered():
        ctx.set(spec.name, spec.fn(ctx))

    present = ctx.adj_close.notna()
    long_index = present.stack(future_stack=True)
    long_index = long_index[long_index].index  # (date, symbol) rows that have a bar
    long_index = long_index.set_names(["date", "symbol"])

    cols: dict[str, pd.Series] = {}
    # Exclusions are decided by configuration, never by looking at the data: a data-driven omission
    # would depend on the whole sample and break truncation invariance.
    excluded = set(cfg.base.features.exclude)
    if cfg.base.data.vix is None:
        excluded.add("india_vix")
    tier1 = [c for c in REGISTRY.names(tier=1, model_input=True) if c not in excluded]
    tier2 = [c for c in REGISTRY.names(tier=2) if c not in excluded]
    for name in tier1 + ["cluster_id"]:
        wide = ctx.get(name)
        cols[name] = wide.stack(future_stack=True).reindex(long_index).astype("float64")
    for name in tier2:
        series = ctx.get(name)
        cols[name] = pd.Series(
            series.reindex(long_index.get_level_values("date")).to_numpy(), index=long_index
        )
    for name, wide in (("close", ctx.close), ("adj_close", ctx.adj_close), ("volume", ctx.volume)):
        cols[name] = wide.stack(future_stack=True).reindex(long_index).astype("float64")
    tv20 = (ctx.close * ctx.volume).rolling(20, min_periods=20).mean()
    cols["traded_value_20d"] = tv20.stack(future_stack=True).reindex(long_index).astype("float64")
    frame = pd.DataFrame(cols, index=long_index).sort_index()

    omitted = sorted(excluded)

    n_in_xs = frame.groupby(level="date").size()
    min_n = cfg.base.data.min_symbols_per_day
    skipped = [pd.Timestamp(d) for d in n_in_xs.index[n_in_xs < min_n]]
    if skipped:
        log.warning(
            "days_skipped_too_few_symbols",
            n=len(skipped),
            first=str(skipped[0].date()),
            last=str(skipped[-1].date()),
        )
        frame = frame.loc[~frame.index.get_level_values("date").isin(skipped)]
        n_in_xs = n_in_xs[n_in_xs >= min_n]

    frame["n_in_xs"] = n_in_xs.reindex(frame.index.get_level_values("date")).to_numpy().astype("float64")
    frame["sector"] = frame.index.get_level_values("symbol").map(
        lambda s: universe.sector_of.get(s, "unknown")
    )
    frame["staleness_flag"] = (frame["volume"] <= 0).astype("int64")

    lo, hi = cfg.base.features.winsor_pct
    wins = winsorise_xs(frame, tier1, lo, hi)
    ranked = rank_to_unit(wins, tier1)
    ranked.columns = [f"{c}_rk" for c in tier1]
    frame = pd.concat([frame, ranked], axis=1)

    ordered = tier1 + [f"{c}_rk" for c in tier1] + tier2 + CONTEXT_COLUMNS
    frame = frame.loc[:, ordered]
    return FeaturePanel(
        frame=frame,
        tier1=tier1,
        tier2=tier2,
        skipped_days=skipped,
        omitted=omitted,
        cutoff=cutoff,
        snapshot_id=store.snapshot_id,
        registry_version=REGISTRY.version_hash(),
    )


def nan_report(panel: FeaturePanel) -> pd.DataFrame:
    f = panel.frame
    cols = panel.tier1 + panel.tier2
    return pd.DataFrame(
        {
            "nan_frac": f[cols].isna().mean().round(4),
            "lookback": [REGISTRY.get(c).lookback_days for c in cols],
        }
    )
