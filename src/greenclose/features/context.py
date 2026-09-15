"""Tier-2 market-context features (one value per day) and the non-model correlation clusters.

These cannot rank stocks, but they interact with stock features and gate the abstention rule.
`cluster_id` is *not* a model input: the decision layer uses it to take at most one pick per
correlation cluster (Ledoit-Wolf shrunk 60-day correlation matrix, average-linkage clustering).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from greenclose.features.panel import PanelContext
from greenclose.features.registry import feature

CORR_WINDOW = 60
DEFAULT_CLUSTERS = 8


@feature("nifty_ret_1d", tier=2, lookback_days=1)
def nifty_ret_1d(ctx: PanelContext) -> pd.Series:
    return ctx.nifty_ret


@feature("nifty_ret_5d", tier=2, lookback_days=5)
def nifty_ret_5d(ctx: PanelContext) -> pd.Series:
    return ctx.nifty_close / ctx.nifty_close.shift(5) - 1.0


@feature("nifty_vol_20d", tier=2, lookback_days=21)
def nifty_vol_20d(ctx: PanelContext) -> pd.Series:
    # the index series has a few gaps (see price.INDEX_GAP_TOLERANCE); tolerate one inside the window
    return ctx.nifty_ret.rolling(20, min_periods=19).std()


@feature("nifty_vs_200dma", tier=2, lookback_days=200)
def nifty_vs_200dma(ctx: PanelContext) -> pd.Series:
    return ctx.nifty_close / ctx.nifty_close.rolling(200, min_periods=190).mean() - 1.0


def _xs_values(wide: pd.DataFrame) -> pd.Series:
    """Stack a wide frame into the valid (date, symbol) values only.

    Cross-sectional reductions must run on this long form, not on `wide.<op>(axis=1)`: a wide frame
    built from data truncated at T has fewer columns (symbols that list later are absent), and
    numpy's summation order depends on the row width, which breaks bitwise truncation invariance.
    """
    return wide.stack(future_stack=True).dropna()


@feature("breadth", tier=2, lookback_days=1)
def breadth(ctx: PanelContext) -> pd.Series:
    """Fraction of the universe that closed up today."""
    r = _xs_values(ctx.ret)
    g = r.groupby(level=0)
    return ((r > 0).groupby(level=0).sum() / g.count()).reindex(ctx.dates)


@feature("xs_dispersion", tier=2, lookback_days=1)
def xs_dispersion(ctx: PanelContext) -> pd.Series:
    return _xs_values(ctx.ret).groupby(level=0).std().reindex(ctx.dates)


@feature("india_vix", tier=2, lookback_days=0)
def india_vix(ctx: PanelContext) -> pd.Series:
    """India VIX level. All-NaN when the series is unavailable; build() then omits the column."""
    if ctx.vix is None:
        return pd.Series(np.nan, index=ctx.dates, dtype="float64")
    return ctx.vix


# ----------------------------------------------------------------------------- correlation structure


def _corr_windows(ctx: PanelContext) -> list[tuple[list[str], np.ndarray] | None]:
    """Per date: (symbols with a complete trailing window, their Ledoit-Wolf shrunk correlation)."""
    key = "_corr_windows"
    if ctx.has(key):
        return ctx.get(key)  # type: ignore[no-any-return]
    window = int(ctx.params.get("corr_window", CORR_WINDOW))
    r = ctx.ret.to_numpy(dtype="float64")
    cols = list(ctx.ret.columns)
    out: list[tuple[list[str], np.ndarray] | None] = []
    for i in range(len(ctx.dates)):
        if i + 1 < window:
            out.append(None)
            continue
        block = r[i + 1 - window : i + 1]
        ok = ~np.isnan(block).any(axis=0)
        if ok.sum() < 3:
            out.append(None)
            continue
        x = block[:, ok]
        cov = LedoitWolf().fit(x).covariance_
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        out.append(([c for c, k in zip(cols, ok, strict=True) if k], corr))
    ctx.set(key, out)
    return out


@feature("avg_pairwise_corr_60d", tier=2, lookback_days=CORR_WINDOW)
def avg_pairwise_corr_60d(ctx: PanelContext) -> pd.Series:
    vals = np.full(len(ctx.dates), np.nan)
    for i, item in enumerate(_corr_windows(ctx)):
        if item is None:
            continue
        _, corr = item
        n = corr.shape[0]
        vals[i] = (corr.sum() - n) / (n * (n - 1))
    return pd.Series(vals, index=ctx.dates)


@feature("cluster_id", tier=1, lookback_days=CORR_WINDOW, model_input=False, ranked=False)
def cluster_id(ctx: PanelContext) -> pd.DataFrame:
    """Average-linkage cluster label (1..n_clusters) on the shrunk 60-day correlation matrix."""
    n_clusters = int(ctx.params.get("n_clusters", DEFAULT_CLUSTERS))
    col_pos = {c: j for j, c in enumerate(ctx.ret.columns)}
    values = np.full((len(ctx.dates), len(ctx.ret.columns)), np.nan, dtype="float64")
    for i, item in enumerate(_corr_windows(ctx)):
        if item is None:
            continue
        syms, corr = item
        dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
        np.fill_diagonal(dist, 0.0)
        z = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(z, t=min(n_clusters, len(syms)), criterion="maxclust")
        for s, lab in zip(syms, labels, strict=True):
            values[i, col_pos[s]] = float(lab)
    return pd.DataFrame(values, index=ctx.dates, columns=ctx.ret.columns)
