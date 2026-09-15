"""Tier-1 stock-level features (brief section 6.1). Every window is trailing; nothing is backfilled.

Returns and price-shape features use adjusted prices; volume-value features use the raw (split-adjusted)
close, which is what traded value actually was. `min_periods == window` everywhere except the two
volume features, where a zero-volume print inside the window would otherwise blank 20 days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greenclose.features.panel import PanelContext
from greenclose.features.registry import feature

RET_HORIZONS = (1, 2, 3, 5, 10, 20, 60, 120, 240)
VOL_WINDOWS = (5, 20, 60)
SECTOR_HORIZONS = (1, 5, 20)
RESID_HORIZONS = (1, 5, 20)
BETA_WINDOW = 60
INDEX_GAP_TOLERANCE = 3  # missing index prints tolerated inside a 60-day market window


def _roll(
    df: pd.DataFrame | pd.Series, n: int, min_periods: int | None = None
) -> pd.core.window.rolling.Rolling:
    return df.rolling(n, min_periods=n if min_periods is None else min_periods)


# ----------------------------------------------------------------------------- reversal / momentum


def _make_ret(n: int) -> None:
    @feature(f"ret_{n}d", tier=1, lookback_days=n)
    def _ret(ctx: PanelContext) -> pd.DataFrame:
        return ctx.adj_close / ctx.adj_close.shift(n) - 1.0


for _n in RET_HORIZONS:
    _make_ret(_n)


@feature("mom_12_1", tier=1, lookback_days=240)
def mom_12_1(ctx: PanelContext) -> pd.DataFrame:
    """240-day return excluding the most recent 20 days."""
    return ctx.adj_close.shift(20) / ctx.adj_close.shift(240) - 1.0


# ----------------------------------------------------------------------------- volatility


def _make_vol(n: int) -> None:
    @feature(f"vol_{n}d", tier=1, lookback_days=n + 1)
    def _vol(ctx: PanelContext) -> pd.DataFrame:
        return _roll(ctx.ret, n).std()


for _n in VOL_WINDOWS:
    _make_vol(_n)


@feature("vol_ratio", tier=1, lookback_days=61, deps=("vol_5d", "vol_60d"))
def vol_ratio(ctx: PanelContext) -> pd.DataFrame:
    return ctx.get("vol_5d") / ctx.get("vol_60d")


@feature("ret_1d_scaled", tier=1, lookback_days=21, deps=("ret_1d", "vol_20d"))
def ret_1d_scaled(ctx: PanelContext) -> pd.DataFrame:
    return ctx.get("ret_1d") / ctx.get("vol_20d")


@feature("ret_5d_scaled", tier=1, lookback_days=21, deps=("ret_5d", "vol_20d"))
def ret_5d_scaled(ctx: PanelContext) -> pd.DataFrame:
    return ctx.get("ret_5d") / (ctx.get("vol_20d") * np.sqrt(5.0))


@feature("atr_14_pct", tier=1, lookback_days=15)
def atr_14_pct(ctx: PanelContext) -> pd.DataFrame:
    prev_close = ctx.adj_close.shift(1)
    tr = pd.concat(
        [ctx.high_adj - ctx.low_adj, (ctx.high_adj - prev_close).abs(), (ctx.low_adj - prev_close).abs()],
        axis=0,
        keys=["hl", "hc", "lc"],
    )
    true_range = tr.groupby(level=1).max()  # elementwise max across the three, NaN-aware
    true_range = true_range.reindex(ctx.dates)
    return _roll(true_range, 14).mean() / ctx.adj_close


# ----------------------------------------------------------------------------- volume


@feature("vol_anomaly", tier=1, lookback_days=21)
def vol_anomaly(ctx: PanelContext) -> pd.DataFrame:
    """log(volume_t / median of the previous 20 days' volume). Zero-volume days are NaN, not -inf."""
    vol = ctx.volume.where(ctx.volume > 0)
    med = _roll(vol.shift(1), 20, min_periods=15).median()
    return np.log(vol / med)


@feature("log_traded_value_20d", tier=1, lookback_days=20)
def log_traded_value_20d(ctx: PanelContext) -> pd.DataFrame:
    tv = ctx.close * ctx.volume
    return np.log(_roll(tv, 20).mean().where(lambda x: x > 0))


@feature("amihud_20d", tier=1, lookback_days=21)
def amihud_20d(ctx: PanelContext) -> pd.DataFrame:
    tv = (ctx.close * ctx.volume).where(ctx.volume > 0)
    return _roll(ctx.ret.abs() / tv, 20, min_periods=15).mean()


# ----------------------------------------------------------------------------- range / position


@feature("pos_in_range_20d", tier=1, lookback_days=20)
def pos_in_range_20d(ctx: PanelContext) -> pd.DataFrame:
    lo = _roll(ctx.low_adj, 20).min()
    hi = _roll(ctx.high_adj, 20).max()
    return (ctx.adj_close - lo) / (hi - lo)


@feature("dist_52w_high", tier=1, lookback_days=252)
def dist_52w_high(ctx: PanelContext) -> pd.DataFrame:
    return ctx.adj_close / _roll(ctx.high_adj, 252).max() - 1.0


@feature("hl_pct_20d", tier=1, lookback_days=20)
def hl_pct_20d(ctx: PanelContext) -> pd.DataFrame:
    return _roll((ctx.high_adj - ctx.low_adj) / ctx.adj_close, 20).mean()


# ----------------------------------------------------------------------------- session decomposition


@feature("gap", tier=1, lookback_days=21, deps=("vol_20d",))
def gap(ctx: PanelContext) -> pd.DataFrame:
    """Overnight gap open_t / close_{t-1} - 1, standardised by vol_20d."""
    return (ctx.open_adj / ctx.adj_close.shift(1) - 1.0) / ctx.get("vol_20d")


@feature("c2o_sum_5d", tier=1, lookback_days=6)
def c2o_sum_5d(ctx: PanelContext) -> pd.DataFrame:
    return _roll(ctx.open_adj / ctx.adj_close.shift(1) - 1.0, 5).sum()


@feature("o2c_sum_5d", tier=1, lookback_days=5)
def o2c_sum_5d(ctx: PanelContext) -> pd.DataFrame:
    return _roll(ctx.adj_close / ctx.open_adj - 1.0, 5).sum()


# ----------------------------------------------------------------------------- market-relative


def _rolling_market_moments(ctx: PanelContext) -> dict[str, pd.DataFrame | pd.Series]:
    """Trailing 60-day moments of stock and NIFTY returns, computed once and cached on the context."""
    key = "_mkt_moments"
    if ctx.has(key):
        return ctx.get(key)  # type: ignore[no-any-return]
    # The free index series has a handful of gaps (1 January, Muhurat weekend sessions) on days the
    # stocks did trade. A strict window would blank 60 rows for every stock after each gap, so the
    # market moments tolerate up to INDEX_GAP_TOLERANCE missing observations. Nothing is imputed.
    n = BETA_WINDOW
    mp = n - INDEX_GAP_TOLERANCE
    r, m = ctx.ret, ctx.nifty_ret
    mean_r = _roll(r, n, mp).mean()
    mean_m = _roll(m, n, mp).mean()
    mean_rm = _roll(r.mul(m, axis=0), n, mp).mean()
    mean_mm = _roll(m * m, n, mp).mean()
    mean_rr = _roll(r * r, n, mp).mean()
    cov = mean_rm.sub(mean_r.mul(mean_m, axis=0), axis=0)
    var_m = mean_mm - mean_m * mean_m
    var_r = mean_rr - mean_r * mean_r
    beta = cov.div(var_m, axis=0)
    alpha = mean_r.sub(beta.mul(mean_m, axis=0), axis=0)
    corr = cov.div(np.sqrt(var_m), axis=0) / np.sqrt(var_r)
    out: dict[str, pd.DataFrame | pd.Series] = {"beta": beta, "alpha": alpha, "corr": corr}
    ctx.set(key, out)
    return out


@feature("beta_60d", tier=1, lookback_days=61)
def beta_60d(ctx: PanelContext) -> pd.DataFrame:
    return _rolling_market_moments(ctx)["beta"]  # type: ignore[return-value]


@feature("corr_60d_nifty", tier=1, lookback_days=61)
def corr_60d_nifty(ctx: PanelContext) -> pd.DataFrame:
    return _rolling_market_moments(ctx)["corr"]  # type: ignore[return-value]


def _daily_residual(ctx: PanelContext) -> pd.DataFrame:
    key = "_daily_resid"
    if ctx.has(key):
        return ctx.get(key)  # type: ignore[no-any-return]
    mm = _rolling_market_moments(ctx)
    beta, alpha = mm["beta"], mm["alpha"]
    fitted = alpha + beta.mul(ctx.nifty_ret, axis=0)  # type: ignore[union-attr]
    resid = ctx.ret - fitted
    ctx.set(key, resid)
    return resid


def _make_resid(k: int) -> None:
    @feature(f"resid_ret_{k}d", tier=1, lookback_days=BETA_WINDOW + k, deps=("beta_60d",))
    def _resid(ctx: PanelContext) -> pd.DataFrame:
        e = _daily_residual(ctx)
        return e if k == 1 else _roll(e, k, k - 1).sum()  # one index gap tolerated inside the window


for _k in RESID_HORIZONS:
    _make_resid(_k)


@feature("rel_strength_5d", tier=1, lookback_days=61, deps=("ret_5d", "beta_60d"))
def rel_strength_5d(ctx: PanelContext) -> pd.DataFrame:
    nifty_5d = ctx.nifty_close / ctx.nifty_close.shift(5) - 1.0
    return ctx.get("ret_5d") - ctx.get("beta_60d").mul(nifty_5d, axis=0)


# ----------------------------------------------------------------------------- sector-relative


def _loo_sector_mean(ctx: PanelContext, wide: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out mean of `wide` across the stock's own sector, per date."""
    # Sums run on the stacked valid values (long form), never across wide-frame columns: the column
    # count differs between a truncated and a full build and would change numpy's summation order.
    out = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns, dtype="float64")
    sectors: dict[str, list[str]] = {}
    for s in wide.columns:
        sectors.setdefault(ctx.sector_of.get(s, "unknown"), []).append(s)
    for members in sectors.values():
        block = wide[members]
        long = block.stack(future_stack=True).dropna()
        g = long.groupby(level=0)
        total = g.sum().reindex(wide.index)
        count = g.count().reindex(wide.index).astype("float64")
        denom = (count - 1.0).where(count > 1)
        loo = block.rsub(total, axis=0).div(denom, axis=0)
        out[members] = loo
    return out


def _make_sector_rel(k: int) -> None:
    @feature(f"sector_rel_ret_{k}d", tier=1, lookback_days=k, deps=(f"ret_{k}d",))
    def _sector_rel(ctx: PanelContext) -> pd.DataFrame:
        r = ctx.get(f"ret_{k}d")
        return r - _loo_sector_mean(ctx, r)


for _k in SECTOR_HORIZONS:
    _make_sector_rel(_k)
