"""Forward-return labels (brief section 7).

    fwd_c2c = adj_close[t+h] / adj_close[t] - 1        overnight gap + full next session(s)
    fwd_c2o = open_adj[t+1] / adj_close[t] - 1         overnight gap only
    fwd_o2c = adj_close[t+h] / open_adj[t+1] - 1       next session(s) only

Per date, across the universe, on the configured base return:
    y_raw       fwd                       reported only
    y_demeaned  fwd - mean(fwd)           PRIMARY TRAINING TARGET
    y_resid     fwd - beta_60d * nifty_fwd
    y_rank      within-day rank, 0..n-1   lambdarank label
    y_green     1[fwd > 0]                diagnostic only
    y_beat      1[fwd > mean(fwd)]        the meaningful binary label

`label_available_idx` is the trading-day ordinal at which the label becomes known (t_idx + h). The
+/-5 sigma winsorisation of continuous targets is fitted inside the training fold (models/train.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from greenclose.config import AppConfig
from greenclose.domain.protocols import BarStore

HORIZON_COLUMN = {"c2c": "fwd_c2c", "c2o": "fwd_c2o", "o2c": "fwd_o2c"}
Y_COLUMNS = ["y_raw", "y_demeaned", "y_resid", "y_rank", "y_green", "y_beat"]


@dataclass
class LabelPanel:
    frame: pd.DataFrame
    horizon_days: int
    base: str  # which fwd_* the y_* columns are built from
    snapshot_id: str = ""


def forward_returns(
    store: BarStore, symbols: list[str], *, cutoff: pd.Timestamp, horizon_days: int, benchmark: str
) -> pd.DataFrame:
    """fwd_c2c / fwd_c2o / fwd_o2c / nifty_fwd_c2c / label_available_idx on the store's full calendar."""
    bars = store.get_bars(symbols, cutoff=cutoff)
    index = store.get_index(cutoff=cutoff)
    dates = pd.DatetimeIndex(np.sort(bars["date"].unique()))
    adj = bars.pivot(index="date", columns="symbol", values="adj_close").reindex(dates)
    opn = bars.pivot(index="date", columns="symbol", values="open_adj").reindex(dates)
    h = horizon_days
    fwd_c2c = adj.shift(-h) / adj - 1.0
    fwd_c2o = opn.shift(-1) / adj - 1.0
    fwd_o2c = adj.shift(-h) / opn.shift(-1) - 1.0
    nifty = index.loc[index["symbol"] == benchmark].set_index("date")["close"].reindex(dates)
    nifty_fwd = nifty.shift(-h) / nifty - 1.0
    pos = pd.Series(np.arange(len(dates)), index=dates)

    def stack(w: pd.DataFrame, name: str) -> pd.Series:
        s = w.stack(future_stack=True)
        s.name = name
        return s

    out = pd.concat([stack(fwd_c2c, "fwd_c2c"), stack(fwd_c2o, "fwd_c2o"), stack(fwd_o2c, "fwd_o2c")], axis=1)
    out.index = out.index.set_names(["date", "symbol"])
    d = out.index.get_level_values("date")
    out["nifty_fwd_c2c"] = nifty_fwd.reindex(d).to_numpy()
    out["t_idx"] = pos.reindex(d).to_numpy()
    out["label_available_idx"] = out["t_idx"] + h
    return out


def build_labels(
    features: pd.DataFrame,
    store: BarStore,
    cfg: AppConfig,
    *,
    horizon_days: int | None = None,
    base: str | None = None,
) -> LabelPanel:
    """Labels aligned to the feature panel's (date, symbol) rows. Needs `beta_60d` from the features."""
    h = horizon_days or cfg.base.labels.horizon_days
    base = base or cfg.base.labels.horizon
    fwd_col = HORIZON_COLUMN[base]
    symbols = sorted(features.index.get_level_values("symbol").unique().tolist())
    cutoff = pd.Timestamp(features.index.get_level_values("date").max())
    # the store hands out bars <= cutoff; labels need bars up to cutoff + h, so ask for the horizon too
    far = cutoff + pd.Timedelta(days=int(h * 2 + 10))
    fwd = forward_returns(store, symbols, cutoff=far, horizon_days=h, benchmark=cfg.base.data.benchmark)
    fwd = fwd.reindex(features.index)

    y = fwd[fwd_col]
    g = y.groupby(level="date")
    mean = g.transform("mean")
    frame = fwd.copy()
    frame["y_raw"] = y
    frame["y_demeaned"] = y - mean
    frame["y_resid"] = y - features["beta_60d"] * fwd["nifty_fwd_c2c"]
    # deterministic integer rank: ties broken by symbol order via a stable sort
    order = pd.DataFrame({"y": y, "sym": y.index.get_level_values("symbol")})
    frame["y_rank"] = (
        order.sort_values(["y", "sym"], kind="mergesort")
        .groupby(level="date")
        .cumcount()
        .reindex(frame.index)
        .where(y.notna())
        .astype("float64")
    )
    frame["y_green"] = (y > 0).astype("float64").where(y.notna())
    frame["y_beat"] = (y > mean).astype("float64").where(y.notna())
    frame["n_labelled"] = g.transform("count").astype("float64")
    return LabelPanel(frame=frame, horizon_days=h, base=base, snapshot_id=store.snapshot_id)


def summarise(panel: LabelPanel) -> pd.DataFrame:
    f = panel.frame
    y = f["y_raw"]
    per_day_green = f["y_green"].groupby(level="date").mean()
    rows = {
        "rows": len(f),
        "labelled_rows": int(y.notna().sum()),
        "mean_fwd_bps": float(y.mean() * 1e4),
        "std_fwd_pct": float(y.std() * 100),
        "mean_daily_green_frac": float(per_day_green.mean()),
        "std_daily_green_frac": float(per_day_green.std()),
        "mean_y_beat": float(f["y_beat"].mean()),
        "y_demeaned_std_pct": float(f["y_demeaned"].std() * 100),
        "y_resid_std_pct": float(f["y_resid"].std() * 100),
    }
    return pd.DataFrame({"value": rows})
