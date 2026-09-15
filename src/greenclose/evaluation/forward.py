"""The nightly forward-prediction log (brief section 15).

Predictions only. No orders, no broker, no money.

Why this exists: it is the only evaluation in the project that *provably cannot leak*, because the
prediction is timestamped and written before the outcome exists. It costs nothing, needs no broker,
and starts accumulating value the day it is switched on.

Each run does three things:
  1. scores all symbols at today's cut-off with the pinned, frozen model and appends one row per
     symbol (not just the selected ones) to data/forward_log.parquet;
  2. back-fills realised outcomes for every earlier date whose forward return is now known;
  3. appends one honest line to research/forward_log_summary.md.

Statistical honesty: roughly 180 forward days are needed to establish a rank IC of 0.03 at t = 2, and
roughly 1,300 days for a P&L edge of 0.10% a day. A short forward log is an integrity check, not
evidence of an edge, and the summary says so on every line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from greenclose.backtest.costs import CostModel
from greenclose.config import AppConfig, Paths
from greenclose.domain.models import PortfolioState, Product
from greenclose.evaluation.baselines import PLANNING_SLIPPAGE_BPS, default_decision
from greenclose.ingest.store import ParquetBarStore
from greenclose.labels.targets import forward_returns
from greenclose.models.registry import ModelRegistry
from greenclose.ops.logging import get_logger
from greenclose.strategy.decide import decide_with_trace

log = get_logger(__name__)

IC_DAYS_FOR_T2 = 180  # days needed to establish a rank IC of 0.03 at t = 2
PNL_DAYS_FOR_T2 = 1300  # days needed to establish a 0.10%/day P&L edge at t = 2

PREDICTION_COLUMNS = [
    "date",
    "symbol",
    "model_version",
    "score",
    "rank",
    "prob_green",
    "prob_beat_mean",
    "expected_return_bps",
    "selected",
    "abstained",
    "abstain_reason",
    "snapshot_id",
    "config_hash",
    "git_sha",
    "cutoff_ist",
    "written_at",
]
OUTCOME_COLUMNS = [
    "fwd_c2c",
    "fwd_c2o",
    "fwd_o2c",
    "excess_vs_universe",
    "direction_correct",
    "beat_universe_mean",
    "sim_gross_pnl",
    "sim_net_pnl",
]


@dataclass
class ForwardResult:
    asof: pd.Timestamp
    n_rows: int
    n_selected: int
    backfilled_dates: list[str]
    log_path: Path
    summary_line: str
    top_picks: pd.DataFrame


def log_path(paths: Paths) -> Path:
    return paths.data_dir / "forward_log.parquet"


def read_log(paths: Paths) -> pd.DataFrame:
    p = log_path(paths)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=[*PREDICTION_COLUMNS, *OUTCOME_COLUMNS])


def _score_with_pinned(
    features_today: pd.DataFrame, registry: ModelRegistry, model_columns: list[str]
) -> tuple[pd.DataFrame, str]:
    card, model, pre = registry.load_pinned()
    missing = [c for c in card.model_columns if c not in features_today.columns]
    if missing:
        raise KeyError(f"pinned model {card.model_version} needs columns absent from the panel: {missing}")
    X = pre.transform_X(features_today[card.model_columns].to_numpy(dtype="float64"))
    from greenclose.config import load_config
    from greenclose.evaluation.ladder import build_head

    head = build_head(card.head_kind, load_config())
    out = head.predict(model, X)
    out.index = features_today.index
    return out, card.model_version


def run_forward(
    cfg: AppConfig,
    paths: Paths,
    *,
    store: ParquetBarStore,
    features: pd.DataFrame,
    asof: pd.Timestamp | None = None,
    model_version: str | None = None,
) -> ForwardResult:
    """Score the latest cross-section, append it, and back-fill whatever outcomes are now known."""
    from greenclose.evaluation.ladder import model_columns_of
    from greenclose.ops.provenance import git_sha, utc_now_iso

    dates = pd.DatetimeIndex(features.index.get_level_values("date").unique()).sort_values()
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp(dates[-1])
    if asof not in set(dates):
        raise ValueError(f"no feature row for {asof.date()}; run `greenclose fetch` and `features` first")
    today = features.xs(asof, level="date")

    registry = ModelRegistry(paths.data_dir / "models")
    preds, version = _score_with_pinned(today, registry, model_columns_of(features))
    if model_version:
        version = model_version

    # the same decision function the backtest calls: no second copy, ever
    dec = default_decision(cfg)
    if "expected_return_bps" not in preds.columns:
        dec = dec.with_(abstain_enabled=False)
    empty_portfolio = PortfolioState(
        asof=asof.date(),
        asof_index=len(dates) - 1,
        cash_settled=cfg.base.simulation.capital,
        cash_unsettled=0.0,
        positions=(),
        equity=cfg.base.simulation.capital,
        prev_asof=dates[-2].date() if len(dates) > 1 else None,
    )
    _, trace = decide_with_trace(asof.date(), today, preds, empty_portfolio, dec)

    rows = pd.DataFrame(index=today.index.rename("symbol")).reset_index()
    rows["date"] = asof
    rows["model_version"] = version
    rows["score"] = preds["score"].reindex(today.index).to_numpy()
    rows["rank"] = trace["rank"].reindex(today.index).to_numpy()
    rows["prob_green"] = (
        preds["prob_green"].reindex(today.index).to_numpy() if "prob_green" in preds else np.nan
    )
    rows["prob_beat_mean"] = (
        preds["prob_beat"].reindex(today.index).to_numpy() if "prob_beat" in preds else np.nan
    )
    rows["expected_return_bps"] = (
        preds["expected_return_bps"].reindex(today.index).to_numpy()
        if "expected_return_bps" in preds
        else np.nan
    )
    rows["selected"] = trace["action"].reindex(today.index).eq("BUY").to_numpy()
    rows["abstained"] = trace["abstained"].reindex(today.index).to_numpy()
    rows["abstain_reason"] = trace["reason"].reindex(today.index).to_numpy()
    rows["snapshot_id"] = store.snapshot_id
    rows["config_hash"] = cfg.hash()
    rows["git_sha"] = git_sha(paths.root)
    rows["cutoff_ist"] = f"{asof.date()} {cfg.calendar.information_cutoff} {cfg.calendar.timezone}"
    rows["written_at"] = utc_now_iso()
    for c in OUTCOME_COLUMNS:
        rows[c] = np.nan
    rows = rows[[*PREDICTION_COLUMNS, *OUTCOME_COLUMNS]]

    existing = read_log(paths)
    if len(existing):
        same_day = pd.to_datetime(existing["date"]) == asof
        # Re-running the same evening is fine. Rewriting a prediction whose outcome is already known
        # is not: it would let the log be re-rolled until it looked good, which is the one thing a
        # forward log exists to prevent.
        if same_day.any() and existing.loc[same_day, "fwd_c2c"].notna().any():
            raise ValueError(
                f"{asof.date()} already has predictions with realised outcomes. The forward log is "
                "append-only once an outcome is known; it cannot be re-rolled."
            )
        existing = existing[~same_day]
    combined = pd.concat([existing, rows], ignore_index=True)
    combined, backfilled = backfill_outcomes(combined, cfg, store)
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    out = log_path(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)

    picks = rows.loc[rows["selected"], ["symbol", "rank", "score", "expected_return_bps"]]
    line = summary_line(combined, asof, picks)
    append_summary(paths, line)
    return ForwardResult(
        asof=asof,
        n_rows=len(rows),
        n_selected=int(rows["selected"].sum()),
        backfilled_dates=backfilled,
        log_path=out,
        summary_line=line,
        top_picks=picks,
    )


def backfill_outcomes(
    frame: pd.DataFrame, cfg: AppConfig, store: ParquetBarStore
) -> tuple[pd.DataFrame, list[str]]:
    """Fill realised outcomes for every logged date whose forward return is now observable."""
    if not len(frame):
        return frame, []
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    pending = frame["fwd_c2c"].isna()
    if not pending.any():
        return frame, []
    symbols = sorted(frame["symbol"].unique().tolist())
    far = pd.Timestamp(frame["date"].max()) + pd.Timedelta(days=30)
    fwd = forward_returns(
        store,
        symbols,
        cutoff=far,
        horizon_days=cfg.base.labels.horizon_days,
        benchmark=cfg.base.data.benchmark,
    )
    idx = pd.MultiIndex.from_arrays([frame["date"], frame["symbol"]], names=["date", "symbol"])
    for col in ("fwd_c2c", "fwd_c2o", "fwd_o2c"):
        vals = fwd[col].reindex(idx).to_numpy()
        frame.loc[pending, col] = vals[pending.to_numpy()]
    mean_by_date = frame.groupby("date")["fwd_c2c"].transform("mean")
    frame.loc[pending, "excess_vs_universe"] = (frame["fwd_c2c"] - mean_by_date)[pending]
    frame.loc[pending, "direction_correct"] = ((frame["score"].rank() > 0) & (frame["fwd_c2c"] > 0))[
        pending
    ].astype(float)
    frame.loc[pending, "beat_universe_mean"] = (frame["fwd_c2c"] > mean_by_date)[pending].astype(float)
    frame = _simulate_pnl(frame, cfg, pending)
    filled = frame.loc[pending & frame["fwd_c2c"].notna(), "date"]
    return frame, sorted({str(pd.Timestamp(d).date()) for d in filled})


def _simulate_pnl(frame: pd.DataFrame, cfg: AppConfig, pending: pd.Series) -> pd.DataFrame:
    """Gross and net P&L the selected names would have produced at the configured position size."""
    cm = CostModel(
        cfg.charges,
        product=Product(cfg.base.simulation.product),
        cost_multiplier=cfg.base.simulation.cost_multiplier,
    )
    size = cfg.base.simulation.position_size
    rt = cm.round_trip(size, slippage_bps_per_leg=PLANNING_SLIPPAGE_BPS).total
    sel = pending & frame["selected"].astype(bool) & frame["fwd_c2c"].notna()
    frame.loc[sel, "sim_gross_pnl"] = frame.loc[sel, "fwd_c2c"] * size
    frame.loc[sel, "sim_net_pnl"] = frame.loc[sel, "fwd_c2c"] * size - rt
    not_sel = pending & ~frame["selected"].astype(bool)
    frame.loc[not_sel, ["sim_gross_pnl", "sim_net_pnl"]] = 0.0
    return frame


def summary_line(frame: pd.DataFrame, asof: pd.Timestamp, picks: pd.DataFrame) -> str:
    scored = frame[frame["fwd_c2c"].notna()]
    n_days = int(scored["date"].nunique())
    ic = math.nan
    if n_days >= 2:
        from greenclose.evaluation.metrics import daily_ic

        s = scored.set_index(
            [pd.to_datetime(scored["date"]).rename("date"), scored["symbol"].rename("symbol")]
        )
        ic = float(daily_ic(s["score"], s["fwd_c2c"]).mean())
    net = float(scored["sim_net_pnl"].sum(skipna=True)) if "sim_net_pnl" in scored else 0.0
    names = ", ".join(picks["symbol"].tolist()) or "none (all abstained)"
    status = (
        f"integrity check only: {n_days} scored day(s) of the ~{IC_DAYS_FOR_T2} needed for a rank IC of "
        f"0.03 at t=2, and ~{PNL_DAYS_FOR_T2} for a 0.10%/day P&L edge. Not evidence of an edge."
    )
    ic_txt = f"{ic:+.4f}" if ic == ic else "n/a"
    return (
        f"- **{asof.date()}** picks: {names} | forward-log IC to date {ic_txt} over {n_days} scored days | "
        f"simulated net P&L to date Rs {net:,.0f} | {status}"
    )


def append_summary(paths: Paths, line: str) -> Path:
    p = paths.research_dir / "forward_log_summary.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(
            "# GREENCLOSE forward prediction log\n\n"
            "Predictions only. No broker, no orders, no money. One line per run.\n\n"
            "Every line is an integrity check, not evidence: a short forward log cannot establish an "
            "edge, and the power note on each line says how far it is from being able to.\n\n",
            encoding="utf-8",
        )
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return p


def forward_metrics(paths: Paths) -> dict[str, Any]:
    """Whatever the forward log can honestly support so far."""
    frame = read_log(paths)
    scored = frame[frame["fwd_c2c"].notna()] if len(frame) else frame
    if not len(scored):
        return {"n_days": 0, "note": "no scored days yet"}
    from greenclose.evaluation.metrics import daily_ic
    from greenclose.evaluation.significance import newey_west

    s = scored.set_index([pd.to_datetime(scored["date"]).rename("date"), scored["symbol"].rename("symbol")])
    ic = daily_ic(s["score"], s["fwd_c2c"])
    nw = newey_west(ic) if len(ic) >= 5 else {"t": math.nan}
    return {
        "n_days": int(scored["date"].nunique()),
        "n_rows": int(len(scored)),
        "ic_mean": float(ic.mean()) if len(ic) else math.nan,
        "ic_t_nw": nw["t"],
        "sim_net_pnl": float(scored["sim_net_pnl"].sum(skipna=True)),
        "sim_gross_pnl": float(scored["sim_gross_pnl"].sum(skipna=True)),
        "days_needed_for_ic": IC_DAYS_FOR_T2,
        "days_needed_for_pnl": PNL_DAYS_FOR_T2,
        "note": "a short forward log is an integrity check, not evidence of an edge",
    }


def _unused(_: date) -> None:  # pragma: no cover
    return None
