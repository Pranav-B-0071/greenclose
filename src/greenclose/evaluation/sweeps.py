"""Required sweeps (brief section 12.2).

Two families, split by what they cost:

*Decision-layer sweeps* change only what happens after the model has spoken, so they replay cached
predictions through the engine and cost nothing to retrain: hold period, K, cost multiplier, product
(CNC vs MIS), and the overlap-aware versus fixed-hold rebalance rule.

*Retrain sweeps* change the label or the training window, so the model must be refitted: the C2C /
C2O / O2C horizon split (Lou-Polk-Skouras) and expanding versus 2-year rolling windows.

Every sweep cell is one out-of-sample configuration and is appended to research_log, because the
Deflated Sharpe Ratio reads its trial count from that file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from greenclose.backtest.costs import CostModel
from greenclose.backtest.engine import DayPanel, FrameScorer, RunSpec, run_backtest
from greenclose.config import AppConfig, Paths
from greenclose.domain.models import Product
from greenclose.evaluation.baselines import (
    PLANNING_SLIPPAGE_BPS,
    RunEvaluation,
    default_decision,
    evaluate_run,
    log_trial,
)
from greenclose.ops.provenance import Provenance
from greenclose.ops.research_log import ResearchLog
from greenclose.strategy.decide import DecisionConfig

HOLD_PERIODS = (1, 3, 5, 10)
K_VALUES = (1, 3, 5, 10)
COST_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
PRODUCTS = ("CNC", "MIS")
REBALANCE_MODES = ("overlap", "fixed")


@dataclass
class SweepCell:
    group: str
    label: str
    decision: DecisionConfig
    product: Product
    cost_multiplier: float
    extra: dict[str, Any] = field(default_factory=dict)


def _round_trip_bps(cfg: AppConfig, product: Product, multiplier: float) -> float:
    cm = CostModel(cfg.charges, product=product, cost_multiplier=multiplier)
    return cm.round_trip_bps(cfg.base.simulation.position_size, slippage_bps_per_leg=PLANNING_SLIPPAGE_BPS)


def decision_cells(cfg: AppConfig) -> list[SweepCell]:
    """Every decision-layer cell. The abstention threshold follows the cell's own cost model."""
    base = default_decision(cfg)
    product = Product(cfg.base.simulation.product)
    cells: list[SweepCell] = []
    for h in HOLD_PERIODS:
        cells.append(
            SweepCell(
                "hold", f"hold={h}d", base.with_(hold_days=h), product, cfg.base.simulation.cost_multiplier
            )
        )
    for k in K_VALUES:
        cells.append(SweepCell("k", f"K={k}", base.with_(k=k), product, cfg.base.simulation.cost_multiplier))
    for m in COST_MULTIPLIERS:
        dec = base.with_(round_trip_cost_bps=_round_trip_bps(cfg, product, m))
        cells.append(SweepCell("cost_multiplier", f"costs x{m:g}", dec, product, m))
    for p in PRODUCTS:
        prod = Product(p)
        dec = base.with_(
            product=prod, round_trip_cost_bps=_round_trip_bps(cfg, prod, cfg.base.simulation.cost_multiplier)
        )
        cells.append(SweepCell("product", f"product={p}", dec, prod, cfg.base.simulation.cost_multiplier))
    for mode in REBALANCE_MODES:
        cells.append(
            SweepCell(
                "rebalance",
                f"rebalance={mode}",
                base.with_(rebalance_mode=mode),
                product,
                cfg.base.simulation.cost_multiplier,
            )
        )
    return cells


def run_decision_sweep(
    predictions: pd.DataFrame,
    *,
    cfg: AppConfig,
    paths: Paths,
    panel: DayPanel,
    labels: pd.DataFrame,
    dates: pd.DatetimeIndex,
    nifty_ret: pd.Series,
    snapshot_id: str,
    model_name: str,
    seed: int,
    log: ResearchLog | None,
    n_boot: int = 2000,
    groups: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Replay one prediction frame through every decision-layer cell. No model is refitted."""
    rows: list[dict[str, Any]] = []
    for cell in decision_cells(cfg):
        if groups is not None and cell.group not in groups:
            continue
        dec = cell.decision
        if "expected_return_bps" not in predictions.columns:
            dec = dec.with_(abstain_enabled=False)
        spec = RunSpec(
            name=f"sweep_{cell.group}_{cell.label}_{model_name}_s{seed}",
            scorer=FrameScorer(predictions),
            decision=dec,
            dates=dates,
            description=f"sweep {cell.group}: {cell.label} on {model_name}",
            family="sweep",
            product=cell.product,
            cost_multiplier=cell.cost_multiplier,
            slippage_override_bps=cfg.base.simulation.slippage_override_bps,
            capital=cfg.base.simulation.capital,
            seed=seed,
        )
        prov = Provenance.create(config_hash=cfg.hash(), snapshot_id=snapshot_id, seed=seed, label="sweep")
        result = run_backtest(spec, panel, cfg, provenance=prov)
        ev = evaluate_run(
            result, labels, capital=cfg.base.simulation.capital, nifty_ret=nifty_ret, n_boot=n_boot
        )
        if log is not None:
            log_trial(log, result, ev, family="sweep")
        rows.append(
            {
                "group": cell.group,
                "cell": cell.label,
                "model": model_name,
                "seed": seed,
                "ic_mean": ev.family_a.get("ic_mean", np.nan),
                "net_pa": ev.family_b.get("net_return_pa", np.nan),
                "net_ci_lo": ev.family_b.get("net_pa_ci_lo", np.nan),
                "net_ci_hi": ev.family_b.get("net_pa_ci_hi", np.nan),
                "gross_pa": ev.family_b.get("gross_return_pa", np.nan),
                "sharpe": ev.family_b.get("sharpe", np.nan),
                "net_bps_inv": ev.family_b.get("net_bps_per_day_invested", np.nan),
                "gross_bps_inv": ev.family_b.get("gross_bps_per_day_invested", np.nan),
                "cost_bps_inv": ev.family_b.get("cost_bps_per_day_invested", np.nan),
                "costs_over_gross": ev.family_b.get("costs_over_gross_pnl", np.nan),
                "turnover_pa": ev.family_b.get("turnover_pa", np.nan),
                "n_trades": ev.family_b.get("n_trades", np.nan),
                "abstain_rate": ev.family_b.get("abstain_rate", np.nan),
                "max_dd": ev.family_b.get("max_drawdown", np.nan),
            }
        )
    return pd.DataFrame(rows)


def subperiod_tables(ev: RunEvaluation, panel: DayPanel) -> dict[str, pd.DataFrame]:
    """IC and net return per calendar year and per volatility tercile (brief 12.3)."""
    from greenclose.evaluation import performance

    out: dict[str, pd.DataFrame] = {}
    r = ev.daily_ret
    if len(r):
        out["per_year_return"] = performance.per_year(r)
    if len(ev.per_date):
        ic = ev.per_date["ic"].dropna()
        by_year = ic.groupby(ic.index.year).agg(["mean", "std", "size"])
        by_year.columns = ["ic_mean", "ic_std", "n_days"]
        by_year["icir"] = by_year["ic_mean"] / by_year["ic_std"]
        out["per_year_ic"] = by_year.reset_index(names="year")
        vol = pd.Series(panel.high_vol).astype(int)
        tercile = pd.qcut(
            pd.Series(
                {
                    d: float(panel.by_date[d]["nifty_vol_20d"].iloc[0])
                    for d in panel.dates
                    if d in panel.by_date
                }
            ).rank(method="first"),
            3,
            labels=["low_vol", "mid_vol", "high_vol"],
        ).astype(str)
        joined = pd.DataFrame({"ic": ic, "tercile": tercile.reindex(ic.index)}).dropna()
        if len(joined):
            g = joined.groupby("tercile")["ic"].agg(["mean", "std", "size"])
            g.columns = ["ic_mean", "ic_std", "n_days"]
            out["per_vol_tercile_ic"] = g.reset_index()
        if len(r):
            out["per_vol_tercile_return"] = performance.per_regime(r, tercile, "vol_tercile")
        del vol
    return out


def format_sweep(df: pd.DataFrame) -> str:
    fmt = {
        "ic_mean": "{:+.4f}",
        "net_pa": "{:+.1%}",
        "net_ci_lo": "{:+.1%}",
        "net_ci_hi": "{:+.1%}",
        "gross_pa": "{:+.1%}",
        "sharpe": "{:+.2f}",
        "net_bps_inv": "{:+.2f}",
        "gross_bps_inv": "{:+.2f}",
        "cost_bps_inv": "{:.2f}",
        "turnover_pa": "{:.1f}",
        "abstain_rate": "{:.0%}",
        "max_dd": "{:.0%}",
        "n_trades": "{:.0f}",
    }

    def _f(v: object, f: str) -> str:
        if not isinstance(v, float) or np.isnan(v):
            return ""
        return "inf" if np.isinf(v) else f.format(v)

    cols = ["group", "cell", *[c for c in fmt if c in df.columns]]
    return df[cols].to_string(
        index=False, formatters={k: (lambda v, f=fmt[k]: _f(v, f)) for k in fmt if k in df.columns}
    )


# ----------------------------------------------------------------------------- retrain sweeps

HORIZONS = ("c2c", "c2o", "o2c")
WINDOWS = (("expanding", None), ("rolling_2y", 2))


def run_retrain_sweep(
    *,
    cfg: AppConfig,
    paths: Paths,
    store: Any,
    features: pd.DataFrame,
    panel: DayPanel,
    dates: pd.DatetimeIndex,
    nifty_ret: pd.Series,
    snapshot_id: str,
    head_kind: str,
    model_columns: list[str],
    seed: int,
    log: ResearchLog | None,
    n_boot: int = 2000,
    horizons: tuple[str, ...] = HORIZONS,
    windows: tuple[tuple[str, int | None], ...] = WINDOWS,
    hold_days: tuple[int, ...] = (1,),
) -> pd.DataFrame:
    """Refit the model per (label horizon, training window, label horizon length) and evaluate.

    The horizon split is the Lou-Polk-Skouras test: momentum is documented to accrue overnight and
    reversal intraday, so close-to-close can blend a predictable component with an opposing one.
    """
    import json

    from greenclose.evaluation.ladder import build_head, evaluate_predictions, ladder_rungs, schedule_path
    from greenclose.labels.targets import build_labels
    from greenclose.models.train import walk_forward_predict

    # Reuse the hyperparameter choice the ladder already made rather than re-running the 72-cell grid
    # for every horizon and window. This is a compute decision, stated here and in the report: the
    # sweep asks what the *label* and the *window* do, holding the model's own settings fixed.
    schedule = None
    rung = next((r for r in ladder_rungs() if r.head_kind == head_kind), None)
    if rung is not None:
        f = schedule_path(paths, rung, cfg, snapshot_id)
        if f.exists():
            schedule = json.loads(f.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        for h_days in hold_days:
            labels = build_labels(features, store, cfg, horizon_days=h_days, base=horizon).frame
            for window_name, rolling in windows:
                run_cfg = cfg.override({"base.walk_forward.rolling_years": rolling})
                run = walk_forward_predict(
                    features,
                    labels,
                    model_columns=model_columns,
                    head=build_head(head_kind, run_cfg),
                    cfg=run_cfg,
                    oos_dates=dates,
                    seed=seed,
                    hparam_schedule=schedule,
                )
                label = f"{horizon}/{window_name}" + (f"/h={h_days}d" if len(hold_days) > 1 else "")
                ev = evaluate_predictions(
                    f"retrain_{horizon}_{window_name}_h{h_days}_s{seed}",
                    run.predictions,
                    cfg=run_cfg,
                    paths=paths,
                    panel=panel,
                    labels=labels,
                    dates=dates,
                    nifty_ret=nifty_ret,
                    snapshot_id=snapshot_id,
                    seed=seed,
                    description=f"retrain sweep {label} on {head_kind}",
                    n_boot=n_boot,
                    log_to=log,
                    persist=False,
                )
                rows.append(
                    {
                        "group": "retrain",
                        "cell": label,
                        "model": head_kind,
                        "seed": seed,
                        "horizon": horizon,
                        "window": window_name,
                        "label_days": h_days,
                        "ic_mean": ev.family_a.get("ic_mean", np.nan),
                        "ic_t_nw": ev.family_a.get("ic_t_nw", np.nan),
                        "net_pa": ev.family_b.get("net_return_pa", np.nan),
                        "gross_pa": ev.family_b.get("gross_return_pa", np.nan),
                        "sharpe": ev.family_b.get("sharpe", np.nan),
                        "net_bps_inv": ev.family_b.get("net_bps_per_day_invested", np.nan),
                        "gross_bps_inv": ev.family_b.get("gross_bps_per_day_invested", np.nan),
                        "turnover_pa": ev.family_b.get("turnover_pa", np.nan),
                        "n_trades": ev.family_b.get("n_trades", np.nan),
                    }
                )
    return pd.DataFrame(rows)
