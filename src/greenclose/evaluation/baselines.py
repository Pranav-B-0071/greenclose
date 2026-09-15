"""Baselines B0-B5 (brief section 12.1) on identical dates, universe and cost model, plus the shared
run-evaluation and persistence helpers every later phase reuses.

    B0  uniform random top-K, N simulations     -> the null distribution; beat its 95th percentile
    B1  buy-and-hold NIFTY                       -> active management adds anything after costs?
    B2  equal-weight universe, monthly rebalance -> does *selection* add value? (the toughest)
    B3  always-long the highest-beta name        -> not merely levered market exposure?
    B4  top-K by 20-day return (momentum)
    B5  bottom-K by 1-day return (reversal)      -> the effect a daily model most likely rediscovers
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from greenclose.backtest.costs import CostModel
from greenclose.backtest.engine import (
    ConstantScorer,
    DayPanel,
    FeatureScorer,
    RandomScorer,
    RunResult,
    RunSpec,
    decision_config_from,
    run_backtest,
)
from greenclose.config import AppConfig, Paths
from greenclose.domain.models import Product
from greenclose.evaluation import metrics, performance
from greenclose.evaluation.significance import newey_west, stationary_bootstrap_ci
from greenclose.ops.provenance import Provenance
from greenclose.ops.research_log import ResearchLog, Trial
from greenclose.strategy.decide import DecisionConfig

PLANNING_SLIPPAGE_BPS = 5.0  # per leg, the brief's reference assumption for the abstention threshold


def planning_round_trip_bps(cfg: AppConfig) -> float:
    cm = CostModel(
        cfg.charges,
        product=Product(cfg.base.simulation.product),
        cost_multiplier=cfg.base.simulation.cost_multiplier,
    )
    return cm.round_trip_bps(cfg.base.simulation.position_size, slippage_bps_per_leg=PLANNING_SLIPPAGE_BPS)


def default_decision(cfg: AppConfig) -> DecisionConfig:
    return decision_config_from(cfg, round_trip_cost_bps=planning_round_trip_bps(cfg))


def baseline_specs(cfg: AppConfig, dates: pd.DatetimeIndex, *, n_universe: int) -> list[RunSpec]:
    """B2..B5. B0 is produced by `run_b0_distribution`; B1 needs no engine run."""
    dec = default_decision(cfg).with_(abstain_enabled=False)  # baselines have no return-magnitude head
    common = dict(
        dates=dates,
        product=Product(cfg.base.simulation.product),
        cost_multiplier=cfg.base.simulation.cost_multiplier,
        slippage_override_bps=cfg.base.simulation.slippage_override_bps,
        capital=cfg.base.simulation.capital,
    )
    specs: list[RunSpec] = []
    specs.append(
        RunSpec(
            name="B2_equal_weight",
            scorer=ConstantScorer(),
            decision=dec.with_(
                k=n_universe,
                sizing="equal",
                sector_max=0,
                cluster_max=0,
                rebalance_frequency="monthly",
                rebalance_band=0.05,
                min_position=0.0,
            ),
            description="B2 equal-weight universe, monthly rebalance",
            **common,  # type: ignore[arg-type]
        )
    )
    specs.append(
        RunSpec(
            name="B3_highest_beta",
            scorer=FeatureScorer("beta_60d"),
            decision=dec.with_(k=1, sector_max=0, cluster_max=0),
            description="B3 always-long the highest-beta name",
            **common,  # type: ignore[arg-type]
        )
    )
    specs.append(
        RunSpec(
            name="B4_momentum_20d",
            scorer=FeatureScorer("ret_20d"),
            decision=dec,
            description="B4 top-K by 20-day return",
            **common,
        )
    )  # type: ignore[arg-type]
    specs.append(
        RunSpec(
            name="B5_reversal_1d",
            scorer=FeatureScorer("ret_1d"),
            decision=dec.with_(select_from="bottom"),
            description="B5 bottom-K by 1-day return",
            **common,
        )  # type: ignore[arg-type]
    )
    return specs


# ----------------------------------------------------------------------------- evaluation


@dataclass
class RunEvaluation:
    name: str
    family_b: dict[str, float]
    family_a: dict[str, float]
    per_date: pd.DataFrame
    daily_ret: pd.Series
    extra: dict[str, float] = field(default_factory=dict)

    def row(self) -> dict[str, float | str]:
        a, b = self.family_a, self.family_b
        return {
            "name": self.name,
            "ic_mean": a.get("ic_mean", math.nan),
            "ic_t_nw": a.get("ic_t_nw", math.nan),
            "ic_ci_lo": a.get("ic_ci_lo", math.nan),
            "ic_ci_hi": a.get("ic_ci_hi", math.nan),
            "lift_at_3": a.get("lift_at_3_mean", math.nan),
            "decile_spread_bps": a.get("decile_spread_mean", math.nan) * 1e4
            if not math.isnan(a.get("decile_spread_mean", math.nan))
            else math.nan,
            "gross_pa": b.get("gross_return_pa", math.nan),
            "net_pa": b.get("net_return_pa", math.nan),
            "net_ci_lo": b.get("net_pa_ci_lo", math.nan),
            "net_ci_hi": b.get("net_pa_ci_hi", math.nan),
            "sharpe": b.get("sharpe", math.nan),
            "max_dd": b.get("max_drawdown", math.nan),
            "costs_over_gross": b.get("costs_over_gross_pnl", math.nan),
            "turnover_pa": b.get("turnover_pa", math.nan),
            "n_trades": b.get("n_trades", math.nan),
            "missed_rate": b.get("missed_trade_rate", math.nan),
            "n_days": b.get("n_days", math.nan),
            "gross_bps_inv": b.get("gross_bps_per_day_invested", math.nan),
            "net_bps_inv": b.get("net_bps_per_day_invested", math.nan),
            "deployment": b.get("mean_deployment", math.nan),
        }


def evaluate_run(
    result: RunResult,
    labels: pd.DataFrame,
    *,
    capital: float,
    nifty_ret: pd.Series | None = None,
    realised_col: str = "fwd_c2c",
    n_boot: int = 10_000,
    seed: int = 0,
) -> RunEvaluation:
    fam_b = performance.summarise(result.daily, capital, benchmark_ret=nifty_ret)
    r = performance.daily_returns(result.daily, capital)
    if n_boot > 0 and len(r) >= 10:
        lo, hi = stationary_bootstrap_ci(r.to_numpy(), n_boot=n_boot, seed=seed)
        fam_b["net_pa_ci_lo"], fam_b["net_pa_ci_hi"] = (
            lo * performance.TRADING_DAYS,
            hi * performance.TRADING_DAYS,
        )
    fam_a: dict[str, float] = {}
    per_date = pd.DataFrame()
    if len(result.replay) and "score" in result.replay.columns:
        pred = result.replay["score"]
        if pred.notna().any() and pred.groupby(level="date").nunique().max() > 1:
            realised = labels[realised_col].reindex(pred.index)
            fam_a, per_date = metrics.family_a(
                pred,
                realised,
                expected_return=result.replay["expected_return_bps"] / 1e4
                if "expected_return_bps" in result.replay.columns
                else None,
                prob_beat=result.replay["prob_beat"] if "prob_beat" in result.replay.columns else None,
                prob_green=result.replay["prob_green"] if "prob_green" in result.replay.columns else None,
            )
            nw = newey_west(per_date["ic"])
            fam_a["ic_t_nw"], fam_a["ic_se_nw"] = nw["t"], nw["se"]
            if n_boot > 0:
                fam_a["ic_ci_lo"], fam_a["ic_ci_hi"] = stationary_bootstrap_ci(
                    per_date["ic"].to_numpy(), n_boot=n_boot, seed=seed
                )
                fam_a["lift_at_3_ci_lo"], fam_a["lift_at_3_ci_hi"] = stationary_bootstrap_ci(
                    per_date["lift_at_3"].to_numpy(), n_boot=n_boot, seed=seed
                )
    return RunEvaluation(name=result.name, family_b=fam_b, family_a=fam_a, per_date=per_date, daily_ret=r)


def persist_run(result: RunResult, evaluation: RunEvaluation, paths: Paths) -> Path:
    run_id = result.provenance.run_id if result.provenance else result.name
    out = paths.data_dir / "runs" / f"{result.name}__{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    result.daily.to_parquet(out / "daily.parquet", index=False)
    if len(result.replay):
        result.replay.to_parquet(out / "replay.parquet")
    if len(result.fills):
        result.fills.to_parquet(out / "fills.parquet", index=False)
    if len(evaluation.per_date):
        evaluation.per_date.to_parquet(out / "per_date.parquet")
    summary = {
        "name": result.name,
        "description": result.spec.description,
        "family": result.spec.family,
        "provenance": result.provenance.as_dict() if result.provenance else {},
        "decision": asdict(result.spec.decision) | {"product": result.spec.decision.product.value},
        "family_a": evaluation.family_a,
        "family_b": evaluation.family_b,
        "extra": evaluation.extra,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return out


def log_trial(
    log: ResearchLog,
    result: RunResult,
    evaluation: RunEvaluation,
    *,
    counts_for_dsr: bool = True,
    family: str | None = None,
) -> int:
    prov = result.provenance
    return log.append(
        Trial(
            description=f"{result.name}: {result.spec.description}",
            family=family or result.spec.family,
            config_hash=prov.config_hash if prov else "",
            snapshot_id=prov.snapshot_id if prov else "",
            git_sha=prov.git_sha if prov else "",
            seed=result.spec.seed,
            oos_ic=evaluation.family_a.get("ic_mean"),
            oos_sharpe=evaluation.family_b.get("sharpe_per_period"),
            oos_net_return_pa=evaluation.family_b.get("net_return_pa"),
            n_days=int(evaluation.family_b.get("n_days", 0)),
            counts_for_dsr=counts_for_dsr,
            extra={"run_id": prov.run_id if prov else ""},
        )
    )


# ----------------------------------------------------------------------------- B0 in parallel

_WORKER: dict[str, object] = {}


def _b0_init(snapshot_id: str, root: str, capital: float) -> None:
    """Each worker rebuilds the day panel once from the cached parquet (Windows uses spawn)."""
    from greenclose.config import Paths as _Paths
    from greenclose.config import load_config as _load
    from greenclose.pipeline_io import load_features

    paths = _Paths(root=Path(root))
    cfg = _load(paths)
    feats = load_features(paths, snapshot_id)
    _WORKER["cfg"] = cfg
    _WORKER["panel"] = DayPanel.from_features(feats)
    _WORKER["labels"] = None
    _WORKER["capital"] = capital


def _b0_one(args: tuple[int, list[pd.Timestamp], dict[str, object]]) -> dict[str, float]:
    """One uniform-random simulation through the identical engine and decision layer."""
    seed, dates, dec_kwargs = args
    cfg = _WORKER["cfg"]
    panel = _WORKER["panel"]
    dec = DecisionConfig(**dec_kwargs)  # type: ignore[arg-type]
    spec = RunSpec(
        name=f"B0_random_{seed:04d}",
        scorer=RandomScorer(seed),
        decision=dec,
        dates=pd.DatetimeIndex(dates),
        description="B0 uniform random top-K",
        seed=seed,
        product=dec.product,
        cost_multiplier=cfg.base.simulation.cost_multiplier,  # type: ignore[union-attr]
        slippage_override_bps=cfg.base.simulation.slippage_override_bps,  # type: ignore[union-attr]
        capital=float(_WORKER["capital"]),  # type: ignore[arg-type]
    )
    result = run_backtest(spec, panel, cfg)  # type: ignore[arg-type]
    fam_b = performance.summarise(result.daily, float(_WORKER["capital"]))  # type: ignore[arg-type]
    row = {"sim": float(seed)}
    for k in (
        "net_return_pa",
        "gross_return_pa",
        "sharpe",
        "max_drawdown",
        "costs_over_gross",
        "net_bps_per_day_invested",
        "gross_bps_per_day_invested",
        "mean_deployment",
    ):
        row[k] = float(fam_b.get(k, math.nan))
    if len(result.replay) and "score" in result.replay.columns:
        pred = result.replay["score"]
        realised = _WORKER["realised"].reindex(pred.index)  # type: ignore[union-attr]
        ic = metrics.daily_ic(pred, realised)
        row["ic_mean"] = float(ic.mean())
        row["lift_at_3"] = float(metrics.lift_at_k(pred, realised, 3).mean())
        row["p_at_3_beat"] = float(metrics.precision_at_k(pred, realised, 3, "beat").mean())
    return row


def run_b0_distribution(
    cfg: AppConfig,
    paths: Paths,
    *,
    snapshot_id: str,
    dates: pd.DatetimeIndex,
    labels: pd.DataFrame,
    decision: DecisionConfig,
    n_sims: int,
    jobs: int = 1,
    realised_col: str = "fwd_c2c",
) -> pd.DataFrame:
    """B0: `n_sims` uniform-random top-K runs through the identical engine. Beat its 95th percentile."""
    dec_kwargs = {k: v for k, v in asdict(decision).items()}
    dec_kwargs["product"] = decision.product
    args = [(s, list(dates), dec_kwargs) for s in range(n_sims)]
    realised = labels[realised_col]
    if jobs <= 1:
        _b0_init(snapshot_id, str(paths.root), cfg.base.simulation.capital)
        _WORKER["realised"] = realised
        rows = [_b0_one(a) for a in args]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=jobs,
            initializer=_b0_init_with_labels,
            initargs=(snapshot_id, str(paths.root), cfg.base.simulation.capital, realised),
        ) as pool:
            rows = pool.map(_b0_one, args, chunksize=1)
    return pd.DataFrame(rows)


def _b0_init_with_labels(snapshot_id: str, root: str, capital: float, realised: pd.Series) -> None:
    _b0_init(snapshot_id, root, capital)
    _WORKER["realised"] = realised


# ----------------------------------------------------------------------------- driver


def run_baselines(
    cfg: AppConfig,
    paths: Paths,
    *,
    panel: DayPanel,
    labels: pd.DataFrame,
    nifty_close: pd.Series,
    snapshot_id: str,
    b0_sims: int,
    n_boot: int,
    log: ResearchLog,
    n_universe: int,
    jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run B0..B5 on the OOS window. Returns (summary table, B0 per-simulation distribution)."""
    dates = panel.oos_dates(
        min_train_years=cfg.base.walk_forward.min_train_years,
        start=cfg.base.data.start,
        holdout_start=cfg.base.holdout.start,
    )
    capital = cfg.base.simulation.capital
    nifty_ret = (nifty_close / nifty_close.shift(1) - 1.0).reindex(dates)
    rows: list[dict[str, float | str]] = []
    b0_rows: list[dict[str, float]] = []

    # B1: index proxy, no trades, no costs
    b1_daily = performance.nifty_buy_and_hold(nifty_close, dates, capital)
    b1 = RunResult(
        spec=RunSpec(
            name="B1_nifty_buy_hold",
            scorer=ConstantScorer(),
            decision=default_decision(cfg),
            dates=dates,
            description="B1 buy-and-hold NIFTY (no costs; index proxy)",
        ),
        daily=b1_daily,
        replay=pd.DataFrame(),
        fills=pd.DataFrame(),
        provenance=Provenance.create(config_hash=cfg.hash(), snapshot_id=snapshot_id, seed=0, label="B1"),
    )
    ev1 = evaluate_run(b1, labels, capital=capital, nifty_ret=nifty_ret, n_boot=n_boot)
    persist_run(b1, ev1, paths)
    log_trial(log, b1, ev1)
    rows.append(ev1.row())

    for spec in baseline_specs(cfg, dates, n_universe=n_universe):
        spec.family = "baseline"
        prov = Provenance.create(
            config_hash=cfg.hash(), snapshot_id=snapshot_id, seed=spec.seed, label=spec.name.split("_")[0]
        )
        result = run_backtest(spec, panel, cfg, provenance=prov)
        ev = evaluate_run(result, labels, capital=capital, nifty_ret=nifty_ret, n_boot=n_boot)
        persist_run(result, ev, paths)
        log_trial(log, result, ev)
        rows.append(ev.row())

    b0 = pd.DataFrame(b0_rows)
    if b0_sims > 0:
        b0 = run_b0_distribution(
            cfg,
            paths,
            snapshot_id=snapshot_id,
            dates=dates,
            labels=labels,
            decision=default_decision(cfg).with_(abstain_enabled=False),
            n_sims=b0_sims,
            jobs=jobs,
        )
        b0 = b0.rename(columns={"net_return_pa": "net_pa", "gross_return_pa": "gross_pa"})
    if len(b0):
        q = b0[["ic_mean", "gross_pa", "net_pa", "sharpe", "lift_at_3"]].quantile([0.05, 0.5, 0.95])
        for name, r in (
            ("B0_random_p05", q.iloc[0]),
            ("B0_random_p50", q.iloc[1]),
            ("B0_random_p95", q.iloc[2]),
        ):
            rows.append(
                {
                    "name": f"{name} (n={len(b0)})",
                    "ic_mean": float(r["ic_mean"]),
                    "lift_at_3": float(r["lift_at_3"]),
                    "gross_pa": float(r["gross_pa"]),
                    "net_pa": float(r["net_pa"]),
                    "sharpe": float(r["sharpe"]),
                    "n_days": float(len(dates)),
                }
            )
        out = paths.data_dir / "runs"
        out.mkdir(parents=True, exist_ok=True)
        b0.to_parquet(out / f"B0_distribution__{snapshot_id}__n{len(b0)}.parquet", index=False)
        log.append(
            Trial(
                description=f"B0 random top-K, {len(b0)} sims (null distribution; not a tried config)",
                family="baseline",
                config_hash=cfg.hash(),
                snapshot_id=snapshot_id,
                git_sha=b1.provenance.git_sha if b1.provenance else "",
                seed=0,
                oos_ic=float(b0["ic_mean"].mean()),
                oos_sharpe=float(b0["sharpe"].mean() / math.sqrt(performance.TRADING_DAYS)),
                oos_net_return_pa=float(b0["net_pa"].mean()),
                n_days=len(dates),
                counts_for_dsr=False,
                extra={
                    "n_sims": len(b0),
                    "net_pa_p95": float(b0["net_pa"].quantile(0.95)),
                    "ic_p95": float(b0["ic_mean"].quantile(0.95)),
                },
            )
        )
    table = pd.DataFrame(rows).set_index("name")
    return table, b0


def format_table(table: pd.DataFrame) -> str:
    cols = [
        "ic_mean",
        "ic_t_nw",
        "lift_at_3",
        "decile_spread_bps",
        "gross_pa",
        "net_pa",
        "net_ci_lo",
        "net_ci_hi",
        "sharpe",
        "max_dd",
        "costs_over_gross",
        "turnover_pa",
        "gross_bps_inv",
        "net_bps_inv",
        "deployment",
    ]
    t = table.reindex(columns=cols)
    fmt = {
        "ic_mean": "{:+.4f}",
        "ic_t_nw": "{:+.2f}",
        "lift_at_3": "{:+.3f}",
        "decile_spread_bps": "{:+.1f}",
        "gross_pa": "{:+.1%}",
        "net_pa": "{:+.1%}",
        "net_ci_lo": "{:+.1%}",
        "net_ci_hi": "{:+.1%}",
        "sharpe": "{:+.2f}",
        "max_dd": "{:.1%}",
        "costs_over_gross": "{:.0%}",
        "gross_bps_inv": "{:+.2f}",
        "net_bps_inv": "{:+.2f}",
        "deployment": "{:.0%}",
        "turnover_pa": "{:.1f}",
        "n_trades": "{:.0f}",
    }

    def _fmt(v: object, f: str) -> str:
        if not isinstance(v, float) or np.isnan(v):
            return ""
        return "inf" if np.isinf(v) else f.format(v)

    return t.to_string(formatters={k: (lambda v, f=f: _fmt(v, f)) for k, f in fmt.items()})
