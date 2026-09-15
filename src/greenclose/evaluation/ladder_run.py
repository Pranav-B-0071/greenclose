"""Orchestration for `greenclose ladder`: fit every rung, evaluate, compare, and record.

Kept out of cli.py so the CLI stays a thin argument parser and this stays testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from greenclose.backtest.engine import DayPanel
from greenclose.config import AppConfig, Paths
from greenclose.evaluation.baselines import RunEvaluation
from greenclose.evaluation.ladder import (
    Rung,
    compare_rungs,
    dump_json,
    evaluate_predictions,
    fit_rungs,
    format_ladder,
    h2_rungs,
    h2_table,
    ladder_rungs,
    log_grid_trials,
    predictions_path,
    rank_average,
    seed_summary,
)
from greenclose.ops.provenance import git_sha
from greenclose.ops.research_log import ResearchLog
from greenclose.pipeline_io import load_features, load_labels, open_store

FAMILY_B_KEYS = {
    "net_pa": "net_return_pa",
    "gross_pa": "gross_return_pa",
    "sharpe": "sharpe",
    "net_bps_inv": "net_bps_per_day_invested",
    "gross_bps_inv": "gross_bps_per_day_invested",
    "costs_over_gross": "costs_over_gross_pnl",
}


@dataclass
class LadderOutput:
    table: pd.DataFrame
    comparisons: dict[str, dict[str, float | str]]
    h2: pd.DataFrame | None
    h2_comparisons: dict[str, dict[str, float | str]]
    lines: list[str]
    elapsed_s: float


def _summarise_rung(name: str, evals: list[RunEvaluation]) -> dict[str, Any]:
    row: dict[str, Any] = dict(evals[0].row())
    row["name"] = name
    ic = seed_summary(evals, "ic_mean")
    row["ic_mean"], row["ic_sd"] = ic["mean"], ic["std"]
    row["ic_t_nw"] = seed_summary(evals, "ic_t_nw")["mean"]
    row["lift_at_3"] = seed_summary(evals, "lift_at_3_mean")["mean"]
    row["lift_at_3_ci_lo"] = seed_summary(evals, "lift_at_3_ci_lo")["mean"]
    row["lift_at_3_ci_hi"] = seed_summary(evals, "lift_at_3_ci_hi")["mean"]
    row["ic_ci_lo"] = seed_summary(evals, "ic_ci_lo")["mean"]
    row["ic_ci_hi"] = seed_summary(evals, "ic_ci_hi")["mean"]
    row["decile_spread_bps"] = seed_summary(evals, "decile_spread_mean")["mean"] * 1e4
    for short, key in FAMILY_B_KEYS.items():
        row[short] = seed_summary(evals, key, "family_b")["mean"]
    row["n_seeds"] = float(len(evals))
    return row


def run_ladder(
    *,
    snapshot: str | None,
    rungs: str,
    seeds: str,
    jobs: int,
    n_boot: int,
    with_h2: bool,
) -> LadderOutput:
    cfg, paths = _load(), Paths()
    store = open_store(paths, snapshot)
    feats = load_features(paths, store.snapshot_id)
    labels_df = load_labels(paths, store.snapshot_id, cfg.base.labels.horizon_days, cfg.base.labels.horizon)
    panel = DayPanel.from_features(feats)
    dates = panel.oos_dates(
        min_train_years=cfg.base.walk_forward.min_train_years,
        start=cfg.base.data.start,
        holdout_start=cfg.base.holdout.start,
    )
    idx = store.get_index(cutoff=feats.index.get_level_values("date").max())
    nifty = idx.loc[idx["symbol"] == cfg.base.data.benchmark].set_index("date")["close"]
    nifty_ret = nifty.reindex(nifty.index.union(dates)).ffill().reindex(dates).pct_change().fillna(0.0)
    seed_list = [int(x) for x in seeds.split(",")] if seeds else list(cfg.base.seeds)
    lo, _, hi = rungs.partition("-")
    lo_n, hi_n = int(lo), int(hi or lo)
    wanted = [r for r in ladder_rungs() if lo_n <= r.number <= hi_n]
    log = ResearchLog(paths.research_dir / "research_log.jsonl")
    sha = git_sha(paths.root)
    lines: list[str] = [
        f"snapshot {store.snapshot_id}  OOS {dates[0].date()}..{dates[-1].date()} ({len(dates)} days)",
        f"rungs {[r.name for r in wanted]}  seeds {seed_list}  jobs {jobs}",
    ]

    t0 = time.perf_counter()
    meta = fit_rungs(wanted, seed_list, paths=paths, snapshot_id=store.snapshot_id, jobs=jobs)
    lines.append(f"fitted {len(meta)} (rung, seed) pairs in {time.perf_counter() - t0:.0f}s")
    for name, seed, _, n_grid, dt in meta:
        if n_grid or dt:
            lines.append(f"  {name:24s} seed {seed}  grid cells {n_grid:4d}  {dt:6.0f}s")

    by_rung: dict[str, list[RunEvaluation]] = {}
    rows: list[dict[str, Any]] = []
    for r in wanted:
        evals = [
            evaluate_predictions(
                f"R{r.number}_{r.name}_s{seed}",
                pd.read_parquet(predictions_path(paths, r, cfg, store.snapshot_id, seed)),
                cfg=cfg,
                paths=paths,
                panel=panel,
                labels=labels_df,
                dates=dates,
                nifty_ret=nifty_ret,
                snapshot_id=store.snapshot_id,
                seed=seed,
                description=r.description,
                n_boot=n_boot if seed == seed_list[0] else 0,
                log_to=log,
            )
            for seed in seed_list
        ]
        by_rung[r.name] = evals
        log_grid_trials(log, r, sum(m[3] for m in meta if m[0] == r.name), cfg, store.snapshot_id, sha)
        rows.append(_summarise_rung(f"R{r.number} {r.name}", evals))

    members = [r for r in wanted if 3 <= r.number <= 6]
    if hi_n >= 8 and len(members) >= 2:
        ens_evals = []
        for seed in seed_list:
            frames = [
                pd.read_parquet(predictions_path(paths, m, cfg, store.snapshot_id, seed)) for m in members
            ]
            ens_evals.append(
                evaluate_predictions(
                    f"R8_rank_ensemble_s{seed}",
                    rank_average(frames),
                    cfg=cfg,
                    paths=paths,
                    panel=panel,
                    labels=labels_df,
                    dates=dates,
                    nifty_ret=nifty_ret,
                    snapshot_id=store.snapshot_id,
                    seed=seed,
                    description=f"Rung 8: rank-average ensemble of {[m.name for m in members]}",
                    n_boot=n_boot if seed == seed_list[0] else 0,
                    log_to=log,
                )
            )
        by_rung["rank_ensemble"] = ens_evals
        rows.append(_summarise_rung("R8 rank_ensemble", ens_evals))

    table = pd.DataFrame(rows).set_index("name")
    lines.append("")
    lines.append("LADDER (mean over seeds)")
    lines.append(format_ladder(table))

    lines.append("")
    lines.append("RUNG COMPARISONS (paired stationary bootstrap on daily IC, against the rung below)")
    names = list(by_rung)
    comparisons: dict[str, dict[str, float | str]] = {}
    for lower_name, upper_name in zip(names[:-1], names[1:], strict=False):
        c = compare_rungs(by_rung[lower_name], by_rung[upper_name])
        comparisons[f"{upper_name} vs {lower_name}"] = c
        lines.append(
            f"  {upper_name:22s} vs {lower_name:22s} dIC {c['diff']:+.4f} "
            f"[{c['lo']:+.4f}, {c['hi']:+.4f}]  seed spread {c['seed_spread']:.4f}  -> {c['verdict']}"
        )

    h2_tab: pd.DataFrame | None = None
    h2_cmp: dict[str, dict[str, float | str]] = {}
    if with_h2:
        lines.append("")
        lines.append("H2: ranking vs classification (same features, same folds, same seeds)")
        fit_rungs(h2_rungs(), seed_list, paths=paths, snapshot_id=store.snapshot_id, jobs=jobs)
        h2_evals: dict[str, list[RunEvaluation]] = {}
        for r in h2_rungs():
            h2_evals[r.name] = [
                evaluate_predictions(
                    f"H2_{r.name}_s{seed}",
                    pd.read_parquet(predictions_path(paths, r, cfg, store.snapshot_id, seed)),
                    cfg=cfg,
                    paths=paths,
                    panel=panel,
                    labels=labels_df,
                    dates=dates,
                    nifty_ret=nifty_ret,
                    snapshot_id=store.snapshot_id,
                    seed=seed,
                    description=r.description,
                    n_boot=0,
                    log_to=log,
                    persist=False,
                )
                for seed in seed_list
            ]
        h2_tab = h2_table(h2_evals)
        lines.append(h2_tab.to_string(float_format=lambda v: f"{v:+.4f}"))
        base = "logistic_green"
        for other in ("lgbm_huber", "lgbm_lambdarank"):
            c = compare_rungs(h2_evals[base], h2_evals[other])
            h2_cmp[f"{other} vs {base}"] = c
            lines.append(
                f"  {other:16s} vs {base}: dIC {c['diff']:+.4f} "
                f"[{c['lo']:+.4f}, {c['hi']:+.4f}] -> {c['verdict']}"
            )

    dump_json(
        paths.research_dir / "ladder.json",
        {
            "snapshot_id": store.snapshot_id,
            "git_sha": sha,
            "config_hash": cfg.hash(),
            "seeds": seed_list,
            "oos_days": len(dates),
            "oos_start": str(dates[0].date()),
            "oos_end": str(dates[-1].date()),
            "table": table.reset_index().to_dict("records"),
            "comparisons": comparisons,
            "h2": h2_tab.reset_index().to_dict("records") if h2_tab is not None else [],
            "h2_comparisons": h2_cmp,
        },
    )
    elapsed = time.perf_counter() - t0
    lines.append("")
    lines.append(f"research_log rows: {log.count()} (counting for DSR: {log.count(dsr_only=True)})")
    lines.append(f"saved {paths.research_dir / 'ladder.json'}")
    lines.append(f"total elapsed {elapsed:.0f}s")
    return LadderOutput(
        table=table, comparisons=comparisons, h2=h2_tab, h2_comparisons=h2_cmp, lines=lines, elapsed_s=elapsed
    )


def _load() -> AppConfig:
    from greenclose.config import load_config

    return load_config()


__all__ = ["LadderOutput", "Rung", "run_ladder"]
