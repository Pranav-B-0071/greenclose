"""The model ladder (brief section 10).

Climb in order. A rung may not be declared an improvement on the one below unless the difference
survives a paired stationary bootstrap over days *and* exceeds the spread across the five seeds. If
seed variance is larger than the difference between two models, the report says so in those words.

    0  uniform random top-K                      (the null distribution; from `greenclose baselines`)
    1  equal-weight universe, monthly rebalance   (does selection add anything at all?)
    2  single-feature sorts: reversal, momentum   (do the classic effects exist here?)
    3  ridge on y_demeaned; logistic on y_beat    (the real baseline)
    4  LightGBM regression, Huber objective       (does non-linearity add anything?)
    5  LightGBM lambdarank                        (the ranking-vs-regression question)
    6  CatBoost + XGBoost                         (ensemble diversity; disagreement as confidence)
    7  probability calibration on a purged fold   (do the probabilities mean anything?)
    8  rank-average ensemble of rungs 3-6         (does averaging beat the best single model?)

Predictions are cached per (rung, seed, snapshot, config) so the decision-layer sweeps in P8 never
retrain anything.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from greenclose.backtest.engine import DayPanel, FrameScorer, RunSpec, run_backtest
from greenclose.config import AppConfig, Paths
from greenclose.domain.models import Product
from greenclose.evaluation.baselines import (
    RunEvaluation,
    default_decision,
    evaluate_run,
    log_trial,
    persist_run,
)
from greenclose.evaluation.significance import paired_bootstrap_diff
from greenclose.models.specs import (
    CatBoostHead,
    LGBMRankerHead,
    LGBMRegressorHead,
    LogisticHead,
    RidgeHead,
    XGBoostHead,
    lightgbm_grid,
)
from greenclose.models.train import walk_forward_predict
from greenclose.ops.logging import get_logger
from greenclose.ops.provenance import Provenance
from greenclose.ops.research_log import ResearchLog, Trial

log = get_logger(__name__)


@dataclass
class Rung:
    number: int
    name: str
    head_kind: str  # resolved to a head object inside the worker, so specs stay picklable
    description: str
    target: str
    calibrate: bool = False
    select_hparams: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def key(self, cfg: AppConfig, snapshot_id: str, seed: int) -> str:
        payload = json.dumps(
            {
                "rung": self.number,
                "name": self.name,
                "kind": self.head_kind,
                "target": self.target,
                "calibrate": self.calibrate,
                "params": self.params,
                "cfg": cfg.hash(),
                "snapshot": snapshot_id,
                "seed": seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_head(kind: str, cfg: AppConfig, params: dict[str, Any] | None = None) -> Any:
    grid = lightgbm_grid(cfg.grid.lightgbm_cells())
    purge = cfg.base.walk_forward.purge_days
    heads: dict[str, Any] = {
        "ridge": lambda: RidgeHead(alphas=tuple(cfg.grid.ridge["alpha"])),
        "logistic_beat": lambda: LogisticHead(name="logistic_beat", target="y_beat", prob_column="prob_beat"),
        "logistic_green": lambda: LogisticHead(
            name="logistic_green", target="y_green", prob_column="prob_green"
        ),
        "lgbm_huber": lambda: LGBMRegressorHead(grid=grid, purge=purge),
        "lgbm_lambdarank": lambda: LGBMRankerHead(grid=grid, purge=purge),
        "catboost": lambda: CatBoostHead(purge=purge),
        "xgboost": lambda: XGBoostHead(purge=purge),
    }
    if kind not in heads:
        raise KeyError(f"unknown head {kind!r}; known: {sorted(heads)}")
    head = heads[kind]()
    for k, v in (params or {}).items():
        setattr(head, k, v)
    return head


def ladder_rungs() -> list[Rung]:
    return [
        Rung(3, "ridge", "ridge", "Rung 3: ridge on demeaned forward return", "y_demeaned"),
        Rung(3, "logistic_beat", "logistic_beat", "Rung 3: logistic on beat-the-mean", "y_beat"),
        Rung(4, "lgbm_huber", "lgbm_huber", "Rung 4: LightGBM regression, Huber objective", "y_demeaned"),
        Rung(
            5,
            "lgbm_lambdarank",
            "lgbm_lambdarank",
            "Rung 5: LightGBM lambdarank, one query group per date",
            "y_rank",
        ),
        Rung(6, "catboost", "catboost", "Rung 6: CatBoost regression (ensemble diversity)", "y_demeaned"),
        Rung(6, "xgboost", "xgboost", "Rung 6: XGBoost regression (ensemble diversity)", "y_demeaned"),
        Rung(
            7,
            "lgbm_huber_calibrated",
            "lgbm_huber",
            "Rung 7: rung 4 with isotonic calibration on a purged fold",
            "y_demeaned",
            calibrate=True,
        ),
    ]


def h2_rungs() -> list[Rung]:
    """The headline experiment: same features, same folds, same seeds, three heads (brief 10.3)."""
    return [
        Rung(
            3,
            "logistic_green",
            "logistic_green",
            "H2 (a): per-stock logistic classifier on y_green",
            "y_green",
        ),
        Rung(4, "lgbm_huber", "lgbm_huber", "H2 (b): LightGBM regression on y_demeaned", "y_demeaned"),
        Rung(5, "lgbm_lambdarank", "lgbm_lambdarank", "H2 (c): LightGBM lambdarank on y_rank", "y_rank"),
    ]


# ----------------------------------------------------------------------------- prediction cache


def predictions_path(paths: Paths, rung: Rung, cfg: AppConfig, snapshot_id: str, seed: int) -> Path:
    return (
        paths.data_dir / "predictions" / f"{rung.name}__s{seed}__{rung.key(cfg, snapshot_id, seed)}.parquet"
    )


def schedule_path(paths: Paths, rung: Rung, cfg: AppConfig, snapshot_id: str) -> Path:
    """Where the seed-0 hyperparameter choices live. Shared by every seed of this rung."""
    return paths.data_dir / "predictions" / f"{rung.name}__hparams__{rung.key(cfg, snapshot_id, 0)}.json"


_W: dict[str, Any] = {}


def _worker_init(root: str, snapshot_id: str, threads: int = 0) -> None:
    from greenclose.config import Paths as _Paths
    from greenclose.config import load_config as _load
    from greenclose.pipeline_io import load_features, load_labels

    if threads > 0:
        # Without this every worker asks BLAS and LightGBM for all 16 cores and they thrash each
        # other; capping per-worker threads is worth several times the wall clock in a pool.
        _W["threads"] = threads
        try:
            from threadpoolctl import threadpool_limits

            _W["_limiter"] = threadpool_limits(limits=threads)
        except ImportError:  # pragma: no cover
            pass
    paths = _Paths(root=Path(root))
    cfg = _load(paths)
    feats = load_features(paths, snapshot_id)
    _W["cfg"] = cfg
    _W["paths"] = paths
    _W["features"] = feats
    _W["labels"] = load_labels(paths, snapshot_id, cfg.base.labels.horizon_days, cfg.base.labels.horizon)
    _W["model_columns"] = model_columns_of(feats)
    _W["oos"] = DayPanel.from_features(feats).oos_dates(
        min_train_years=cfg.base.walk_forward.min_train_years,
        start=cfg.base.data.start,
        holdout_start=cfg.base.holdout.start,
    )
    _W["snapshot_id"] = snapshot_id


def _worker_fit(args: tuple[Rung, int]) -> tuple[str, int, str, int, float]:
    """Fit one (rung, seed) walk-forward and cache its predictions. Returns cache metadata."""
    import time

    rung, seed = args
    cfg, paths = _W["cfg"], _W["paths"]
    out = predictions_path(paths, rung, cfg, _W["snapshot_id"], seed)
    if out.exists():
        return rung.name, seed, str(out), 0, 0.0
    t0 = time.perf_counter()
    sched_file = schedule_path(paths, rung, cfg, _W["snapshot_id"])
    schedule = json.loads(sched_file.read_text(encoding="utf-8")) if sched_file.exists() else None
    head = build_head(rung.head_kind, cfg, rung.params)
    threads = int(_W.get("threads", 0))
    if threads > 0 and hasattr(head, "params"):
        head.params = {**head.params, **_thread_params(rung.head_kind, threads)}
    run = walk_forward_predict(
        _W["features"],
        _W["labels"],
        model_columns=_W["model_columns"],
        head=head,
        cfg=cfg,
        oos_dates=_W["oos"],
        seed=seed,
        select_hparams=rung.select_hparams,
        calibrate=rung.calibrate,
        hparam_schedule=schedule,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    run.predictions.to_parquet(out)
    if schedule is None and run.hparam_schedule:
        sched_file.write_text(json.dumps(run.hparam_schedule, indent=2, default=str), encoding="utf-8")
    return rung.name, seed, str(out), run.n_grid_evaluations, time.perf_counter() - t0


def _thread_params(kind: str, threads: int) -> dict[str, Any]:
    """Per-library thread caps, so N pool workers use N x threads cores rather than N x all cores."""
    if kind.startswith("lgbm"):
        return {"num_threads": threads}
    if kind == "xgboost":
        return {"nthread": threads}
    if kind == "catboost":
        return {"thread_count": threads}
    return {}


def model_columns_of(features: pd.DataFrame) -> list[str]:
    """Ranked Tier-1 columns plus Tier-2 context. Raw (unranked) columns are never model inputs."""
    tier1 = [c for c in features.columns if c.endswith("_rk")]
    tier2 = [
        c
        for c in features.columns
        if c.startswith(("nifty_", "breadth", "xs_", "india_", "avg_pairwise")) and not c.endswith("_rk")
    ]
    return tier1 + tier2


def fit_rungs(
    rungs: list[Rung], seeds: list[int], *, paths: Paths, snapshot_id: str, jobs: int = 1
) -> list[tuple[str, int, str, int, float]]:
    """Fit every (rung, seed). The first seed of each rung runs first and alone, because it performs
    the hyperparameter search whose result the other seeds replay."""
    if not seeds:
        return []
    first, rest = seeds[0], seeds[1:]
    lead = [(r, first) for r in rungs]
    follow = [(r, s) for r in rungs for s in rest]
    if jobs <= 1:
        _worker_init(str(paths.root), snapshot_id)
        return [_worker_fit(a) for a in lead + follow]
    threads = max(1, (os.cpu_count() or 8) // jobs)
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=jobs, initializer=_worker_init, initargs=(str(paths.root), snapshot_id, threads)
    ) as pool:
        out = pool.map(_worker_fit, lead, chunksize=1)
        out += pool.map(_worker_fit, follow, chunksize=1)
    return out


# ----------------------------------------------------------------------------- evaluation


def rank_average(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Rung 8: average the within-day ranks of several prediction frames (scale-free by construction)."""
    ranks = []
    for f in frames:
        r = f["score"].groupby(level="date").rank(pct=True)
        ranks.append(r)
    avg = pd.concat(ranks, axis=1).mean(axis=1)
    out = pd.DataFrame({"score": avg})
    # carry a magnitude head through if every member has one, so the abstention gate still works
    bps = [f["expected_return_bps"] for f in frames if "expected_return_bps" in f.columns]
    if bps:
        out["expected_return_bps"] = pd.concat(bps, axis=1).mean(axis=1)
    return out


def evaluate_predictions(
    name: str,
    predictions: pd.DataFrame,
    *,
    cfg: AppConfig,
    paths: Paths,
    panel: DayPanel,
    labels: pd.DataFrame,
    dates: pd.DatetimeIndex,
    nifty_ret: pd.Series,
    snapshot_id: str,
    seed: int,
    description: str,
    n_boot: int,
    log_to: ResearchLog | None,
    persist: bool = True,
) -> RunEvaluation:
    """Run the shared decision layer over a prediction frame and evaluate both metric families."""
    dec = default_decision(cfg)
    if "expected_return_bps" not in predictions.columns:
        dec = dec.with_(abstain_enabled=False)  # no magnitude head: the cost gate cannot be applied
    spec = RunSpec(
        name=name,
        scorer=FrameScorer(predictions),
        decision=dec,
        dates=dates,
        description=description,
        family="ladder",
        product=Product(cfg.base.simulation.product),
        cost_multiplier=cfg.base.simulation.cost_multiplier,
        slippage_override_bps=cfg.base.simulation.slippage_override_bps,
        capital=cfg.base.simulation.capital,
        seed=seed,
    )
    prov = Provenance.create(config_hash=cfg.hash(), snapshot_id=snapshot_id, seed=seed, label=name)
    result = run_backtest(spec, panel, cfg, provenance=prov)
    ev = evaluate_run(result, labels, capital=cfg.base.simulation.capital, nifty_ret=nifty_ret, n_boot=n_boot)
    if persist:
        persist_run(result, ev, paths)
    if log_to is not None:
        log_trial(log_to, result, ev, family="ladder")
    return ev


def seed_summary(evals: list[RunEvaluation], metric: str, family: str = "family_a") -> dict[str, float]:
    vals = [getattr(e, family).get(metric, math.nan) for e in evals]
    arr = np.array([v for v in vals if v == v], dtype="float64")
    if len(arr) == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "n": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": float(len(arr)),
    }


def compare_rungs(
    lower: list[RunEvaluation], upper: list[RunEvaluation], metric: str = "ic"
) -> dict[str, float | str]:
    """Paired stationary bootstrap on the per-date difference, plus the seed-spread honesty check."""
    a = pd.concat([e.per_date[metric] for e in upper if metric in e.per_date], axis=1).mean(axis=1)
    b = pd.concat([e.per_date[metric] for e in lower if metric in e.per_date], axis=1).mean(axis=1)
    boot = paired_bootstrap_diff(a, b, n_boot=5000)
    spread_upper = seed_summary(upper, "ic_mean")["std"]
    spread_lower = seed_summary(lower, "ic_mean")["std"]
    seed_spread = max(spread_upper, spread_lower)
    diff = boot["diff"]
    beaten = bool(boot["lo"] > 0)
    real = beaten and abs(diff) > seed_spread
    verdict = (
        "beaten"
        if real
        else ("not beaten" if not beaten else "difference smaller than the seed spread: not real")
    )
    return {
        "diff": diff,
        "lo": boot["lo"],
        "hi": boot["hi"],
        "seed_spread": seed_spread,
        "verdict": verdict,
    }


def h2_table(results: dict[str, list[RunEvaluation]]) -> pd.DataFrame:
    """Precision@1, Precision@3, rank IC and NDCG@5 for the three H2 heads, mean over seeds."""
    rows = []
    for name, evals in results.items():
        row: dict[str, float | str] = {"head": name}
        for metric in ("ic_mean", "p_at_1_beat_mean", "p_at_3_beat_mean", "lift_at_3_mean", "ndcg_at_5_mean"):
            s = seed_summary(evals, metric)
            row[metric] = s["mean"]
            row[f"{metric}_sd"] = s["std"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("head")


def log_grid_trials(
    log: ResearchLog, rung: Rung, n_cells: int, cfg: AppConfig, snapshot_id: str, git_sha: str
) -> None:
    """Every hyperparameter cell evaluated counts toward the multiple-testing penalty (invariant 9)."""
    if n_cells <= 0:
        return
    log.append(
        Trial(
            description=f"{rung.name}: {n_cells} hyperparameter cells evaluated by purged CV in-window",
            family="grid",
            config_hash=cfg.hash(),
            snapshot_id=snapshot_id,
            git_sha=git_sha,
            seed=0,
            counts_for_dsr=True,
            extra={"rung": rung.number, "n_cells": n_cells},
        )
    )


def format_ladder(table: pd.DataFrame) -> str:
    fmt = {
        "ic_mean": "{:+.4f}",
        "ic_sd": "{:.4f}",
        "ic_t_nw": "{:+.2f}",
        "lift_at_3": "{:+.4f}",
        "decile_spread_bps": "{:+.1f}",
        "gross_pa": "{:+.1%}",
        "net_pa": "{:+.1%}",
        "sharpe": "{:+.2f}",
        "costs_over_gross": "{:.0%}",
        "gross_bps_inv": "{:+.2f}",
        "net_bps_inv": "{:+.2f}",
    }

    def _f(v: object, f: str) -> str:
        if not isinstance(v, float) or np.isnan(v):
            return ""
        return "inf" if np.isinf(v) else f.format(v)

    cols = [c for c in fmt if c in table.columns]
    return table[cols].to_string(formatters={k: (lambda v, f=fmt[k]: _f(v, f)) for k in cols})


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def rung_asdict(r: Rung) -> dict[str, Any]:
    return asdict(r)
