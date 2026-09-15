"""The leakage suite run against the *real* snapshot (brief section 13).

`tests/test_leakage.py` runs the same six checks offline on a synthetic panel in CI. This module runs
them on the actual data with the ridge pipeline, because a harness that is clean on synthetic data can
still be leaky on the real one (a real corporate action, a real listing gap, a real index hole).

Phase 6 gates Phase 7: if any check here fails, no model may be built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from greenclose.backtest.engine import DayPanel
from greenclose.config import AppConfig, Paths
from greenclose.evaluation import metrics
from greenclose.evaluation.significance import newey_west
from greenclose.features.build import build
from greenclose.ingest.store import ParquetBarStore
from greenclose.models.train import RidgeHead, walk_forward_predict
from greenclose.ops.logging import get_logger

log = get_logger(__name__)

PLACEBO_IC_MAX = 0.005
PLACEBO_T_MAX = 2.0
POSITIVE_CONTROL_IC_MIN = 0.5
INJECTION_TARGET_IC = 0.10
INJECTION_BAND = (0.07, 0.13)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    numbers: dict[str, float] = field(default_factory=dict)

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name}: {self.detail}"


@dataclass
class LeakageReport:
    checks: list[Check]
    snapshot_id: str
    ran_at: str

    @property
    def n_passed(self) -> int:
        return sum(c.passed for c in self.checks)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        head = f"LEAKAGE SUITE on snapshot {self.snapshot_id}: {self.n_passed}/{len(self.checks)} PASS"
        return "\n".join([head, *(c.line() for c in self.checks)])


def _permute_within_date(labels: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute labels within each date, preserving the cross-sectional structure exactly."""
    rng = np.random.default_rng(seed)
    cols = ["fwd_c2c", "fwd_c2o", "fwd_o2c", "y_raw", "y_demeaned", "y_resid", "y_rank", "y_green", "y_beat"]
    out = labels.copy()
    for _, block in labels.groupby(level="date"):
        out.loc[block.index, cols] = block[cols].to_numpy()[rng.permutation(len(block))]
    return out


def _ridge_ic(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_columns: list[str],
    cfg: AppConfig,
    oos: pd.DatetimeIndex,
    *,
    extra: pd.DataFrame | None = None,
    seed: int = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    run = walk_forward_predict(
        features,
        labels,
        model_columns=model_columns,
        head=RidgeHead(),
        cfg=cfg,
        oos_dates=oos,
        seed=seed,
        extra_feature_frame=extra,
    )
    realised = labels["fwd_c2c"].reindex(run.predictions.index)
    return metrics.daily_ic(run.predictions["score"], realised), run.predictions


def run_suite(
    cfg: AppConfig,
    paths: Paths,
    *,
    store: ParquetBarStore,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_columns: list[str],
    n_truncation: int = 10,
    seed: int = 0,
) -> LeakageReport:
    panel = DayPanel.from_features(features)
    oos = panel.oos_dates(
        min_train_years=cfg.base.walk_forward.min_train_years,
        start=cfg.base.data.start,
        holdout_start=cfg.base.holdout.start,
    )
    checks: list[Check] = []

    # --- 13.1 shuffled-label placebo
    shuffled = _permute_within_date(labels, seed=11)
    ic, _ = _ridge_ic(features, shuffled, model_columns, cfg, oos)
    nw = newey_west(ic)
    ok = abs(float(ic.mean())) < PLACEBO_IC_MAX and abs(nw["t"]) < PLACEBO_T_MAX
    checks.append(
        Check(
            "13.1 shuffled-label placebo",
            ok,
            f"mean IC {ic.mean():+.5f} (bar |IC| < {PLACEBO_IC_MAX}), NW t {nw['t']:+.2f} "
            f"(bar |t| < {PLACEBO_T_MAX}), {len(ic)} days",
            {"ic": float(ic.mean()), "t": nw["t"], "n_days": float(len(ic))},
        )
    )

    # --- 13.2 truncation invariance
    dates = pd.DatetimeIndex(features.index.get_level_values("date").unique())
    rng = np.random.default_rng(13)
    picks = sorted(rng.choice(np.arange(300, len(dates) - 1), size=n_truncation, replace=False))
    bad: list[str] = []
    for i in picks:
        t = pd.Timestamp(dates[i])
        part = build(t, store, cfg.universe, cfg).frame
        expected = features.loc[features.index.get_level_values("date") <= t]
        try:
            pd.testing.assert_frame_equal(part, expected, check_exact=True)
        except AssertionError as exc:
            bad.append(f"{t.date()}: {str(exc).splitlines()[0]}")
    checks.append(
        Check(
            "13.2 truncation invariance",
            not bad,
            f"bitwise identical at {len(picks) - len(bad)}/{len(picks)} random cut-offs"
            + ("" if not bad else f"; first failure {bad[0]}"),
            {"n_cutoffs": float(len(picks)), "n_failed": float(len(bad))},
        )
    )

    # --- 13.3 positive control (the label injected as a feature)
    extra = labels[["y_demeaned"]].rename(columns={"y_demeaned": "leak"})
    ic_pc, _ = _ridge_ic(features, labels, model_columns, cfg, oos, extra=extra)
    checks.append(
        Check(
            "13.3 positive control",
            float(ic_pc.mean()) > POSITIVE_CONTROL_IC_MIN,
            f"mean IC {ic_pc.mean():+.3f} with the label as a feature (bar > {POSITIVE_CONTROL_IC_MIN})",
            {"ic": float(ic_pc.mean())},
        )
    )

    # --- 13.4 synthetic-signal injection at a known magnitude
    rho = INJECTION_TARGET_IC
    rng2 = np.random.default_rng(4)
    y = labels["y_demeaned"]
    z = (y - y.groupby(level="date").transform("mean")) / y.groupby(level="date").transform("std")
    injected = pd.DataFrame(
        {"injected": rho * z + math.sqrt(1 - rho**2) * rng2.standard_normal(len(z))}, index=labels.index
    )
    ic_inj, preds_inj = _ridge_ic(features, labels, model_columns, cfg, oos, extra=injected)
    lo, hi = INJECTION_BAND
    spread = float(
        metrics.decile_spread(preds_inj["score"], labels["fwd_c2c"].reindex(preds_inj.index)).mean() * 1e4
    )
    checks.append(
        Check(
            "13.4 synthetic-signal injection",
            lo <= float(ic_inj.mean()) <= hi,
            f"recovered IC {ic_inj.mean():+.4f} for an injected rho of {rho:.2f} "
            f"(band [{lo}, {hi}]); decile spread {spread:+.1f} bps/day",
            {"ic": float(ic_inj.mean()), "decile_spread_bps": spread},
        )
    )

    # --- 13.5 cut-off enforcement
    probe = ParquetBarStore.from_snapshot(
        paths.data_dir / "snapshots" / store.snapshot_id, record_access=True
    )
    type_error = False
    try:
        probe.get_bars(cfg.universe.symbols)  # type: ignore[call-arg]
    except TypeError:
        type_error = True
    for i in picks[:3]:
        build(pd.Timestamp(dates[i]), probe, cfg.universe, cfg)
    violations = probe.violations()
    checks.append(
        Check(
            "13.5 cut-off enforcement",
            type_error and not violations,
            f"get_bars without cutoff raises TypeError: {type_error}; "
            f"store handed out no row past the cutoff in {len(probe.access_log)} reads "
            f"({len(violations)} violations)",
            {"violations": float(len(violations))},
        )
    )

    # --- 13.6 in-fold transformers and split integrity
    run = walk_forward_predict(
        features,
        labels,
        model_columns=model_columns,
        head=RidgeHead(alpha=1.0),
        cfg=cfg,
        oos_dates=oos,
        seed=seed,
        select_hparams=False,
    )
    h, purge = cfg.base.labels.horizon_days, cfg.base.walk_forward.purge_days
    date_pos = pd.Series(np.arange(len(dates)), index=dates.sort_values())
    t_idx = date_pos.reindex(features.index.get_level_values("date")).to_numpy()
    first_col = features[model_columns[0]].to_numpy()
    y_ok = labels["y_demeaned"].reindex(features.index).notna().to_numpy()
    problems: list[str] = []
    for r in run.retrains:
        if r["train_last_idx"] + h + purge >= r["retrain_idx"]:
            problems.append(f"{r['retrain_date'].date()}: training rows reach the retrain day")
        mask = (t_idx + h + purge < r["retrain_idx"]) & y_ok
        if int(mask.sum()) != r["n_train_rows"]:
            problems.append(f"{r['retrain_date'].date()}: row count {r['n_train_rows']} != {int(mask.sum())}")
        col = first_col[mask]
        recomputed = float(np.where(np.isnan(col), np.nanmedian(col), col).mean())
        if abs(recomputed - r["pre_x_mean_first"]) > 1e-9:
            problems.append(
                f"{r['retrain_date'].date()}: scaler statistic differs from a training-row recompute"
            )
    checks.append(
        Check(
            "13.6 split integrity and in-fold fits",
            not problems,
            f"{len(run.retrains)} retrains checked: purge respected, row counts exact, scaler "
            f"statistics reproduce on training rows only"
            + ("" if not problems else f"; {len(problems)} problems, first: {problems[0]}"),
            {"n_retrains": float(len(run.retrains)), "n_problems": float(len(problems))},
        )
    )

    return LeakageReport(
        checks=checks,
        snapshot_id=store.snapshot_id,
        ran_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
