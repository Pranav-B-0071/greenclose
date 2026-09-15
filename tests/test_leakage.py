"""The leakage suite (brief section 13). Offline, deterministic, runs in CI.

13.1 shuffled-label placebo      13.2 truncation invariance      13.3 positive control (time-shift)
13.4 synthetic-signal injection  13.5 cut-off enforcement        13.6 split integrity / in-fold fits
(13.7, the golden decision file, lives in tests/test_decide.py.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from greenclose.backtest.engine import DayPanel, FrameScorer, RunSpec, run_backtest
from greenclose.config import AppConfig
from greenclose.evaluation import metrics, performance
from greenclose.evaluation.baselines import default_decision
from greenclose.evaluation.significance import newey_west
from greenclose.features.build import FeaturePanel, build
from greenclose.labels.targets import build_labels
from greenclose.models.splitters import PurgedGroupTimeSeriesSplit
from greenclose.models.train import RidgeHead, walk_forward_predict
from tests.fixtures.synthetic import SyntheticData, synthetic_panel


@pytest.fixture(scope="module")
def trunc_data() -> SyntheticData:
    return synthetic_panel(n_symbols=30, n_days=500, seed=2)


# ----------------------------------------------------------------------------- 13.2 truncation invariance


def test_13_2_truncation_invariance(trunc_data: SyntheticData, synth_config: AppConfig) -> None:
    """Features built on data truncated at T are bitwise identical, for all dates <= T, to features built
    on the full dataset. Ten random T across the sample."""
    store = trunc_data.store()
    full = build(trunc_data.dates[-1], store, trunc_data.universe, synth_config).frame
    rng = np.random.default_rng(13)
    picks = sorted(rng.choice(np.arange(300, len(trunc_data.dates) - 1), size=10, replace=False))
    for i in picks:
        t = pd.Timestamp(trunc_data.dates[i])
        part = build(t, store, trunc_data.universe, synth_config).frame
        expected = full.loc[full.index.get_level_values("date") <= t]
        pd.testing.assert_frame_equal(part, expected, check_exact=True, obj=f"cutoff={t.date()}")


# ----------------------------------------------------------------------------- shared pipeline fixture


@dataclass
class Pipeline:
    data: SyntheticData
    cfg: AppConfig
    panel: FeaturePanel
    labels: pd.DataFrame
    oos: pd.DatetimeIndex

    def ridge(self, labels: pd.DataFrame | None = None, extra: pd.DataFrame | None = None) -> pd.DataFrame:
        run = walk_forward_predict(
            self.panel.frame,
            labels if labels is not None else self.labels,
            model_columns=self.panel.model_columns,
            head=RidgeHead(),
            cfg=self.cfg,
            oos_dates=self.oos,
            seed=0,
            extra_feature_frame=extra,
        )
        return run.predictions

    def ic(self, predictions: pd.DataFrame, labels: pd.DataFrame | None = None) -> pd.Series:
        realised = (labels if labels is not None else self.labels)["fwd_c2c"].reindex(predictions.index)
        return metrics.daily_ic(predictions["score"], realised)


@pytest.fixture(scope="module")
def pipe(synth_config: AppConfig) -> Pipeline:
    data = synthetic_panel(n_symbols=50, n_days=700, seed=7)
    cfg = synth_config.override({"base.walk_forward.min_train_years": 1, "base.holdout.start": "2030-01-01"})
    store = data.store()
    panel = build(data.dates[-1], store, data.universe, cfg)
    labels = build_labels(panel.frame, store, cfg).frame
    oos = DayPanel.from_features(panel.frame).oos_dates(
        min_train_years=1, start="2018-01-01", holdout_start="2030-01-01"
    )
    return Pipeline(data=data, cfg=cfg, panel=panel, labels=labels, oos=oos)


def _permute_within_date(labels: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = ["fwd_c2c", "fwd_c2o", "fwd_o2c", "y_raw", "y_demeaned", "y_resid", "y_rank", "y_green", "y_beat"]
    out = labels.copy()
    for _, block in labels.groupby(level="date"):
        perm = rng.permutation(len(block))
        out.loc[block.index, cols] = block[cols].to_numpy()[perm]
    return out


# ----------------------------------------------------------------------------- 13.1 shuffled-label placebo


def test_13_1_shuffled_label_placebo(pipe: Pipeline) -> None:
    """Labels permuted within each date, whole pipeline re-run: no signal may be found."""
    shuffled = _permute_within_date(pipe.labels, seed=11)
    ic = pipe.ic(pipe.ridge(labels=shuffled), labels=shuffled)
    nw = newey_west(ic)
    assert len(ic) > 300
    assert abs(nw["t"]) < 2.5, f"placebo IC t-stat {nw['t']:.2f}"
    assert abs(ic.mean()) < 0.02, f"placebo mean IC {ic.mean():.4f}"


# ----------------------------------------------------------------------------- 13.3 positive control


def test_13_3_positive_control_time_shift(pipe: Pipeline) -> None:
    """Inject the label itself as a feature: the harness must detect an overwhelming signal."""
    extra = pipe.labels[["y_demeaned"]].rename(columns={"y_demeaned": "leak"})
    ic = pipe.ic(pipe.ridge(extra=extra))
    assert ic.mean() > 0.5, f"positive control IC {ic.mean():.3f}"


# --------------------------------------------------------------------------- 13.4 synthetic-signal injection


def _theoretical_decile_spread(
    rho: float, n: int, sigma: float, seed: int = 0, n_days: int = 20_000
) -> float:
    """Monte Carlo: expected top-minus-bottom decile realised return for a feature with correlation rho."""
    rng = np.random.default_rng(seed)
    m = max(1, int(np.floor(n * 0.1)))
    y = rng.standard_normal((n_days, n))
    x = rho * y + np.sqrt(1 - rho**2) * rng.standard_normal((n_days, n))
    order = np.argsort(-x, axis=1)
    top = np.take_along_axis(y, order[:, :m], axis=1).mean(axis=1)
    bottom = np.take_along_axis(y, order[:, -m:], axis=1).mean(axis=1)
    return float((top - bottom).mean() * sigma)


def test_13_4_synthetic_signal_injection_recovers_the_right_magnitude(pipe: Pipeline) -> None:
    """x = rho * z(y) + sqrt(1 - rho^2) * eps with rho chosen for a theoretical rank IC of ~0.10.
    The pipeline must recover an IC in [0.07, 0.13] and a P&L of the right order of magnitude."""
    rho = 0.10
    rng = np.random.default_rng(4)
    y = pipe.labels["y_demeaned"]
    z = (y - y.groupby(level="date").transform("mean")) / y.groupby(level="date").transform("std")
    x = rho * z + np.sqrt(1 - rho**2) * rng.standard_normal(len(z))
    extra = pd.DataFrame({"injected": x}, index=pipe.labels.index)
    preds = pipe.ridge(extra=extra)
    ic = pipe.ic(preds)
    assert 0.07 <= ic.mean() <= 0.13, f"recovered IC {ic.mean():.3f}"

    # magnitude: realised decile spread vs the Monte-Carlo expectation for this rho and panel width
    realised = pipe.labels["fwd_c2c"].reindex(preds.index)
    spread = metrics.decile_spread(preds["score"], realised).mean()
    sigma = float((realised - realised.groupby(level="date").transform("mean")).std())
    n = int(pipe.labels["n_labelled"].reindex(preds.index).median())
    expected = _theoretical_decile_spread(rho, n, sigma)
    assert 0.5 * expected <= spread <= 1.5 * expected, (
        f"decile spread {spread:.5f} vs expected {expected:.5f}"
    )

    # and the strategy layer converts it into gross P&L of the right sign
    dec = default_decision(pipe.cfg).with_(abstain_enabled=False)
    day_panel = DayPanel.from_features(pipe.panel.frame)
    res = run_backtest(
        RunSpec(name="inject", scorer=FrameScorer(preds), decision=dec, dates=pipe.oos), day_panel, pipe.cfg
    )
    summ = performance.summarise(res.daily, pipe.cfg.base.simulation.capital)
    assert summ["gross_return_pa"] > 0, "injected signal must produce positive gross P&L"


# ----------------------------------------------------------------------------- 13.5 cut-off enforcement


def test_13_5_feature_build_never_touches_the_future(
    trunc_data: SyntheticData, synth_config: AppConfig
) -> None:
    store = trunc_data.store(record_access=True)
    for i in (320, 400, 480):
        t = pd.Timestamp(trunc_data.dates[i])
        build(t, store, trunc_data.universe, synth_config)
        assert store.violations() == []
        touched = store.max_date_touched()
        assert touched is not None and touched <= t
        store.access_log.clear()


# ----------------------------------------------------------------------------- 13.6 in-fold transformers


def test_13_6_transformers_are_fitted_on_training_rows_only(pipe: Pipeline) -> None:
    run = walk_forward_predict(
        pipe.panel.frame,
        pipe.labels,
        model_columns=pipe.panel.model_columns,
        head=RidgeHead(alpha=1.0),
        cfg=pipe.cfg,
        oos_dates=pipe.oos,
        seed=0,
        select_hparams=False,
    )
    assert run.retrains, "no retrains happened"
    h, purge = pipe.cfg.base.labels.horizon_days, pipe.cfg.base.walk_forward.purge_days
    dates = pipe.panel.frame.index.get_level_values("date")
    all_dates = pd.DatetimeIndex(dates.unique()).sort_values()
    pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
    t_idx = pos.reindex(dates).to_numpy()
    first_col = pipe.panel.frame[pipe.panel.model_columns[0]]
    y = pipe.labels["y_demeaned"].reindex(pipe.panel.frame.index)
    for r in run.retrains[:3]:
        assert r["train_last_idx"] + h + purge < r["retrain_idx"], "training rows overlap the retrain day"
        mask = (t_idx + h + purge < r["retrain_idx"]) & y.notna().to_numpy()
        assert int(mask.sum()) == r["n_train_rows"]
        col = first_col.to_numpy()[mask]
        imputed = np.where(np.isnan(col), np.nanmedian(col), col)
        assert abs(float(imputed.mean()) - r["pre_x_mean_first"]) < 1e-9, (
            "scaler was not fitted on training rows"
        )


def test_13_6_splitter_cannot_shuffle() -> None:
    with pytest.raises(TypeError):
        PurgedGroupTimeSeriesSplit(3, shuffle=True)  # type: ignore[call-arg]
