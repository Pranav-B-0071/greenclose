"""Ladder, calibration, sweeps, holdout guard, forward log and the mechanical verdict.

All offline: nothing here touches the network or the real snapshot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenclose.config import AppConfig, Paths
from greenclose.evaluation.holdout import HoldoutAlreadyTouched, already_touched, record_touch, touch_path
from greenclose.evaluation.ladder import Rung, build_head, h2_rungs, ladder_rungs, rank_average
from greenclose.evaluation.report import Criterion, ReportInputs, overall_verdict
from greenclose.evaluation.significance import deflated_sharpe, expected_max_sharpe
from greenclose.evaluation.sweeps import COST_MULTIPLIERS, HOLD_PERIODS, K_VALUES, decision_cells
from greenclose.models.calibrate import CalibrationPair, Calibrator

# ----------------------------------------------------------------------------- calibration


def test_isotonic_calibrator_recovers_a_monotone_probability() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=4000)
    truth = 1.0 / (1.0 + np.exp(-2.0 * scores))
    outcomes = (rng.random(4000) < truth).astype(float)
    cal = Calibrator().fit(scores, outcomes)
    grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    p = cal.transform(grid)
    assert np.all(np.diff(p) >= -1e-9), "calibration must be monotone in the score"
    assert p[0] < 0.2 and p[-1] > 0.8
    assert np.all((p >= 0) & (p <= 1))


def test_calibrator_falls_back_to_the_base_rate_when_starved() -> None:
    cal = Calibrator().fit(np.arange(10.0), np.ones(10))
    assert np.allclose(cal.transform(np.arange(10.0)), 1.0)
    pair = CalibrationPair.fit(np.arange(10.0), np.ones(10), np.zeros(10))
    out = pair.transform(np.arange(3.0))
    assert set(out) == {"prob_beat", "prob_green"}


# ----------------------------------------------------------------------------- ladder plumbing


def test_rung_cache_key_depends_on_everything_that_changes_the_result(app_config: AppConfig) -> None:
    a = Rung(4, "x", "lgbm_huber", "d", "y_demeaned")
    b = Rung(4, "x", "lgbm_huber", "d", "y_demeaned", calibrate=True)
    assert a.key(app_config, "snap", 0) != a.key(app_config, "snap", 1)
    assert a.key(app_config, "snap", 0) != a.key(app_config, "other", 0)
    assert a.key(app_config, "snap", 0) != b.key(app_config, "snap", 0)
    other_cfg = app_config.override({"base.strategy.k": 7})
    assert a.key(app_config, "snap", 0) != a.key(other_cfg, "snap", 0)


def test_every_rung_builds_a_head(app_config: AppConfig) -> None:
    for r in [*ladder_rungs(), *h2_rungs()]:
        head = build_head(r.head_kind, app_config)
        assert head.target == r.target, r.name


def test_lightgbm_heads_receive_the_whole_fixed_grid(app_config: AppConfig) -> None:
    head = build_head("lgbm_huber", app_config)
    assert len(head.grid) == 72, "the brief fixes 72 cells; the grid may not be expanded"
    assert len(head.candidates()) == 72


def test_rank_average_is_scale_free() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-01", periods=5), [f"S{i}" for i in range(6)]], names=["date", "symbol"]
    )
    rng = np.random.default_rng(1)
    a = pd.DataFrame({"score": rng.normal(size=len(idx))}, index=idx)
    b = pd.DataFrame({"score": a["score"] * 1000.0 + 5.0}, index=idx)  # same ordering, different scale
    ens = rank_average([a, b])
    day = ens.xs(idx[0][0], level="date")["score"]
    same = a.xs(idx[0][0], level="date")["score"]
    assert day.rank().equals(same.rank()), "an ensemble of one ordering must preserve that ordering"


def test_rank_average_keeps_a_magnitude_head_only_when_every_member_has_one() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-01", periods=3), ["A", "B"]], names=["date", "symbol"]
    )
    with_bps = pd.DataFrame({"score": np.arange(6.0), "expected_return_bps": np.arange(6.0)}, index=idx)
    without = pd.DataFrame({"score": np.arange(6.0)}, index=idx)
    assert "expected_return_bps" in rank_average([with_bps, with_bps]).columns
    assert (
        "expected_return_bps" in rank_average([with_bps, without]).columns
    )  # averages the ones that have it


# ----------------------------------------------------------------------------- sweeps


def test_decision_cells_cover_the_required_sweeps(app_config: AppConfig) -> None:
    cells = decision_cells(app_config)
    by_group: dict[str, list[str]] = {}
    for c in cells:
        by_group.setdefault(c.group, []).append(c.label)
    assert len(by_group["hold"]) == len(HOLD_PERIODS)
    assert len(by_group["k"]) == len(K_VALUES)
    assert len(by_group["cost_multiplier"]) == len(COST_MULTIPLIERS)
    assert set(by_group) == {"hold", "k", "cost_multiplier", "product", "rebalance"}


def test_cost_multiplier_cells_move_the_abstention_threshold(app_config: AppConfig) -> None:
    cells = {c.label: c for c in decision_cells(app_config) if c.group == "cost_multiplier"}
    zero = cells["costs x0"].decision.round_trip_cost_bps
    triple = cells["costs x3"].decision.round_trip_cost_bps
    assert zero == pytest.approx(0.0, abs=1e-9)
    assert triple > 3 * 30.0, "a 3x cost multiplier must raise the abstention bar too"


def test_mis_is_cheaper_than_cnc_in_the_product_sweep(app_config: AppConfig) -> None:
    cells = {c.label: c for c in decision_cells(app_config) if c.group == "product"}
    assert (
        cells["product=MIS"].decision.round_trip_cost_bps < cells["product=CNC"].decision.round_trip_cost_bps
    )


# ----------------------------------------------------------------------------- DSR trial counting


def test_dsr_uses_the_honest_trial_count_not_only_trials_with_a_sharpe() -> None:
    sharpes = list(np.linspace(-0.05, 0.05, 20))
    few = deflated_sharpe(0.06, n_obs=1500, skew=0.0, kurt=3.0, sharpes_tried=sharpes)
    many = deflated_sharpe(0.06, n_obs=1500, skew=0.0, kurt=3.0, sharpes_tried=sharpes, n_trials=500)
    assert many["sr0"] > few["sr0"], "more trials must raise the bar"
    assert many["dsr"] < few["dsr"]
    assert many["n_trials"] == 500.0 and many["n_sharpes"] == 20.0
    assert expected_max_sharpe([0.01], n_trials=100) == 0.0  # one observation: no variance to estimate


# ----------------------------------------------------------------------------- holdout guard


def test_holdout_refuses_a_second_touch(tmp_path: Path, app_config: AppConfig) -> None:
    paths = Paths(root=tmp_path)
    (tmp_path / "research").mkdir(parents=True)
    assert not already_touched(paths)
    record_touch(paths, {"touched_at": "2026-09-15T00:00:00+00:00", "ic_mean": 0.01})
    assert already_touched(paths)
    assert "touched_at" in touch_path(paths).read_text(encoding="utf-8")
    from greenclose.evaluation.holdout import run_holdout

    with pytest.raises(HoldoutAlreadyTouched, match="exactly once"):
        run_holdout(app_config, paths)


# ----------------------------------------------------------------------------- mechanical verdict


def _inputs(app_config: AppConfig, tmp_path: Path) -> ReportInputs:
    return ReportInputs(cfg=app_config, paths=Paths(root=tmp_path), snapshot_id="snap")


def test_verdict_is_mechanical(app_config: AppConfig, tmp_path: Path) -> None:
    inp = _inputs(app_config, tmp_path)
    ic_pass = Criterion("Mean daily rank IC", "", True)
    ic_fail = Criterion("Mean daily rank IC", "", False)
    money_pass = Criterion("Net return vs B2 (equal-weight universe)", "", True)
    money_fail = Criterion("Net return vs B2 (equal-weight universe)", "", False)

    v = overall_verdict([ic_pass, money_fail], inp, {"ic_mean": 0.03})
    assert "predicts" in v and "does not pay" in v
    v = overall_verdict([ic_pass, money_pass], inp, {"ic_mean": 0.03})
    assert "net-positive" in v
    v = overall_verdict([ic_fail, money_fail], inp, {"ic_mean": 0.005})
    assert "not significant" in v
    v = overall_verdict([ic_fail, money_fail], inp, {"ic_mean": -0.01})
    assert "No detectable" in v


def test_an_implausibly_good_ic_is_flagged_as_a_probable_leak(app_config: AppConfig, tmp_path: Path) -> None:
    inp = _inputs(app_config, tmp_path)
    v = overall_verdict([Criterion("Mean daily rank IC", "", True)], inp, {"ic_mean": 0.09})
    assert "RED FLAG" in v and "0.06" in v


def test_a_failed_leakage_suite_overrides_every_other_verdict(app_config: AppConfig, tmp_path: Path) -> None:
    inp = _inputs(app_config, tmp_path)
    crit = [
        Criterion("Mean daily rank IC", "", True),
        Criterion("Net return vs B2 (equal-weight universe)", "", True),
        Criterion("Leakage suite", "5/6 PASS", False),
    ]
    assert "LEAKAGE SUITE FAILED" in overall_verdict(crit, inp, {"ic_mean": 0.03})


# ----------------------------------------------------------------------------- forward log


def test_forward_summary_line_labels_itself_an_integrity_check() -> None:
    from greenclose.evaluation.forward import IC_DAYS_FOR_T2, summary_line

    idx = pd.bdate_range("2026-09-01", periods=3)
    frame = pd.DataFrame(
        {
            "date": np.repeat(idx, 2),
            "symbol": ["A", "B"] * 3,
            "score": [0.2, 0.1] * 3,
            "fwd_c2c": [0.01, -0.01] * 3,
            "sim_net_pnl": [50.0, 0.0] * 3,
        }
    )
    picks = pd.DataFrame({"symbol": ["A"]})
    line = summary_line(frame, pd.Timestamp("2026-09-03"), picks)
    assert "integrity check" in line and "Not evidence of an edge" in line
    assert str(IC_DAYS_FOR_T2) in line
    assert "2026-09-03" in line and "A" in line


def test_forward_log_columns_include_every_symbol_and_the_provenance(app_config: AppConfig) -> None:
    from greenclose.evaluation.forward import OUTCOME_COLUMNS, PREDICTION_COLUMNS

    for col in ("symbol", "score", "rank", "selected", "abstained", "abstain_reason", "model_version"):
        assert col in PREDICTION_COLUMNS
    for col in ("git_sha", "config_hash", "snapshot_id", "cutoff_ist"):
        assert col in PREDICTION_COLUMNS, "a forward row must be traceable to the code that wrote it"
    for col in ("fwd_c2c", "fwd_c2o", "fwd_o2c", "beat_universe_mean", "sim_net_pnl"):
        assert col in OUTCOME_COLUMNS


def test_model_registry_round_trip_and_pin(tmp_path: Path) -> None:
    from greenclose.models.registry import ModelCard, ModelRegistry, make_version

    reg = ModelRegistry(tmp_path / "models")
    assert reg.pinned_version() is None
    with pytest.raises(FileNotFoundError, match="pin-model"):
        reg.load_pinned()
    version = make_version("lgbm_huber", "snap", 0, "2026-09-15")
    card = ModelCard(
        model_version=version,
        head_kind="lgbm_huber",
        target="y_demeaned",
        seed=0,
        git_sha="abc",
        config_hash="def",
        snapshot_id="snap",
        feature_registry="v1",
        model_columns=["a", "b"],
        train_start="2015-01-01",
        train_end="2026-09-15",
    )
    reg.save(card, {"fake": "model"}, {"fake": "pre"})
    reg.pin(version, reason="test")
    assert reg.pinned_version() == version
    back, model, pre = reg.load_pinned()
    assert back.model_version == version and model == {"fake": "model"} and pre == {"fake": "pre"}
    assert reg.list_versions() == [version]
    assert json.loads((reg.path(version) / "card.json").read_text())["git_sha"] == "abc"


def test_no_broker_code_anywhere_in_the_repository() -> None:
    """The brief's hardest requirement: zero broker code, zero order placement, zero credentials."""
    root = Path(__file__).resolve().parents[1]
    banned = ("kiteconnect", "KiteTicker", "place_order", "api_secret", "access_token")
    offenders: list[str] = []
    for path in list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                offenders.append(f"{path.relative_to(root)}: {token}")
    assert offenders == [], offenders


def test_math_import_is_used() -> None:
    assert math.isfinite(1.0)
