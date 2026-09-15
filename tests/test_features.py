"""Feature engine shape and semantics on the synthetic panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greenclose.config import AppConfig
from greenclose.features.build import FeaturePanel, build
from greenclose.features.registry import REGISTRY
from tests.fixtures.synthetic import SyntheticData, synthetic_panel


@pytest.fixture(scope="module")
def data() -> SyntheticData:
    return synthetic_panel(n_symbols=30, n_days=420, seed=1)


@pytest.fixture(scope="module")
def panel(data: SyntheticData, synth_config: AppConfig) -> FeaturePanel:
    store = data.store()
    return build(data.dates[-1], store, data.universe, synth_config)


def test_feature_count_within_budget(panel: FeaturePanel) -> None:
    assert 30 <= panel.n_model_features <= 50, panel.model_columns
    assert len(panel.tier1) == 35


def test_every_registered_feature_declares_a_lookback() -> None:
    for spec in REGISTRY.ordered():
        assert isinstance(spec.lookback_days, int) and spec.lookback_days >= 0, spec.name


def test_ranks_are_in_unit_interval_and_centred(panel: FeaturePanel) -> None:
    rk = panel.frame[[f"{c}_rk" for c in panel.tier1]]
    assert float(rk.min().min()) >= -1.0 and float(rk.max().max()) <= 1.0
    means = rk.groupby(level="date").mean().dropna(how="all")
    assert means.abs().max().max() < 1e-9  # ranks are symmetric within each day


def test_lookback_rows_are_nan_and_never_backfilled(panel: FeaturePanel) -> None:
    s = panel.frame.xs("S00", level="symbol")["ret_20d"]
    assert s.iloc[:20].isna().all()
    assert s.iloc[20:].notna().mean() > 0.99
    d = panel.frame.xs("S00", level="symbol")["dist_52w_high"]
    assert d.iloc[:251].isna().all()


def test_no_feature_is_entirely_nan(panel: FeaturePanel) -> None:
    cols = panel.tier1 + panel.tier2
    dead = [c for c in cols if panel.frame[c].isna().all()]
    assert dead == []


def test_sector_relative_is_leave_one_out(panel: FeaturePanel, data: SyntheticData) -> None:
    day = panel.frame.xs(data.dates[300], level="date")
    sec = day[day["sector"] == "fin"]
    r = sec["ret_1d"]
    for s in sec.index:
        others = r.drop(s).mean()
        assert abs(sec.loc[s, "sector_rel_ret_1d"] - (r[s] - others)) < 1e-12


def test_cluster_ids_are_small_integers(panel: FeaturePanel, data: SyntheticData) -> None:
    day = panel.frame.xs(data.dates[300], level="date")["cluster_id"].dropna()
    assert set(np.unique(day)).issubset(set(range(1, 9)))


def test_tier2_is_identical_across_symbols(panel: FeaturePanel, data: SyntheticData) -> None:
    day = panel.frame.xs(data.dates[300], level="date")
    for c in panel.tier2:
        assert day[c].nunique(dropna=True) <= 1, c


def test_build_is_deterministic(data: SyntheticData, synth_config: AppConfig) -> None:
    a = build(data.dates[-1], data.store(), data.universe, synth_config).frame
    b = build(data.dates[-1], data.store(), data.universe, synth_config).frame
    pd.testing.assert_frame_equal(a, b, check_exact=True)
