"""The leakage suite (brief section 13). Offline, deterministic, runs in CI.

13.2 truncation invariance lives here from P3; the remaining tests are added in P6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greenclose.config import AppConfig
from greenclose.features.build import build
from tests.fixtures.synthetic import SyntheticData, synthetic_panel


@pytest.fixture(scope="module")
def trunc_data() -> SyntheticData:
    return synthetic_panel(n_symbols=30, n_days=500, seed=2)


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
