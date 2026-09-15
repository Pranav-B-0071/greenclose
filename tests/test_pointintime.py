"""Cut-off enforcement (brief invariant 1, section 13.5)."""

from __future__ import annotations

import pandas as pd
import pytest

from greenclose.ingest.store import ParquetBarStore


def _tiny_store(record_access: bool = False) -> ParquetBarStore:
    dates = pd.bdate_range("2024-01-01", periods=10)
    rows = []
    for sym in ["AAA", "BBB"]:
        for i, d in enumerate(dates):
            px = 100.0 + i
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": px,
                    "high": px + 1,
                    "low": px - 1,
                    "close": px,
                    "adj_close": px * 0.9,
                    "volume": 1000 + i,
                }
            )
    bars = pd.DataFrame(rows)
    index = bars[bars["symbol"] == "AAA"].assign(symbol="^NSEI")
    return ParquetBarStore(bars, index, snapshot_id="test", record_access=record_access)


def test_get_bars_without_cutoff_is_a_typeerror() -> None:
    store = _tiny_store()
    with pytest.raises(TypeError):
        store.get_bars(["AAA"])  # type: ignore[call-arg]


def test_get_bars_positional_cutoff_is_a_typeerror() -> None:
    store = _tiny_store()
    with pytest.raises(TypeError):
        store.get_bars(["AAA"], pd.Timestamp("2024-01-05"))  # type: ignore[misc]


def test_get_bars_rejects_non_timestamp_cutoff() -> None:
    store = _tiny_store()
    with pytest.raises(TypeError):
        store.get_bars(["AAA"], cutoff="2024-01-05")  # type: ignore[arg-type]


def test_cutoff_filters_the_future_inclusive_of_cutoff_day() -> None:
    store = _tiny_store()
    out = store.get_bars(["AAA", "BBB"], cutoff=pd.Timestamp("2024-01-05"))
    assert out["date"].max() == pd.Timestamp("2024-01-05")
    assert set(out["symbol"]) == {"AAA", "BBB"}
    # a non-trading cutoff (Saturday) returns up to the prior trading day
    out2 = store.get_bars(["AAA"], cutoff=pd.Timestamp("2024-01-06"))
    assert out2["date"].max() == pd.Timestamp("2024-01-05")


def test_lookback_days_limits_history() -> None:
    store = _tiny_store()
    out = store.get_bars(["AAA"], cutoff=pd.Timestamp("2024-01-12"), lookback_days=3)
    assert len(out) == 3
    assert out["date"].tolist() == list(pd.bdate_range("2024-01-10", periods=3))


def test_index_respects_cutoff() -> None:
    store = _tiny_store()
    idx = store.get_index(cutoff=pd.Timestamp("2024-01-03"))
    assert idx["date"].max() == pd.Timestamp("2024-01-03")


def test_derived_adjusted_columns() -> None:
    store = _tiny_store()
    out = store.get_bars(["AAA"], cutoff=pd.Timestamp("2024-01-03"))
    assert (out["adj_factor"] - 0.9).abs().max() < 1e-12
    assert (out["open_adj"] - out["open"] * 0.9).abs().max() < 1e-9


def test_access_recorder_never_sees_the_future() -> None:
    store = _tiny_store(record_access=True)
    for c in ["2024-01-02", "2024-01-08", "2024-01-12"]:
        store.get_bars(["AAA", "BBB"], cutoff=pd.Timestamp(c))
        store.get_index(cutoff=pd.Timestamp(c))
    assert store.violations() == []
    assert store.max_date_touched() == pd.Timestamp("2024-01-12")
