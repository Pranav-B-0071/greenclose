"""Phantom holiday bars are removed deterministically."""

from __future__ import annotations

import pandas as pd

from greenclose.ingest.clean import phantom_days, remove_phantom_days


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=6)
    rows = []
    for sym in [f"S{i}" for i in range(10)]:
        for i, d in enumerate(dates):
            vol = 0 if i == 3 else 1000  # day index 3 is a phantom holiday bar for everyone
            if sym == "S0" and i == 5:
                vol = 0  # one symbol with a genuine zero-volume day
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "adj_close": 10.0,
                    "volume": vol,
                }
            )
    return pd.DataFrame(rows)


def test_phantom_day_detected_and_removed() -> None:
    b = _panel()
    days = phantom_days(b)
    assert list(days) == [pd.Timestamp("2024-01-04")]
    cleaned, removed = remove_phantom_days(b)
    assert len(removed) == 1
    assert pd.Timestamp("2024-01-04") not in set(cleaned["date"])
    assert len(cleaned) == 50
    # the single-symbol zero-volume day survives
    assert (cleaned["volume"] == 0).sum() == 1
