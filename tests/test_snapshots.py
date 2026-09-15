"""Snapshot immutability, content hashing and drift detection (brief section 4.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from greenclose.ingest.snapshots import diff_bars, frame_sha256, read_bars, read_manifest, write_snapshot


def _bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=5)
    rows = []
    for sym in ["AAA", "BBB"]:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": 10.0 + i,
                    "high": 11.0 + i,
                    "low": 9.0 + i,
                    "close": 10.5 + i,
                    "adj_close": 10.4 + i,
                    "volume": 100 + i,
                }
            )
    return pd.DataFrame(rows)


def test_hash_is_order_independent_and_deterministic() -> None:
    b = _bars()
    shuffled = b.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert frame_sha256(b) == frame_sha256(shuffled)
    changed = b.copy()
    changed.loc[0, "close"] += 1e-6
    assert frame_sha256(b) != frame_sha256(changed)


def test_snapshot_round_trip_and_immutability(tmp_path: Path) -> None:
    b = _bars()
    idx = b[b["symbol"] == "AAA"].assign(symbol="^NSEI")
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    d = write_snapshot(tmp_path, b, idx, source="test", failed_symbols=["ZZZ"], taken_at=t0)
    m = read_manifest(d)
    assert m.row_count == len(b) and m.symbols == ["AAA", "BBB"] and m.failed_symbols == ["ZZZ"]
    assert m.bars_sha256 == frame_sha256(b)
    pd.testing.assert_frame_equal(read_bars(d), b.sort_values(["symbol", "date"]).reset_index(drop=True))
    # same instant -> refuses to overwrite
    try:
        write_snapshot(tmp_path, b, idx, source="test", failed_symbols=[], taken_at=t0)
    except FileExistsError:
        pass
    else:
        raise AssertionError("snapshot overwrite must be refused")


def test_drift_detects_changed_historical_bar() -> None:
    old = _bars()
    new = old.copy()
    new.loc[(new["symbol"] == "AAA") & (new["date"] == pd.Timestamp("2024-01-03")), "adj_close"] *= 0.98
    new.loc[(new["symbol"] == "BBB") & (new["date"] == pd.Timestamp("2024-01-02")), "volume"] += 5
    rep = diff_bars(old, new, old_id="a", new_id="b")
    assert rep.changed
    assert set(zip(rep.summary["symbol"], rep.summary["column"], strict=True)) == {
        ("AAA", "adj_close"),
        ("BBB", "volume"),
    }
    assert len(rep.detail) == 2


def test_drift_ignores_new_dates_and_flags_removed_symbols() -> None:
    old = _bars()
    extra = old[old["date"] == old["date"].max()].copy()
    extra["date"] = extra["date"] + pd.Timedelta(days=3)
    new = pd.concat([old[old["symbol"] == "AAA"], extra[extra["symbol"] == "AAA"]], ignore_index=True)
    rep = diff_bars(old, new, old_id="a", new_id="b")
    assert len(rep.detail) == 0
    assert rep.removed_symbols == ["BBB"]
    assert len(rep.removed_bars) == 5


def test_raw_hash_ignores_source_adjustment_noise() -> None:
    from greenclose.ingest.snapshots import raw_sha256

    a = _bars()
    b = a.copy()
    b["adj_close"] = b["adj_close"] * (1 + 2e-7)  # float32-level noise in the adjusted series
    assert frame_sha256(a) != frame_sha256(b)
    assert raw_sha256(a) == raw_sha256(b)
    rep = diff_bars(a, b, old_id="a", new_id="b")
    assert not rep.changed, "sub-tolerance adj_close noise must not be reported as drift"
    c = a.copy()
    c.loc[c["symbol"] == "AAA", "adj_close"] *= 0.99  # a real dividend adjustment
    assert diff_bars(a, c, old_id="a", new_id="c").changed
