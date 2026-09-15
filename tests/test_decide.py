"""The shared decision function: constraints, overlap-aware rebalance, abstention, and the golden file
(brief section 13.7): any refactor that silently changes decisions fails here."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenclose.domain.models import PortfolioState, Position
from greenclose.strategy.decide import DecisionConfig, decide, decide_with_trace

GOLDEN = Path(__file__).parent / "golden" / "decide_day.json"


def _day() -> pd.DataFrame:
    syms = [f"S{i}" for i in range(10)]
    return pd.DataFrame(
        {
            "close": [100.0, 200.0, 50.0, 400.0, 80.0, 120.0, 60.0, 300.0, 90.0, 150.0],
            "adj_close": [100.0, 200.0, 50.0, 400.0, 80.0, 120.0, 60.0, 300.0, 90.0, 150.0],
            "volume": [1e6, 1e6, 1e6, 1e6, 0, 1e6, 1e6, 1e6, 1e6, 1e6],
            "traded_value_20d": [1e9] * 10,
            "vol_20d": [0.02, 0.01, 0.03, 0.02, 0.02, 0.015, 0.025, 0.02, 0.02, 0.01],
            "sector": ["fin", "fin", "it", "it", "auto", "auto", "pharma", "pharma", "fmcg", "fmcg"],
            "cluster_id": [1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, np.nan],
            "staleness_flag": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        },
        index=pd.Index(syms, name="symbol"),
    )


def _preds(scores: list[float], edge: list[float] | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"score": scores}, index=pd.Index([f"S{i}" for i in range(10)], name="symbol"))
    if edge is not None:
        df["expected_return_bps"] = edge
    return df


def _state(
    positions: tuple[Position, ...] = (),
    cash: float = 1_000_000.0,
    asof_index: int = 10,
    values: dict[str, float] | None = None,
    prev: date | None = date(2024, 1, 9),
) -> PortfolioState:
    values = values or {}
    return PortfolioState(
        asof=date(2024, 1, 10),
        asof_index=asof_index,
        cash_settled=cash,
        cash_unsettled=0.0,
        positions=positions,
        equity=cash + sum(values.values()),
        prev_asof=prev,
        values=values,
    )


CFG = DecisionConfig(k=3, abstain_enabled=False, round_trip_cost_bps=34.0)
SCORES = [0.9, 0.8, 0.7, 0.6, 0.95, 0.5, 0.4, 0.3, 0.2, 0.1]


def test_top_k_with_sector_cap_skips_same_sector_and_stale_symbol() -> None:
    intents = decide(date(2024, 1, 10), _day(), _preds(SCORES), _state(), CFG)
    # S4 best score but stale; S0 fin, S1 fin (sector cap), S2 it, S3 it (cap) -> S0, S2, S5
    assert [i.symbol for i in intents] == ["S0", "S2", "S5"]
    assert all(i.side.value == "BUY" for i in intents)
    assert all(
        i.qty * i.notional > 0 and abs(i.qty * _day().loc[i.symbol, "close"] - i.notional) < 1e-9
        for i in intents
    )


def test_cluster_cap() -> None:
    day = _day()
    day.loc["S2", "cluster_id"] = 1.0  # same cluster as S0
    intents = decide(date(2024, 1, 10), day, _preds(SCORES), _state(), CFG)
    # S2 now shares cluster 1 with S0 -> skipped; S3 (it) becomes eligible, then S5 (auto)
    assert [i.symbol for i in intents] == ["S0", "S3", "S5"]


def test_inverse_vol_sizing_and_caps() -> None:
    intents = decide(date(2024, 1, 10), _day(), _preds(SCORES), _state(), CFG)
    by = {i.symbol: i.notional for i in intents}
    # inverse-vol: S0 (0.02), S2 (0.03), S5 (0.015): weights 50, 33.3, 66.7 -> mean 50 -> 1,00,000 * w/50
    assert by["S5"] > by["S0"] > by["S2"]
    assert by["S0"] == pytest.approx(100_000.0 * 50 / 50, rel=0.02)
    cfg = CFG.with_(max_position_pct=0.05)  # 5% of 10 lakh = 50,000 cap
    capped = decide(date(2024, 1, 10), _day(), _preds(SCORES), _state(), cfg)
    assert all(i.notional <= 50_000.0 + 1e-6 for i in capped)


def test_overlap_keeps_held_names_and_sells_dropouts() -> None:
    held = (
        Position(
            symbol="S0",
            qty=1000,
            entry_price=100.0,
            entry_adj_close=100.0,
            entry_date=date(2024, 1, 9),
            entry_index=9,
            entry_notional=100_000.0,
        ),
        Position(
            symbol="S9",
            qty=600,
            entry_price=150.0,
            entry_adj_close=150.0,
            entry_date=date(2024, 1, 9),
            entry_index=9,
            entry_notional=90_000.0,
        ),
    )
    state = _state(held, cash=800_000.0, values={"S0": 100_000.0, "S9": 90_000.0})
    intents, trace = decide_with_trace(date(2024, 1, 10), _day(), _preds(SCORES), state, CFG)
    assert [(i.symbol, i.side.value) for i in intents] == [("S9", "SELL"), ("S2", "BUY"), ("S5", "BUY")]
    assert trace.loc["S0", "action"] == "HOLD" and "overlap" in trace.loc["S0", "reason"]


def test_fixed_mode_sells_and_rebuys() -> None:
    held = (
        Position(
            symbol="S0",
            qty=1000,
            entry_price=100.0,
            entry_adj_close=100.0,
            entry_date=date(2024, 1, 9),
            entry_index=9,
            entry_notional=100_000.0,
        ),
    )
    state = _state(held, cash=900_000.0, values={"S0": 100_000.0})
    intents = decide(date(2024, 1, 10), _day(), _preds(SCORES), state, CFG.with_(rebalance_mode="fixed"))
    assert [(i.symbol, i.side.value) for i in intents] == [
        ("S0", "SELL"),
        ("S0", "BUY"),
        ("S2", "BUY"),
        ("S5", "BUY"),
    ]


def test_minimum_hold_period_blocks_early_exit() -> None:
    held = (
        Position(
            symbol="S9",
            qty=600,
            entry_price=150.0,
            entry_adj_close=150.0,
            entry_date=date(2024, 1, 9),
            entry_index=9,
            entry_notional=90_000.0,
        ),
    )
    state = _state(held, cash=900_000.0, values={"S9": 90_000.0})
    intents = decide(date(2024, 1, 10), _day(), _preds(SCORES), state, CFG.with_(hold_days=3))
    assert ("S9", "SELL") not in [(i.symbol, i.side.value) for i in intents]


def test_abstention_uses_expected_edge_against_round_trip_cost() -> None:
    edge = [60.0, 10.0, 20.0, 5.0, 90.0, 45.0, 0.0, 0.0, 0.0, 0.0]
    cfg = CFG.with_(abstain_enabled=True, round_trip_cost_bps=34.0, abstain_buffer_bps=5.0)
    intents, trace = decide_with_trace(date(2024, 1, 10), _day(), _preds(SCORES, edge), _state(), cfg)
    assert [i.symbol for i in intents] == ["S0", "S5"]  # S2 abstained: 20 < 39
    assert trace.loc["S2", "abstained"] and "edge" in trace.loc["S2", "reason"]


def test_settled_cash_limits_entries_in_rank_order() -> None:
    cfg = CFG.with_(min_position=50_000.0, max_position_pct=1.0)  # cash is the only binding limit
    intents = decide(date(2024, 1, 10), _day(), _preds(SCORES), _state(cash=120_000.0), cfg)
    assert [i.symbol for i in intents] == ["S0"]  # 1,00,000 for S0, 20,000 left < minimum


def test_monthly_rebalance_only_trades_on_first_day_of_month() -> None:
    cfg = CFG.with_(rebalance_frequency="monthly")
    assert decide(date(2024, 1, 10), _day(), _preds(SCORES), _state(prev=date(2024, 1, 9)), cfg) == []
    assert len(decide(date(2024, 1, 10), _day(), _preds(SCORES), _state(prev=date(2023, 12, 29)), cfg)) == 3


def test_golden_file() -> None:
    """Frozen day + frozen portfolio -> byte-identical intents. Regenerate deliberately, never silently."""
    held = (
        Position(
            symbol="S9",
            qty=600,
            entry_price=150.0,
            entry_adj_close=150.0,
            entry_date=date(2024, 1, 9),
            entry_index=9,
            entry_notional=90_000.0,
        ),
    )
    state = _state(held, cash=910_000.0, values={"S9": 90_000.0})
    edge = [60.0, 10.0, 20.0, 5.0, 90.0, 45.0, 0.0, 0.0, 0.0, 0.0]
    cfg = CFG.with_(abstain_enabled=True)
    intents = decide(date(2024, 1, 10), _day(), _preds(SCORES, edge), state, cfg)
    got = json.dumps(
        [
            asdict(i) | {"side": i.side.value, "product": i.product.value, "asof": i.asof.isoformat()}
            for i in intents
        ],
        indent=2,
        sort_keys=True,
    )
    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(got, encoding="utf-8")
        pytest.fail("golden file created; re-run the test and commit tests/golden/decide_day.json")
    assert got == GOLDEN.read_text(encoding="utf-8")
