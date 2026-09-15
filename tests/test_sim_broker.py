"""Fill simulator and portfolio ledger invariants (brief section 9.2)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from greenclose.backtest.costs import CostModel
from greenclose.backtest.portfolio import InsufficientCashError, Portfolio
from greenclose.backtest.sim_broker import SimBroker
from greenclose.config import AppConfig
from greenclose.domain.models import Intent, Position, Product, Side


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0, 50.0, 200.0],
            "adj_close": [90.0, 50.0, 200.0],
            "volume": [1_000_000, 0, 500_000],
            "traded_value_20d": [1e9, 1e8, 5e8],
            "vol_20d": [0.018, 0.02, 0.015],
        },
        index=pd.Index(["AAA", "BBB", "CCC"], name="symbol"),
    )


def _intent(sym: str, side: Side, qty: int, notional: float) -> Intent:
    return Intent(
        decision_id=f"x-{sym}",
        asof=date(2024, 1, 2),
        symbol=sym,
        side=side,
        qty=qty,
        notional=notional,
        reason="t",
    )


def test_buy_fills_at_close_with_slippage_as_a_cost_line(app_config: AppConfig) -> None:
    broker = SimBroker(CostModel(app_config.charges, product=Product.CNC), slippage_override_bps=5.0)
    fills = broker.execute(
        [_intent("AAA", Side.BUY, 1000, 100_000.0)], bars_today=_bars(), positions={}, asof=date(2024, 1, 2)
    )
    f = fills[0]
    assert not f.missed and f.price == 100.0 and f.qty == 1000 and f.notional == 100_000.0
    assert f.costs.slippage == pytest.approx(50.0)  # 5 bps of 1,00,000
    assert f.costs.stt == pytest.approx(100.0) and f.costs.stamp == pytest.approx(15.0)
    assert f.costs.dp_charge == 0.0


def test_zero_volume_does_not_fill(app_config: AppConfig) -> None:
    broker = SimBroker(CostModel(app_config.charges), slippage_override_bps=5.0)
    fills = broker.execute(
        [_intent("BBB", Side.BUY, 100, 5_000.0)], bars_today=_bars(), positions={}, asof=date(2024, 1, 2)
    )
    assert fills[0].missed and "volume" in fills[0].missed_reason


def test_sell_uses_adjusted_position_value_and_one_dp_charge(app_config: AppConfig) -> None:
    broker = SimBroker(CostModel(app_config.charges), slippage_override_bps=0.0)
    pos = Position(
        symbol="AAA",
        qty=1000,
        entry_price=80.0,
        entry_adj_close=72.0,
        entry_date=date(2023, 12, 1),
        entry_index=0,
        entry_notional=80_000.0,
    )
    fills = broker.execute(
        [_intent("AAA", Side.SELL, 500, 0.0), _intent("AAA", Side.SELL, 500, 0.0)],
        bars_today=_bars(),
        positions={"AAA": pos},
        asof=date(2024, 1, 2),
    )
    # value now = 80,000 * 90/72 = 100,000; half of it per intent
    assert fills[0].notional == pytest.approx(500 * 100.0)  # qty * raw close for reporting
    assert fills[0].costs.stt == pytest.approx(0.001 * 50_000.0)
    assert fills[0].costs.dp_charge == pytest.approx(15.34)
    assert fills[1].costs.dp_charge == 0.0  # same scrip, same day


def test_portfolio_t_plus_one_settlement_and_overdraw_guard(app_config: AppConfig) -> None:
    cm = CostModel(app_config.charges, product=Product.CNC)
    broker = SimBroker(cm, slippage_override_bps=0.0)
    p = Portfolio(150_000.0)
    bars = _bars()
    buy = broker.execute(
        [_intent("AAA", Side.BUY, 1000, 100_000.0)], bars_today=bars, positions={}, asof=date(2024, 1, 2)
    )[0]
    p.apply_fill(buy, adj_close_today=90.0, asof=date(2024, 1, 2), asof_index=0)
    assert p.cash_settled == pytest.approx(150_000.0 - 100_000.0 - buy.costs.total)
    # next day the stock is up 10% (adjusted); sell everything
    p.settle()
    values = p.mark({"AAA": 99.0})
    assert values["AAA"] == pytest.approx(110_000.0)
    sell = broker.execute(
        [_intent("AAA", Side.SELL, 1000, 0.0)],
        bars_today=bars.assign(adj_close=[99.0, 50.0, 200.0]),
        positions=p.positions,
        asof=date(2024, 1, 3),
    )[0]
    p.apply_fill(sell, adj_close_today=99.0, asof=date(2024, 1, 3), asof_index=1)
    assert p.positions == {}
    assert p.cash_unsettled == pytest.approx(110_000.0 - sell.costs.total)
    # unsettled proceeds cannot fund a purchase the same day
    big = broker.execute(
        [_intent("CCC", Side.BUY, 700, 140_000.0)], bars_today=bars, positions={}, asof=date(2024, 1, 3)
    )[0]
    with pytest.raises(InsufficientCashError):
        p.apply_fill(big, adj_close_today=200.0, asof=date(2024, 1, 3), asof_index=1)
    p.settle()
    p.apply_fill(big, adj_close_today=200.0, asof=date(2024, 1, 4), asof_index=2)
    equity = p.equity(p.mark({"CCC": 200.0}))
    assert equity == pytest.approx(
        150_000.0 + 10_000.0 - buy.costs.total - sell.costs.total - big.costs.total
    )
