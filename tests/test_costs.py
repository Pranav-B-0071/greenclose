"""Reference values from brief section 8.1. If the implementation disagrees, the implementation is wrong."""

from __future__ import annotations

import pytest

from greenclose.backtest.costs import CostModel, break_even_gross_alpha_bps
from greenclose.config import AppConfig
from greenclose.domain.models import Product, Side

NOTIONAL = 100_000.0
SLIP = 5.0  # bps per leg


def test_cnc_round_trip_to_the_paisa(app_config: AppConfig) -> None:
    cm = CostModel(app_config.charges, product=Product.CNC)
    rt = cm.round_trip(NOTIONAL, slippage_bps_per_leg=SLIP)
    assert rt.brokerage == pytest.approx(0.00, abs=0.005)
    assert rt.stt == pytest.approx(200.00, abs=0.005)
    assert rt.stamp == pytest.approx(15.00, abs=0.005)
    assert rt.exchange_txn == pytest.approx(6.14, abs=0.005)
    assert rt.sebi + rt.ipft == pytest.approx(0.40, abs=0.005)
    assert rt.gst == pytest.approx(1.18, abs=0.005)
    assert rt.dp_charge == pytest.approx(15.34, abs=0.005)
    assert rt.statutory == pytest.approx(238.06, abs=0.005)
    assert rt.slippage == pytest.approx(100.00, abs=0.005)
    assert rt.total == pytest.approx(338.06, abs=0.005)
    assert cm.round_trip_bps(NOTIONAL, slippage_bps_per_leg=SLIP) == pytest.approx(33.806, abs=0.001)


def test_mis_round_trip_to_the_paisa(app_config: AppConfig) -> None:
    cm = CostModel(app_config.charges, product=Product.MIS)
    rt = cm.round_trip(NOTIONAL, slippage_bps_per_leg=SLIP)
    assert rt.brokerage == pytest.approx(40.00, abs=0.005)
    assert rt.stt == pytest.approx(25.00, abs=0.005)
    assert rt.stamp == pytest.approx(3.00, abs=0.005)
    assert rt.exchange_txn == pytest.approx(6.14, abs=0.005)
    assert rt.sebi + rt.ipft == pytest.approx(0.40, abs=0.005)
    assert rt.gst == pytest.approx(8.38, abs=0.005)
    assert rt.dp_charge == pytest.approx(0.00, abs=0.005)
    assert rt.statutory == pytest.approx(82.92, abs=0.005)
    assert rt.total == pytest.approx(182.92, abs=0.005)


def test_dp_charge_is_per_scrip_per_sell_day(app_config: AppConfig) -> None:
    cm = CostModel(app_config.charges, product=Product.CNC)
    first = cm.leg(Side.SELL, NOTIONAL, slippage_bps=0.0, first_sell_of_scrip_today=True)
    second = cm.leg(Side.SELL, NOTIONAL, slippage_bps=0.0, first_sell_of_scrip_today=False)
    assert first.dp_charge == pytest.approx(15.34)
    assert second.dp_charge == 0.0
    buy = cm.leg(Side.BUY, NOTIONAL, slippage_bps=0.0)
    assert buy.dp_charge == 0.0 and buy.stamp > 0


def test_mis_brokerage_is_capped_per_order(app_config: AppConfig) -> None:
    cm = CostModel(app_config.charges, product=Product.MIS)
    small = cm.leg(Side.BUY, 10_000.0, slippage_bps=0.0)
    assert small.brokerage == pytest.approx(3.00)  # 0.03% of 10,000 < Rs 20 cap
    big = cm.leg(Side.BUY, 1_000_000.0, slippage_bps=0.0)
    assert big.brokerage == pytest.approx(20.00)


def test_cost_multiplier_scales_everything(app_config: AppConfig) -> None:
    doubled = CostModel(app_config.charges, product=Product.CNC, cost_multiplier=2.0)
    zero = CostModel(app_config.charges, product=Product.CNC, cost_multiplier=0.0)
    assert doubled.round_trip(NOTIONAL, slippage_bps_per_leg=SLIP).total == pytest.approx(
        2 * 338.06, abs=0.01
    )
    assert zero.round_trip(NOTIONAL, slippage_bps_per_leg=SLIP).total == 0.0


def test_slippage_formula(app_config: AppConfig) -> None:
    cm = CostModel(app_config.charges)
    assert cm.slippage_bps(order_value=1e5, adv_value=None, daily_vol=None) == pytest.approx(2.0)
    # base 2 + 0.5 * 180 bps * sqrt(1e5 / 1e9) = 2 + 0.5 * 180 * 0.01 = 2.9
    assert cm.slippage_bps(order_value=1e5, adv_value=1e9, daily_vol=0.018) == pytest.approx(2.9)
    assert cm.slippage_bps(
        order_value=1e5, adv_value=1e9, daily_vol=0.018, high_vol_regime=True
    ) == pytest.approx(5.8)


def test_break_even_matches_brief() -> None:
    # 1.8%/sqrt(250) = 0.1138% net + 0.338% round trip -> ~45 bps gross
    assert break_even_gross_alpha_bps(round_trip_bps=33.806) == pytest.approx(45.19, abs=0.05)
