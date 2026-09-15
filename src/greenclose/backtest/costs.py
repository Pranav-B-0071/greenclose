"""Itemised Indian transaction-cost model (brief section 8). Never a single blended constant.

Every rate comes from config/charges.yaml. GST applies to brokerage + exchange transaction charge +
SEBI + IPFT (the base that reproduces the brief's reference table to the paisa). The DP charge is a
flat amount per scrip per SELL day for delivery, so selling three names costs three DP charges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from greenclose.config import ChargesCfg
from greenclose.domain.models import CostBreakdown, Product, Side

TRADING_DAYS = 250.0


@dataclass(frozen=True, slots=True)
class CostModel:
    charges: ChargesCfg
    product: Product = Product.CNC
    cost_multiplier: float = 1.0

    # ---- one leg

    def leg(
        self,
        side: Side,
        notional: float,
        *,
        slippage_bps: float,
        first_sell_of_scrip_today: bool = True,
    ) -> CostBreakdown:
        if notional <= 0:
            return CostBreakdown()
        m = self.cost_multiplier
        is_buy = side is Side.BUY
        if self.product is Product.CNC:
            cnc = self.charges.delivery_cnc
            brokerage = cnc.brokerage_pct * notional
            dp = cnc.dp_charge_flat if (not is_buy and first_sell_of_scrip_today) else 0.0
            stt_pct = cnc.stt_buy_pct if is_buy else cnc.stt_sell_pct
            stamp_pct, exch_pct = cnc.stamp_buy_pct, cnc.exchange_txn_pct
            sebi_pct, ipft_pct, gst_pct = cnc.sebi_pct, cnc.ipft_pct, cnc.gst_pct
        else:
            mis = self.charges.intraday_mis
            brokerage = min(mis.brokerage_per_order.pct * notional, mis.brokerage_per_order.cap)
            dp = 0.0
            stt_pct = mis.stt_buy_pct if is_buy else mis.stt_sell_pct
            stamp_pct, exch_pct = mis.stamp_buy_pct, mis.exchange_txn_pct
            sebi_pct, ipft_pct, gst_pct = mis.sebi_pct, mis.ipft_pct, mis.gst_pct
        stt = stt_pct * notional
        stamp = stamp_pct * notional if is_buy else 0.0
        exchange_txn = exch_pct * notional
        sebi = sebi_pct * notional
        ipft = ipft_pct * notional
        gst = gst_pct * (brokerage + exchange_txn + sebi + ipft)
        slippage = slippage_bps / 1e4 * notional
        return CostBreakdown(
            brokerage=brokerage * m,
            stt=stt * m,
            stamp=stamp * m,
            exchange_txn=exchange_txn * m,
            sebi=sebi * m,
            ipft=ipft * m,
            gst=gst * m,
            dp_charge=dp * m,
            slippage=slippage * m,
        )

    # ---- round trip

    def round_trip(self, notional: float, *, slippage_bps_per_leg: float) -> CostBreakdown:
        return self.leg(Side.BUY, notional, slippage_bps=slippage_bps_per_leg) + self.leg(
            Side.SELL, notional, slippage_bps=slippage_bps_per_leg
        )

    def round_trip_bps(self, notional: float, *, slippage_bps_per_leg: float) -> float:
        return self.round_trip(notional, slippage_bps_per_leg=slippage_bps_per_leg).total / notional * 1e4

    # ---- slippage

    def slippage_bps(
        self,
        *,
        order_value: float,
        adv_value: float | None,
        daily_vol: float | None,
        high_vol_regime: bool = False,
    ) -> float:
        """bps = base + coef * daily_vol_bps * sqrt(order_value / adv_value), floored at min_bps."""
        s = self.charges.slippage
        bps = s.base_bps
        if adv_value and adv_value > 0 and daily_vol is not None and not math.isnan(daily_vol):
            bps += s.impact_coef * (daily_vol * 1e4) * math.sqrt(order_value / adv_value)
        bps = max(bps, s.min_bps)
        if high_vol_regime:
            bps *= s.high_vol_multiplier
        return bps


def break_even_gross_alpha_bps(
    *, round_trip_bps: float, daily_sigma: float = 0.018, target_sharpe: float = 1.0
) -> float:
    """Gross daily alpha per trade needed for `target_sharpe` net, on a one-position-per-day strategy.

    required net daily mean = target_sharpe * sigma / sqrt(250); gross = net + round-trip cost.
    Brief section 8.2: ~45 bps for sigma 1.8%, Sharpe 1.0 and a 34 bps delivery round trip.
    """
    net_bps = target_sharpe * daily_sigma / math.sqrt(TRADING_DAYS) * 1e4
    return net_bps + round_trip_bps
