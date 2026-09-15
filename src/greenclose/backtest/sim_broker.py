"""Fill simulator (brief section 9.2). The only broker this project will ever have.

- reference price: the day's raw close, a documented proxy for the closing-auction equilibrium price
- slippage: from config/charges.yaml (or a fixed override), always adverse, booked as a cost line
- market orders always fill; a zero-volume or missing bar does not fill and is logged as a missed trade
- sell proceeds and sell-side charges are computed on the position's *adjusted* value (dividends, splits)
- the DP charge is levied once per scrip per sell day
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd

from greenclose.backtest.costs import CostModel
from greenclose.domain.models import CostBreakdown, Fill, Intent, Position, Side


class SimBroker:
    def __init__(self, cost_model: CostModel, *, slippage_override_bps: float | None = None) -> None:
        self.cost_model = cost_model
        self.slippage_override_bps = slippage_override_bps

    def execute(
        self,
        intents: Sequence[Intent],
        *,
        bars_today: pd.DataFrame,
        positions: Mapping[str, Position],
        asof: date,
        high_vol_regime: bool = False,
    ) -> list[Fill]:
        fills: list[Fill] = []
        sold_today: set[str] = set()
        for intent in intents:
            sym = intent.symbol
            if sym not in bars_today.index:
                fills.append(self._missed(intent, asof, "no bar today"))
                continue
            row = bars_today.loc[sym]
            close = float(row["close"])
            volume = float(row["volume"]) if pd.notna(row["volume"]) else 0.0
            if math.isnan(close) or close <= 0 or volume <= 0:
                fills.append(self._missed(intent, asof, "zero volume / no trade today"))
                continue
            adv = float(row["traded_value_20d"]) if pd.notna(row.get("traded_value_20d")) else None
            vol = float(row["vol_20d"]) if pd.notna(row.get("vol_20d")) else None

            if intent.side is Side.BUY:
                qty = intent.qty
                notional = qty * close
            else:
                pos = positions.get(sym)
                if pos is None:
                    fills.append(self._missed(intent, asof, "no position to sell"))
                    continue
                qty = min(intent.qty, pos.qty)
                adj = float(row["adj_close"])
                notional = (qty / pos.qty) * pos.value_at(adj)
            if qty <= 0 or notional <= 0:
                fills.append(self._missed(intent, asof, "zero quantity"))
                continue

            slip = (
                self.slippage_override_bps
                if self.slippage_override_bps is not None
                else self.cost_model.slippage_bps(
                    order_value=notional, adv_value=adv, daily_vol=vol, high_vol_regime=high_vol_regime
                )
            )
            costs = self.cost_model.leg(
                intent.side,
                notional,
                slippage_bps=slip,
                first_sell_of_scrip_today=(sym not in sold_today),
            )
            if intent.side is Side.SELL:
                sold_today.add(sym)
            fills.append(
                Fill(
                    intent=intent,
                    date=asof,
                    qty=qty,
                    price=close,
                    ref_price=close,
                    slippage_bps=slip,
                    costs=costs,
                )
            )
        return fills

    @staticmethod
    def _missed(intent: Intent, asof: date, reason: str) -> Fill:
        return Fill(
            intent=intent,
            date=asof,
            qty=0,
            price=math.nan,
            ref_price=math.nan,
            slippage_bps=0.0,
            costs=CostBreakdown(),
            missed=True,
            missed_reason=reason,
        )
