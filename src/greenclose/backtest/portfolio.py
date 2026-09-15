"""Mutable portfolio ledger used inside the backtest; hands out immutable `PortfolioState` snapshots.

- positions are valued with the *adjusted* close (entry_notional * adj_t / adj_entry), so splits, bonuses
  and dividends are handled implicitly (a dividend is treated as reinvested; documented approximation)
- T+1 settlement: sale proceeds land in `cash_unsettled` and become spendable at the next `settle()`
- buys draw on settled cash only and never overdraw
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from greenclose.domain.models import Fill, PortfolioState, Position, Side

MAX_DAILY_MOVE = 0.25  # a position moving more than this in a day is logged as unexplained


class InsufficientCashError(RuntimeError):
    pass


class Portfolio:
    def __init__(self, capital: float) -> None:
        self.cash_settled = float(capital)
        self.cash_unsettled = 0.0
        self.positions: dict[str, Position] = {}
        self.last_values: dict[str, float] = {}

    # ---- daily cycle

    def settle(self) -> None:
        self.cash_settled += self.cash_unsettled
        self.cash_unsettled = 0.0

    def mark(self, adj_close: Mapping[str, float]) -> dict[str, float]:
        """Value each position at today's adjusted close; a symbol with no bar keeps its last value."""
        values: dict[str, float] = {}
        for sym, pos in self.positions.items():
            px = adj_close.get(sym)
            if px is None or px != px:  # NaN
                values[sym] = self.last_values.get(sym, pos.entry_notional)
            else:
                values[sym] = pos.value_at(float(px))
        return values

    def equity(self, values: Mapping[str, float]) -> float:
        return self.cash_settled + self.cash_unsettled + sum(values.values())

    def state(
        self, asof: date, asof_index: int, prev_asof: date | None, values: Mapping[str, float]
    ) -> PortfolioState:
        return PortfolioState(
            asof=asof,
            asof_index=asof_index,
            cash_settled=self.cash_settled,
            cash_unsettled=self.cash_unsettled,
            positions=tuple(self.positions[s] for s in sorted(self.positions)),
            equity=self.equity(values),
            prev_asof=prev_asof,
            values=dict(values),
        )

    # ---- fills

    def apply_fill(self, fill: Fill, *, adj_close_today: float, asof: date, asof_index: int) -> None:
        if fill.missed:
            return
        sym = fill.intent.symbol
        if fill.intent.side is Side.BUY:
            cost = fill.notional + fill.costs.total
            if cost > self.cash_settled + 1e-6:
                raise InsufficientCashError(
                    f"{asof} {sym}: need {cost:,.2f}, settled cash {self.cash_settled:,.2f}"
                )
            self.cash_settled -= cost
            existing = self.positions.get(sym)
            if existing is None:
                self.positions[sym] = Position(
                    symbol=sym,
                    qty=fill.qty,
                    entry_price=fill.price,
                    entry_adj_close=adj_close_today,
                    entry_date=asof,
                    entry_index=asof_index,
                    entry_notional=fill.notional,
                )
            else:
                value_now = existing.value_at(adj_close_today)
                total_qty = existing.qty + fill.qty
                self.positions[sym] = Position(
                    symbol=sym,
                    qty=total_qty,
                    entry_price=(value_now + fill.notional) / total_qty,
                    entry_adj_close=adj_close_today,
                    entry_date=existing.entry_date,
                    entry_index=existing.entry_index,
                    entry_notional=value_now + fill.notional,
                )
            self.last_values[sym] = self.positions[sym].value_at(adj_close_today)
        else:
            pos = self.positions[sym]
            frac = min(1.0, fill.qty / pos.qty)
            value_now = pos.value_at(adj_close_today)
            proceeds = frac * value_now - fill.costs.total
            self.cash_unsettled += proceeds
            if fill.qty >= pos.qty:
                del self.positions[sym]
                self.last_values.pop(sym, None)
            else:
                self.positions[sym] = Position(
                    symbol=sym,
                    qty=pos.qty - fill.qty,
                    entry_price=pos.entry_price,
                    entry_adj_close=pos.entry_adj_close,
                    entry_date=pos.entry_date,
                    entry_index=pos.entry_index,
                    entry_notional=pos.entry_notional * (1.0 - frac),
                )
                self.last_values[sym] = self.positions[sym].value_at(adj_close_today)

    def remember_values(self, values: Mapping[str, float]) -> None:
        self.last_values = {s: v for s, v in values.items() if s in self.positions}
