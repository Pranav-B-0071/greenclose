"""Typed domain objects. Dataclasses only: no I/O, no pandas, no side effects.

Everything the backtest engine, the strategy layer and the forward runner exchange is one of these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class Product(StrEnum):
    CNC = "CNC"  # delivery
    MIS = "MIS"  # intraday


@dataclass(frozen=True, slots=True)
class Bar:
    date: date
    symbol: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Itemised charges for one leg, in rupees. `slippage` is kept separate from statutory charges."""

    brokerage: float = 0.0
    stt: float = 0.0
    stamp: float = 0.0
    exchange_txn: float = 0.0
    sebi: float = 0.0
    ipft: float = 0.0
    gst: float = 0.0
    dp_charge: float = 0.0
    slippage: float = 0.0

    @property
    def statutory(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.stamp
            + self.exchange_txn
            + self.sebi
            + self.ipft
            + self.gst
            + self.dp_charge
        )

    @property
    def total(self) -> float:
        return self.statutory + self.slippage

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            brokerage=self.brokerage + other.brokerage,
            stt=self.stt + other.stt,
            stamp=self.stamp + other.stamp,
            exchange_txn=self.exchange_txn + other.exchange_txn,
            sebi=self.sebi + other.sebi,
            ipft=self.ipft + other.ipft,
            gst=self.gst + other.gst,
            dp_charge=self.dp_charge + other.dp_charge,
            slippage=self.slippage + other.slippage,
        )


@dataclass(frozen=True, slots=True)
class Intent:
    """What the strategy wants to do. Produced only by `strategy.decide`."""

    decision_id: str
    asof: date
    symbol: str
    side: Side
    qty: int
    notional: float
    reason: str
    product: Product = Product.CNC


@dataclass(frozen=True, slots=True)
class Fill:
    """What the simulated broker did with an intent. `missed=True` means it did not fill."""

    intent: Intent
    date: date
    qty: int
    price: float
    ref_price: float
    slippage_bps: float
    costs: CostBreakdown
    missed: bool = False
    missed_reason: str = ""

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass(frozen=True, slots=True)
class Position:
    """An open long position.

    Value evolves with the *adjusted* close so splits, bonuses and dividends are handled implicitly:
    value_t = entry_notional * adj_close_t / entry_adj_close.
    """

    symbol: str
    qty: int
    entry_price: float
    entry_adj_close: float
    entry_date: date
    entry_index: int  # trading-day ordinal of entry, for hold-period rules
    entry_notional: float

    def value_at(self, adj_close: float) -> float:
        return self.entry_notional * adj_close / self.entry_adj_close


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Immutable view of the portfolio handed to `strategy.decide`."""

    asof: date
    asof_index: int
    cash_settled: float
    cash_unsettled: float
    positions: tuple[Position, ...] = field(default_factory=tuple)
    equity: float = 0.0

    @property
    def held_symbols(self) -> frozenset[str]:
        return frozenset(p.symbol for p in self.positions)

    def position(self, symbol: str) -> Position | None:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None


@dataclass(frozen=True, slots=True)
class Prediction:
    """One row of the forward log: one symbol, one day."""

    date: date
    symbol: str
    model_version: str
    score: float
    rank: int
    prob_green: float | None
    prob_beat_mean: float | None
    expected_return_bps: float | None
    selected: bool
    abstained: bool
    abstain_reason: str
