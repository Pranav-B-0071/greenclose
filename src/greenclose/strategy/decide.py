"""THE decision function (brief invariant 7).

`decide(asof, features, predictions, portfolio, config) -> list[Intent]` is called identically by the
backtest engine and the forward-prediction runner. There is no second copy anywhere.

Inputs are one day's cross-section:
- `features`: today's rows indexed by symbol with close, adj_close, volume, traded_value_20d, vol_20d,
  sector, cluster_id, staleness_flag
- `predictions`: indexed by symbol with `score` and, optionally, `expected_return_bps` (the
  regressor head used by the abstention gate) and probability columns

Pipeline: tradeable filter -> rank -> sector/cluster-constrained top-K -> abstention -> overlap-aware
hold/sell with a minimum hold period -> sizing with caps and settled-cash check -> SELLs then BUYs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from greenclose.domain.models import Intent, PortfolioState, Product, Side

REQUIRED_FEATURES = (
    "close",
    "adj_close",
    "volume",
    "traded_value_20d",
    "vol_20d",
    "sector",
    "cluster_id",
    "staleness_flag",
)


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    k: int = 3
    select_from: str = "top"  # top | bottom
    sizing: str = "inverse_vol"  # inverse_vol | equal
    position_size: float = 100_000.0
    min_position: float = 50_000.0
    max_position_pct: float = 0.40
    liquidity_cap_bps_adv: float = 10.0
    sector_max: int = 1  # 0 = unlimited
    cluster_max: int = 1  # 0 = unlimited
    rebalance_mode: str = "overlap"  # overlap | fixed
    rebalance_frequency: str = "daily"  # daily | monthly
    rebalance_band: float = 1.0  # resize a kept name only if |target - value| > band * target; 1.0 = never
    hold_days: int = 1
    abstain_enabled: bool = True
    abstain_buffer_bps: float = 5.0
    round_trip_cost_bps: float = 34.0
    cash_buffer_pct: float = 0.02
    product: Product = Product.CNC

    def with_(self, **changes: object) -> DecisionConfig:
        return replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class _Row:
    symbol: str
    rank: int
    score: float
    close: float
    vol_20d: float
    traded_value_20d: float
    sector: str
    cluster: str
    tradeable: bool
    edge_bps: float | None


def _rows(features: pd.DataFrame, predictions: pd.DataFrame, config: DecisionConfig) -> list[_Row]:
    """Today's cross-section as plain rows, ordered exactly as the selection rule wants to see it.

    Deliberately free of pandas object churn: a join, an assign and a `sort_values` on a 48-row frame
    cost more than everything else in the decision layer put together, and this runs once per
    simulated day across thousands of runs. Extraction is by column to numpy, the ordering is a
    stable sort over indices, and NaN scores sort last exactly as pandas would place them.
    """
    missing = [c for c in REQUIRED_FEATURES if c not in features.columns]
    if missing:
        raise KeyError(f"features missing columns {missing}")
    if "score" not in predictions.columns:
        raise KeyError("predictions must have a 'score' column")

    index = features.index
    syms = [str(x) for x in index.to_numpy()]

    def col(frame: pd.DataFrame, name: str, *, fill: float | None = None) -> np.ndarray:
        series = frame[name] if name in frame.columns else None
        if series is None:
            return np.full(len(index), np.nan)
        arr = series.to_numpy(dtype="float64", na_value=np.nan)
        return arr if fill is None else np.nan_to_num(arr, nan=fill)

    aligned = predictions if predictions.index.equals(index) else predictions.reindex(index)
    score = col(aligned, "score")
    edge_arr = col(aligned, "expected_return_bps") if "expected_return_bps" in aligned.columns else None
    close = col(features, "close")
    vol20 = col(features, "vol_20d")
    tv20 = col(features, "traded_value_20d")
    stale = col(features, "staleness_flag", fill=1.0)
    volume = col(features, "volume", fill=0.0)
    cluster_id = col(features, "cluster_id")
    sector = [str(x) for x in features["sector"].to_numpy()]

    # same ordering as sort_values(["score", symbol], ascending=[asc, True], na_position="last")
    sign = 1.0 if config.select_from == "bottom" else -1.0
    order = sorted(
        range(len(index)),
        key=lambda i: (1, 0.0, syms[i]) if math.isnan(score[i]) else (0, sign * score[i], syms[i]),
    )

    out: list[_Row] = []
    rank = 0
    inverse_vol = config.sizing == "inverse_vol"
    for i in order:
        sc = float(score[i])
        cl = float(close[i])
        vl = float(vol20[i])
        has_score = not math.isnan(sc)
        tradeable = (
            has_score
            and not math.isnan(cl)
            and cl > 0
            and volume[i] > 0
            and stale[i] == 0.0
            and (not inverse_vol or (not math.isnan(vl) and vl > 0))
        )
        if has_score:
            rank += 1
        cid = float(cluster_id[i])
        cluster = f"solo:{syms[i]}" if math.isnan(cid) else str(int(cid))
        edge: float | None = None
        if edge_arr is not None:
            e = float(edge_arr[i])
            if not math.isnan(e):
                edge = e
        out.append(
            _Row(
                symbol=syms[i],
                rank=rank if has_score else 0,
                score=sc,
                close=cl,
                vol_20d=vl,
                traded_value_20d=float(tv20[i]),
                sector=sector[i],
                cluster=cluster,
                tradeable=bool(tradeable),
                edge_bps=edge,
            )
        )
    return out


def _select_targets(rows: list[_Row], config: DecisionConfig) -> tuple[list[_Row], dict[str, str]]:
    """Greedy constrained top-K in rank order. Returns (targets, reason per skipped candidate)."""
    targets: list[_Row] = []
    reasons: dict[str, str] = {}
    sector_count: dict[str, int] = {}
    cluster_count: dict[str, int] = {}
    for r in rows:
        if len(targets) >= config.k:
            break
        if not r.tradeable:
            continue
        if config.sector_max > 0 and sector_count.get(r.sector, 0) >= config.sector_max:
            reasons[r.symbol] = f"sector cap ({r.sector})"
            continue
        if (
            config.cluster_max > 0
            and not r.cluster.startswith("solo:")
            and cluster_count.get(r.cluster, 0) >= config.cluster_max
        ):
            reasons[r.symbol] = f"cluster cap ({r.cluster})"
            continue
        targets.append(r)
        sector_count[r.sector] = sector_count.get(r.sector, 0) + 1
        cluster_count[r.cluster] = cluster_count.get(r.cluster, 0) + 1
    return targets, reasons


def _target_notionals(targets: list[_Row], equity: float, config: DecisionConfig) -> dict[str, float]:
    if not targets:
        return {}
    if config.sizing == "inverse_vol":
        inv = {r.symbol: 1.0 / r.vol_20d for r in targets}
        mean_inv = sum(inv.values()) / len(inv)
        base = {s: config.position_size * v / mean_inv for s, v in inv.items()}
    else:
        per = min(config.position_size, equity * (1.0 - config.cash_buffer_pct) / max(1, config.k))
        base = {r.symbol: per for r in targets}
    out: dict[str, float] = {}
    for r in targets:
        cap = config.max_position_pct * equity
        if not math.isnan(r.traded_value_20d) and r.traded_value_20d > 0:
            cap = min(cap, config.liquidity_cap_bps_adv / 1e4 * r.traded_value_20d)
        out[r.symbol] = min(base[r.symbol], cap)
    return out


def decide_with_trace(
    asof: date,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    portfolio: PortfolioState,
    config: DecisionConfig,
) -> tuple[list[Intent], pd.DataFrame]:
    rows = _rows(features, predictions, config)
    by_symbol = {r.symbol: r for r in rows}
    targets, skip_reasons = _select_targets(rows, config)

    # abstention: only meaningful when a return-magnitude head is available
    threshold = config.round_trip_cost_bps + config.abstain_buffer_bps
    abstained: dict[str, str] = {}
    if config.abstain_enabled:
        kept: list[_Row] = []
        for r in targets:
            if r.edge_bps is not None and r.edge_bps < threshold:
                abstained[r.symbol] = f"edge {r.edge_bps:.1f} bps < cost {threshold:.1f} bps"
            else:
                kept.append(r)
        targets = kept
    target_symbols = {r.symbol for r in targets}

    rebalance_day = config.rebalance_frequency == "daily" or portfolio.is_first_day_of_month
    notionals = _target_notionals(targets, portfolio.equity, config)

    sells: list[Intent] = []
    buys: list[Intent] = []
    actions: dict[str, str] = {}
    reasons: dict[str, str] = dict(skip_reasons)
    reasons.update(abstained)

    # ---- holdings: keep, sell, or resize
    kept_symbols: set[str] = set()
    for pos in portfolio.positions:
        sym = pos.symbol
        held_days = portfolio.asof_index - pos.entry_index
        if held_days < config.hold_days:
            actions[sym] = "HOLD"
            reasons[sym] = f"minimum hold ({held_days}/{config.hold_days} days)"
            kept_symbols.add(sym)
            continue
        if not rebalance_day:
            actions[sym] = "HOLD"
            reasons[sym] = "not a rebalance day"
            kept_symbols.add(sym)
            continue
        if config.rebalance_mode == "overlap" and sym in target_symbols:
            kept_symbols.add(sym)
            actions[sym] = "HOLD"
            reasons[sym] = "still in top-K (overlap-aware)"
            row = by_symbol.get(sym)
            value = portfolio.values.get(sym, 0.0)
            target = notionals.get(sym, 0.0)
            if (
                row is not None
                and row.tradeable
                and target > 0
                and abs(target - value) > config.rebalance_band * target
            ):
                diff = target - value
                qty = int(math.floor(abs(diff) / row.close))
                if qty > 0 and diff < 0 and qty < pos.qty:
                    sells.append(
                        _intent(asof, sym, Side.SELL, qty, qty * row.close, "resize down to target", config)
                    )
                    actions[sym] = "SELL_PART"
                elif qty > 0 and diff > 0:
                    buys.append(
                        _intent(asof, sym, Side.BUY, qty, qty * row.close, "resize up to target", config)
                    )
                    actions[sym] = "BUY_ADD"
            continue
        row = by_symbol.get(sym)
        if row is None or math.isnan(row.close) or row.close <= 0:
            actions[sym] = "HOLD"
            reasons[sym] = "no tradeable bar today; exit deferred"
            kept_symbols.add(sym)
            continue
        value = portfolio.values.get(sym, pos.entry_notional)
        why = "dropped out of top-K" if config.rebalance_mode == "overlap" else "fixed-hold exit"
        sells.append(_intent(asof, sym, Side.SELL, pos.qty, value, why, config))
        actions[sym] = "SELL"
        reasons[sym] = why

    # ---- new entries, in rank order, limited by settled cash (T+1: today's sale proceeds are not usable)
    # settled cash net of a reserve for the charges the broker adds on top of the notional
    cash = portfolio.cash_settled / (1.0 + config.round_trip_cost_bps / 1e4) - sum(i.notional for i in buys)
    if rebalance_day:
        for r in targets:
            if r.symbol in kept_symbols:
                continue
            notional = notionals.get(r.symbol, 0.0)
            if notional < config.min_position:
                reasons[r.symbol] = f"target {notional:,.0f} below minimum position after caps"
                actions[r.symbol] = "SKIP"
                continue
            if notional > cash:
                if cash >= config.min_position and cash > 0:
                    notional = cash
                else:
                    reasons[r.symbol] = "insufficient settled cash"
                    actions[r.symbol] = "SKIP"
                    continue
            qty = int(math.floor(notional / r.close))
            if qty <= 0:
                reasons[r.symbol] = "price above target notional"
                actions[r.symbol] = "SKIP"
                continue
            notional = qty * r.close
            cash -= notional
            buys.append(_intent(asof, r.symbol, Side.BUY, qty, notional, f"rank {r.rank} entry", config))
            actions[r.symbol] = "BUY"

    trace = pd.DataFrame(
        {
            "symbol": [r.symbol for r in rows],
            "rank": [r.rank for r in rows],
            "score": [r.score for r in rows],
            "tradeable": [r.tradeable for r in rows],
            "target": [r.symbol in target_symbols for r in rows],
            "abstained": [r.symbol in abstained for r in rows],
            "action": [actions.get(r.symbol, "") for r in rows],
            "reason": [reasons.get(r.symbol, "") for r in rows],
            "target_notional": [notionals.get(r.symbol, math.nan) for r in rows],
        }
    ).set_index("symbol")
    return sells + buys, trace


def decide(
    asof: date,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    portfolio: PortfolioState,
    config: DecisionConfig,
) -> list[Intent]:
    intents, _ = decide_with_trace(asof, features, predictions, portfolio, config)
    return intents


def _intent(
    asof: date, symbol: str, side: Side, qty: int, notional: float, reason: str, config: DecisionConfig
) -> Intent:
    return Intent(
        decision_id=f"{asof.isoformat()}-{symbol}-{side.value}",
        asof=asof,
        symbol=symbol,
        side=side,
        qty=qty,
        notional=float(notional),
        reason=reason,
        product=config.product,
    )
