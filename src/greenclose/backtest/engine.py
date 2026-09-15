"""Walk-forward backtest engine (brief section 9.1).

    for T in dates:
        settle T+1 cash; mark positions at today's adjusted close
        predictions = scorer.score(T, features_T)
        intents     = strategy.decide(T, features_T, predictions, portfolio, config)   # SHARED
        fills       = sim_broker.execute(intents, bars_T, costs, slippage)
        portfolio.apply(fills); log every symbol, not just the traded ones

Features come from one `build(cutoff=last_date)` call sliced per day, which is equivalent to building at
every T by the truncation-invariance guarantee (section 13.2). The output is a full replay, not a return
series.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
import pandas as pd

from greenclose.backtest.costs import CostModel
from greenclose.backtest.portfolio import InsufficientCashError, Portfolio
from greenclose.backtest.sim_broker import SimBroker
from greenclose.config import AppConfig
from greenclose.domain.models import CostBreakdown, Fill, Product, Side
from greenclose.ops.provenance import Provenance
from greenclose.strategy.decide import DecisionConfig, decide_with_trace

DAY_COLUMNS = [
    "close",
    "adj_close",
    "volume",
    "traded_value_20d",
    "vol_20d",
    "sector",
    "cluster_id",
    "staleness_flag",
]


# ----------------------------------------------------------------------------- scorers


class Scorer(Protocol):
    name: str

    def score(self, asof: pd.Timestamp, feats_today: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class FeatureScorer:
    """Score = sign * a raw feature column (single-feature sorts, B3/B4/B5)."""

    column: str
    sign: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or f"feature:{'+' if self.sign > 0 else '-'}{self.column}"

    def score(self, asof: pd.Timestamp, feats_today: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"score": self.sign * feats_today[self.column].astype("float64")})


@dataclass
class ConstantScorer:
    """Every symbol scores 0: selection falls back to symbol order (equal-weight universe, B2)."""

    name: str = "constant"

    def score(self, asof: pd.Timestamp, feats_today: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"score": np.zeros(len(feats_today))}, index=feats_today.index)


@dataclass
class RandomScorer:
    """Uniform random scores; deterministic for a given seed and date order (B0)."""

    seed: int
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or f"random:{self.seed}"
        self._rng = np.random.default_rng(self.seed)

    def score(self, asof: pd.Timestamp, feats_today: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"score": self._rng.random(len(feats_today))}, index=feats_today.index)


@dataclass
class FrameScorer:
    """Scores read from a precomputed predictions frame indexed by (date, symbol) (model runs, sweeps)."""

    predictions: pd.DataFrame
    name: str = "frame"

    def score(self, asof: pd.Timestamp, feats_today: pd.DataFrame) -> pd.DataFrame:
        if asof not in self.predictions.index.get_level_values("date"):
            return pd.DataFrame({"score": np.full(len(feats_today), np.nan)}, index=feats_today.index)
        day = self.predictions.xs(asof, level="date")
        return day.reindex(feats_today.index)


# ----------------------------------------------------------------------------- panel


@dataclass
class DayPanel:
    dates: pd.DatetimeIndex
    index_of: dict[pd.Timestamp, int]
    by_date: dict[pd.Timestamp, pd.DataFrame]
    high_vol: pd.Series

    @classmethod
    def from_features(cls, features: pd.DataFrame, nifty_vol_col: str = "nifty_vol_20d") -> DayPanel:
        dates = pd.DatetimeIndex(features.index.get_level_values("date").unique()).sort_values()
        by_date = {pd.Timestamp(d): g.droplevel("date") for d, g in features.groupby(level="date", sort=True)}
        if nifty_vol_col in features.columns:
            nv = features[nifty_vol_col].groupby(level="date").first().reindex(dates)
            threshold = nv.shift(1).expanding(min_periods=250).quantile(0.9)  # top decile, no look-ahead
            high_vol = (nv >= threshold).fillna(False)
        else:
            high_vol = pd.Series(False, index=dates)
        return cls(
            dates=dates,
            index_of={pd.Timestamp(d): i for i, d in enumerate(dates)},
            by_date=by_date,
            high_vol=high_vol,
        )

    def oos_dates(self, *, min_train_years: int, start: str, holdout_start: str) -> pd.DatetimeIndex:
        """Identical evaluation window for every run: first month after the minimum training window
        until the locked holdout begins."""
        first = pd.Timestamp(start) + pd.DateOffset(years=min_train_years)
        first = first.replace(day=1)
        return self.dates[(self.dates >= first) & (self.dates < pd.Timestamp(holdout_start))]


# ----------------------------------------------------------------------------- run


@dataclass
class RunSpec:
    name: str
    scorer: Scorer
    decision: DecisionConfig
    dates: pd.DatetimeIndex
    description: str = ""
    family: str = "baseline"
    product: Product = Product.CNC
    cost_multiplier: float = 1.0
    slippage_override_bps: float | None = None
    capital: float = 1_000_000.0
    seed: int = 0
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class RunResult:
    spec: RunSpec
    daily: pd.DataFrame
    replay: pd.DataFrame
    fills: pd.DataFrame
    provenance: Provenance | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def equity(self) -> pd.Series:
        return self.daily.set_index("date")["equity"]


def _sum_costs(fills: list[Fill]) -> CostBreakdown:
    total = CostBreakdown()
    for f in fills:
        if not f.missed:
            total = total + f.costs
    return total


def run_backtest(
    spec: RunSpec, panel: DayPanel, cfg: AppConfig, provenance: Provenance | None = None
) -> RunResult:
    cost_model = CostModel(cfg.charges, product=spec.product, cost_multiplier=spec.cost_multiplier)
    broker = SimBroker(cost_model, slippage_override_bps=spec.slippage_override_bps)
    portfolio = Portfolio(spec.capital)
    prev_equity = spec.capital
    prev_asof: pd.Timestamp | None = None
    daily_rows: list[dict[str, object]] = []
    replay_parts: list[pd.DataFrame] = []
    fill_rows: list[dict[str, object]] = []

    for asof in spec.dates:
        day = panel.by_date[asof]
        idx = panel.index_of[asof]
        portfolio.settle()
        adj = day["adj_close"].to_dict()
        values = portfolio.mark(adj)
        mtm_pnl = sum(values.values()) - sum(portfolio.last_values.values())
        preds = spec.scorer.score(asof, day)
        state = portfolio.state(asof.date(), idx, prev_asof.date() if prev_asof is not None else None, values)
        intents, trace = decide_with_trace(asof.date(), day, preds, state, spec.decision)
        fills = broker.execute(
            intents,
            bars_today=day,
            positions=portfolio.positions,
            asof=asof.date(),
            high_vol_regime=bool(panel.high_vol.get(asof, False)),
        )
        applied: list[Fill] = []
        for f in fills:
            try:
                portfolio.apply_fill(
                    f,
                    adj_close_today=float(adj.get(f.intent.symbol, np.nan)),
                    asof=asof.date(),
                    asof_index=idx,
                )
                applied.append(f)
            except InsufficientCashError as exc:  # never spend money the account does not have
                applied.append(
                    replace(
                        f,
                        qty=0,
                        costs=CostBreakdown(),
                        missed=True,
                        missed_reason=f"insufficient settled cash: {exc}",
                    )
                )
        fills = applied
        values = portfolio.mark(adj)
        portfolio.remember_values(values)
        costs = _sum_costs(fills)
        equity = portfolio.equity(values)
        net_pnl = equity - prev_equity
        traded = sum(f.notional for f in fills if not f.missed)
        daily_rows.append(
            {
                "date": asof,
                "equity": equity,
                "cash_settled": portfolio.cash_settled,
                "cash_unsettled": portfolio.cash_unsettled,
                "n_positions": len(portfolio.positions),
                "position_value_total": sum(values.values()),
                "gross_pnl": mtm_pnl,
                "net_pnl": net_pnl,
                "costs_total": costs.total,
                "costs_statutory": costs.statutory,
                "costs_slippage": costs.slippage,
                "costs_dp": costs.dp_charge,
                "costs_stt": costs.stt,
                "turnover_notional": traded,
                "n_buys": sum(1 for f in fills if not f.missed and f.intent.side is Side.BUY),
                "n_sells": sum(1 for f in fills if not f.missed and f.intent.side is Side.SELL),
                "n_missed": sum(1 for f in fills if f.missed),
                "n_targets": int(trace["target"].sum()),
                "n_abstained": int(trace["abstained"].sum()),
                "high_vol_regime": bool(panel.high_vol.get(asof, False)),
            }
        )
        fill_by_symbol = {f.intent.symbol: f for f in fills}
        syms = list(trace.index)
        extra: dict[str, object] = {}
        for col in preds.columns:
            if col != "score":
                extra[col] = preds[col].reindex(trace.index).to_numpy()
        extra["side"] = [fill_by_symbol[s].intent.side.value if s in fill_by_symbol else "" for s in syms]
        extra["qty"] = [fill_by_symbol[s].qty if s in fill_by_symbol else 0 for s in syms]
        extra["fill_price"] = [fill_by_symbol[s].price if s in fill_by_symbol else np.nan for s in syms]
        extra["slippage_bps"] = [
            fill_by_symbol[s].slippage_bps if s in fill_by_symbol else np.nan for s in syms
        ]
        extra["costs_total"] = [fill_by_symbol[s].costs.total if s in fill_by_symbol else 0.0 for s in syms]
        extra["missed"] = [fill_by_symbol[s].missed if s in fill_by_symbol else False for s in syms]
        extra["held"] = [s in portfolio.positions for s in syms]
        extra["position_value"] = [values.get(s, 0.0) for s in syms]
        extra["date"] = [asof] * len(syms)
        rep = pd.concat([trace, pd.DataFrame(extra, index=trace.index)], axis=1)
        replay_parts.append(rep)
        for f in fills:
            fill_rows.append(
                {
                    "date": asof,
                    "symbol": f.intent.symbol,
                    "side": f.intent.side.value,
                    "qty": f.qty,
                    "price": f.price,
                    "notional": f.notional,
                    "slippage_bps": f.slippage_bps,
                    "costs_total": f.costs.total,
                    "costs_statutory": f.costs.statutory,
                    "dp_charge": f.costs.dp_charge,
                    "missed": f.missed,
                    "missed_reason": f.missed_reason,
                    "reason": f.intent.reason,
                }
            )
        prev_equity = equity
        prev_asof = asof

    daily = pd.DataFrame(daily_rows)
    replay = (
        pd.concat(replay_parts).reset_index().set_index(["date", "symbol"]).sort_index()
        if replay_parts
        else pd.DataFrame()
    )
    fills_df = pd.DataFrame(fill_rows)
    return RunResult(spec=spec, daily=daily, replay=replay, fills=fills_df, provenance=provenance)


def decision_config_from(cfg: AppConfig, *, round_trip_cost_bps: float) -> DecisionConfig:
    s = cfg.base.strategy
    return DecisionConfig(
        k=s.k,
        select_from=s.select_from,
        sizing=s.sizing,
        position_size=cfg.base.simulation.position_size,
        min_position=s.min_position,
        max_position_pct=s.max_position_pct,
        liquidity_cap_bps_adv=s.liquidity_cap_bps_adv,
        sector_max=s.sector_max,
        cluster_max=s.cluster_max,
        rebalance_mode=s.rebalance_mode,
        rebalance_frequency=s.rebalance_frequency,
        hold_days=s.hold_days,
        abstain_enabled=s.abstain.enabled,
        abstain_buffer_bps=s.abstain.buffer_bps,
        round_trip_cost_bps=round_trip_cost_bps,
        product=Product(cfg.base.simulation.product),
    )
