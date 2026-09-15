"""Family B metrics (brief section 11.2): would it have made money?

Everything is computed from the engine's daily frame. Sharpe and Sortino are annualised from daily
returns with a zero risk-free rate (stated). Costs are reported as a share of gross P&L: above ~50% the
design is wrong regardless of the net result.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

TRADING_DAYS = 250.0


def daily_returns(daily: pd.DataFrame, capital: float) -> pd.Series:
    eq = daily.set_index("date")["equity"]
    prev = eq.shift(1).fillna(capital)
    return (eq / prev - 1.0).rename("ret")


def drawdown(equity: pd.Series) -> tuple[float, int]:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min()) if len(dd) else math.nan
    # longest run of days below the previous peak
    under = (dd < 0).to_numpy()
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return mdd, int(longest)


def summarise(
    daily: pd.DataFrame, capital: float, *, benchmark_ret: pd.Series | None = None
) -> dict[str, float]:
    r = daily_returns(daily, capital)
    n = len(r)
    if n == 0:
        return {}
    eq = daily.set_index("date")["equity"]
    years = n / TRADING_DAYS
    total = float(eq.iloc[-1] / capital - 1.0)
    cagr = float((1 + total) ** (1 / years) - 1.0) if years > 0 and total > -1 else math.nan
    vol = float(r.std() * math.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std() * math.sqrt(TRADING_DAYS)) if r.std() > 0 else math.nan
    downside = r[r < 0].std()
    sortino = float(r.mean() / downside * math.sqrt(TRADING_DAYS)) if downside and downside > 0 else math.nan
    mdd, dd_days = drawdown(eq)
    wins, losses = r[r > 0], r[r < 0]
    gross = float(daily["gross_pnl"].sum())
    costs = float(daily["costs_total"].sum())
    net = float(daily["net_pnl"].sum())
    turnover = float(daily["turnover_notional"].sum() / eq.mean() / years) if years > 0 else math.nan
    # Scale-free view: the strategy deploys only part of the account, and a decaying account keeps
    # betting a fixed rupee position size, so returns *on capital* mix skill with deployment. These
    # per-invested-rupee figures are what make B0..B5 and the model rungs comparable to each other.
    invested = float(daily["position_value_total"].mean()) if "position_value_total" in daily else math.nan
    if not (invested > 0):
        invested = math.nan
    out: dict[str, float] = {
        "n_days": float(n),
        "total_return": total,
        "cagr": cagr,
        "net_return_pa": float(net / capital / years) if years > 0 else math.nan,
        "gross_return_pa": float(gross / capital / years) if years > 0 else math.nan,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "drawdown_days": float(dd_days),
        "win_rate": float((r > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else math.nan,
        "avg_loss": float(losses.mean()) if len(losses) else math.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else math.nan,
        "turnover_pa": turnover,
        "gross_pnl": gross,
        "costs_total": costs,
        "net_pnl": net,
        # gross <= 0 means costs exceeded every rupee the signal made; the ratio is reported as
        # infinite rather than as a missing value, which would read as "not measured"
        "costs_over_gross_pnl": float(costs / gross) if gross > 0 else (math.inf if costs > 0 else math.nan),
        "costs_pa_pct_capital": float(costs / capital / years) if years > 0 else math.nan,
        "mean_deployment": float(invested / eq.mean()) if invested == invested else math.nan,
        "gross_bps_per_day_invested": float(gross / invested / n * 1e4) if invested == invested else math.nan,
        "net_bps_per_day_invested": float(net / invested / n * 1e4) if invested == invested else math.nan,
        "cost_bps_per_day_invested": float(costs / invested / n * 1e4) if invested == invested else math.nan,
        "skew": float(r.skew()),
        "kurtosis": float(r.kurt() + 3.0),  # non-excess, as the DSR formula needs
        "sharpe_per_period": float(r.mean() / r.std()) if r.std() > 0 else math.nan,
        "n_trades": float((daily["n_buys"] + daily["n_sells"]).sum()),
        "missed_trade_rate": float(
            daily["n_missed"].sum() / max(1.0, (daily["n_buys"] + daily["n_sells"] + daily["n_missed"]).sum())
        ),
        "abstain_rate": float((daily["n_abstained"] > 0).mean()),
    }
    if benchmark_ret is not None:
        b = benchmark_ret.reindex(r.index).fillna(0.0)
        x = sm.add_constant(b.to_numpy())
        fit = sm.OLS(r.to_numpy(), x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        out["alpha_pa_vs_nifty"] = float(fit.params[0] * TRADING_DAYS)
        out["alpha_t_vs_nifty"] = float(fit.tvalues[0])
        out["beta_vs_nifty"] = float(fit.params[1])
        out["excess_return_pa_vs_nifty"] = float((r.mean() - b.mean()) * TRADING_DAYS)
    return out


def per_year(r: pd.Series) -> pd.DataFrame:
    rows = []
    for year, g in r.groupby(r.index.year):
        sharpe = g.mean() / g.std() * math.sqrt(TRADING_DAYS) if g.std() > 0 else math.nan
        rows.append(
            {
                "year": int(year),
                "n_days": len(g),
                "return": float((1 + g).prod() - 1),
                "sharpe": float(sharpe),
            }
        )
    return pd.DataFrame(rows)


def per_regime(r: pd.Series, regime: pd.Series, name: str = "regime") -> pd.DataFrame:
    df = pd.DataFrame({"r": r, name: regime.reindex(r.index)}).dropna()
    rows = []
    for label, g in df.groupby(name):
        s = g["r"]
        rows.append(
            {
                name: label,
                "n_days": len(s),
                "mean_daily_bps": float(s.mean() * 1e4),
                "sharpe": float(s.mean() / s.std() * math.sqrt(TRADING_DAYS)) if s.std() > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def vol_terciles(nifty_vol: pd.Series) -> pd.Series:
    v = nifty_vol.dropna()
    return pd.qcut(v.rank(method="first"), 3, labels=["low_vol", "mid_vol", "high_vol"]).astype(str)


def nifty_buy_and_hold(nifty_close: pd.Series, dates: pd.DatetimeIndex, capital: float) -> pd.DataFrame:
    """B1: the index proxy, no trading, no costs (an index cannot be bought; a reference, not a strategy)."""
    # the index series has holidays the equity calendar does not (1 January and similar), so carry the
    # last observed close forward across the union before reindexing; never fill backwards
    union = nifty_close.index.union(dates)
    px = nifty_close.reindex(union).ffill().reindex(dates)
    if px.isna().any():
        raise ValueError("benchmark series has no observation at or before the first evaluation date")
    eq = capital * px / px.iloc[0]
    zeros = np.zeros(len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "equity": eq.to_numpy(),
            "cash_settled": zeros,
            "cash_unsettled": zeros,
            "n_positions": np.ones(len(dates)),
            "position_value_total": eq.to_numpy(),
            "gross_pnl": eq.diff().fillna(eq.iloc[0] - capital).to_numpy(),
            "net_pnl": eq.diff().fillna(eq.iloc[0] - capital).to_numpy(),
            "costs_total": zeros,
            "costs_statutory": zeros,
            "costs_slippage": zeros,
            "costs_dp": zeros,
            "costs_stt": zeros,
            "turnover_notional": zeros,
            "n_buys": zeros,
            "n_sells": zeros,
            "n_missed": zeros,
            "n_targets": zeros,
            "n_abstained": zeros,
            "high_vol_regime": np.zeros(len(dates), dtype=bool),
        }
    )
