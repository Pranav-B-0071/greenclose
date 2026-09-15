"""Significance tools (brief sections 11.2-11.4).

- Newey-West t-statistics on daily series (HAC, maxlags 5); the bar is t > 3.0, not 2.0
- stationary bootstrap over days for every headline confidence interval
- Deflated Sharpe Ratio with the honest trial count from research_log
- White's Reality Check / Hansen-style SPA across a family of tried variants
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import StationaryBootstrap
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def newey_west(x: pd.Series | np.ndarray, maxlags: int = 5) -> dict[str, float]:
    """Mean of a daily series with a HAC standard error and t-statistic."""
    arr = np.asarray(pd.Series(x).dropna(), dtype="float64")
    n = len(arr)
    if n < 5:
        return {"mean": float(np.mean(arr)) if n else math.nan, "se": math.nan, "t": math.nan, "n": float(n)}
    model = sm.OLS(arr, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "mean": float(model.params[0]),
        "se": float(model.bse[0]),
        "t": float(model.tvalues[0]),
        "n": float(n),
    }


def stationary_bootstrap_ci(
    x: pd.Series | np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 10_000,
    block: float = 7.0,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of `stat` under a stationary bootstrap over days (expected block length `block`)."""
    arr = np.asarray(pd.Series(x).dropna(), dtype="float64")
    if len(arr) < 10:
        return math.nan, math.nan
    bs = StationaryBootstrap(block, arr, seed=seed)
    draws = np.array([stat(data[0][0]) for data in bs.bootstrap(n_boot)])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(
    a: pd.Series, b: pd.Series, *, n_boot: int = 10_000, block: float = 7.0, seed: int = 0
) -> dict[str, float]:
    """Bootstrap CI on mean(a - b) with a and b aligned by date (a paired comparison over days)."""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    d = (df["a"] - df["b"]).to_numpy()
    lo, hi = stationary_bootstrap_ci(d, n_boot=n_boot, block=block, seed=seed)
    return {"diff": float(d.mean()) if len(d) else math.nan, "lo": lo, "hi": hi, "n": float(len(d))}


def expected_max_sharpe(sharpes: Sequence[float], n_trials: int | None = None) -> float:
    """SR0 of Bailey & Lopez de Prado (2014): expected max Sharpe among N trials under the null.

    `sharpes` estimates the variance of Sharpe across trials; `n_trials` is the honest count of
    configurations tried, which is larger whenever a trial produced no Sharpe (a configuration that
    abstained on every day, say). Understating N would flatter the Deflated Sharpe Ratio.
    """
    n = int(n_trials) if n_trials else len(sharpes)
    if n <= 1 or len(sharpes) <= 1:
        return 0.0
    v = float(np.var(np.asarray(sharpes, dtype="float64"), ddof=1))
    if v <= 0:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e)))


def deflated_sharpe(
    sr_hat: float,
    *,
    n_obs: int,
    skew: float,
    kurt: float,
    sharpes_tried: Sequence[float],
    n_trials: int | None = None,
) -> dict[str, float]:
    """DSR = Phi[(SR^ - SR0) sqrt(T-1) / sqrt(1 - g3 SR^ + (g4-1)/4 SR^^2)], all per-period units.

    `kurt` is the non-excess kurtosis. A DSR below 0.95 means the observed Sharpe is not distinguishable
    from the best expected by chance given how many configurations were tried.
    """
    sr0 = expected_max_sharpe(sharpes_tried, n_trials)
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if n_obs < 2 or denom <= 0 or math.isnan(sr_hat):
        return {"dsr": math.nan, "sr0": sr0, "n_trials": float(n_trials or len(sharpes_tried))}
    z = (sr_hat - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return {
        "dsr": float(stats.norm.cdf(z)),
        "sr0": sr0,
        "n_trials": float(n_trials or len(sharpes_tried)),
        "n_sharpes": float(len(sharpes_tried)),
        "z": float(z),
    }


def spa_pvalue(
    relative: pd.DataFrame, *, n_boot: int = 2_000, block: float = 7.0, seed: int = 0
) -> dict[str, float | str]:
    """White's Reality Check over a family: H0 = no variant beats the benchmark.

    `relative` holds one column per variant of daily benchmark-relative returns. The statistic is
    max_k sqrt(T) mean(d_k); its null distribution comes from a stationary bootstrap of the centred
    series (Hansen's SPA re-centring). Returns the p-value and the best variant.
    """
    df = relative.dropna(how="all").fillna(0.0)
    t = len(df)
    if t < 20 or df.shape[1] == 0:
        return {"p_value": math.nan, "best": "", "stat": math.nan, "n_variants": float(df.shape[1])}
    d = df.to_numpy(dtype="float64")
    means = d.mean(axis=0)
    observed = math.sqrt(t) * means.max()
    centred = d - means
    bs = StationaryBootstrap(block, centred, seed=seed)
    exceed = 0
    for data in bs.bootstrap(n_boot):
        boot = data[0][0]
        stat = math.sqrt(t) * boot.mean(axis=0).max()
        if stat >= observed:
            exceed += 1
    return {
        "p_value": exceed / n_boot,
        "best": str(df.columns[int(means.argmax())]),
        "stat": float(observed),
        "n_variants": float(df.shape[1]),
    }


def per_period_sharpe(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    return float(r.mean() / r.std()) if len(r) > 1 and r.std() > 0 else math.nan
