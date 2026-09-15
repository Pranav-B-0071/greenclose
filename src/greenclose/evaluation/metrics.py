"""Family A metrics (brief section 11.1): did the model predict?

Every per-date metric is computed *within* each date and then averaged over dates. Inputs are Series
indexed by (date, symbol). Fifty stocks on one day are not fifty independent observations, so no
statistic here is computed as if the pooled rows were independent; significance lives in
`significance.py` and always works on the daily series.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def _align(pred: pd.Series, realised: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"p": pred, "r": realised}).dropna()
    return df[df.groupby(level="date")["r"].transform("size") >= 3]


def daily_ic(pred: pd.Series, realised: pd.Series) -> pd.Series:
    """Spearman rank correlation between prediction and realised return, per date."""
    df = _align(pred, realised)
    g = df.groupby(level="date")
    rp = g["p"].rank() - g["p"].rank().groupby(level="date").transform("mean")
    rr = g["r"].rank() - g["r"].rank().groupby(level="date").transform("mean")
    num = (rp * rr).groupby(level="date").sum()
    den = np.sqrt((rp * rp).groupby(level="date").sum() * (rr * rr).groupby(level="date").sum())
    return (num / den.replace(0.0, np.nan)).rename("ic")


def ic_stats(ic: pd.Series) -> dict[str, float]:
    ic = ic.dropna()
    if len(ic) < 2:
        return {
            "ic_mean": math.nan,
            "ic_std": math.nan,
            "icir": math.nan,
            "ic_n": float(len(ic)),
            "ic_frac_pos": math.nan,
        }
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else math.nan,
        "ic_n": float(len(ic)),
        "ic_frac_pos": float((ic > 0).mean()),
    }


def _top_k(df: pd.DataFrame, k: int) -> pd.Series:
    order = df.groupby(level="date")["p"].rank(ascending=False, method="first")
    return order <= k


def precision_at_k(pred: pd.Series, realised: pd.Series, k: int, mode: str = "green") -> pd.Series:
    """Fraction of the top-K that were green (r > 0) or beat the day's cross-sectional mean."""
    df = _align(pred, realised)
    top = _top_k(df, k)
    if mode == "green":
        hit = df["r"] > 0
    elif mode == "beat":
        hit = df["r"] > df.groupby(level="date")["r"].transform("mean")
    else:
        raise ValueError(mode)
    return hit[top].groupby(level="date").mean().rename(f"p_at_{k}_{mode}")


def daily_green_fraction(realised: pd.Series) -> pd.Series:
    return (realised.dropna() > 0).groupby(level="date").mean().rename("green_frac")


def lift_at_k(pred: pd.Series, realised: pd.Series, k: int) -> pd.Series:
    """Precision@K(green) minus the day's green fraction. Raw accuracy is meaningless without it."""
    p = precision_at_k(pred, realised, k, "green")
    base = daily_green_fraction(realised).reindex(p.index)
    return (p - base).rename(f"lift_at_{k}")


def ndcg_at_k(pred: pd.Series, realised: pd.Series, k: int = 5) -> pd.Series:
    """NDCG@k with gain = within-day rank of the realised return scaled to [0, 1]."""
    df = _align(pred, realised)
    n = df.groupby(level="date")["r"].transform("size")
    gain = (df.groupby(level="date")["r"].rank(method="first") - 1.0) / (n - 1.0).clip(lower=1.0)
    pos = df.groupby(level="date")["p"].rank(ascending=False, method="first")
    disc = 1.0 / np.log2(pos + 1.0)
    dcg = (gain * disc).where(pos <= k, 0.0).groupby(level="date").sum()
    ideal_pos = df.groupby(level="date")["r"].rank(ascending=False, method="first")
    idcg = (gain / np.log2(ideal_pos + 1.0)).where(ideal_pos <= k, 0.0).groupby(level="date").sum()
    return (dcg / idcg.replace(0.0, np.nan)).rename(f"ndcg_at_{k}")


def mrr(pred: pd.Series, realised: pd.Series) -> pd.Series:
    """Mean reciprocal rank: 1 / predicted position of the day's best realised stock."""
    df = _align(pred, realised)
    pos = df.groupby(level="date")["p"].rank(ascending=False, method="first")
    best = df.groupby(level="date")["r"].rank(ascending=False, method="first") == 1
    return (1.0 / pos[best]).groupby(level="date").first().rename("mrr")


def decile_spread(pred: pd.Series, realised: pd.Series, frac: float = 0.1) -> pd.Series:
    """Mean realised return of the top decile minus the bottom decile, per date (gross, long-short)."""
    df = _align(pred, realised)
    n = df.groupby(level="date")["p"].transform("size")
    m = np.maximum(1, np.floor(n * frac)).astype(int)
    order = df.groupby(level="date")["p"].rank(ascending=False, method="first")
    top = df["r"].where(order <= m).groupby(level="date").mean()
    bottom = df["r"].where(order > n - m).groupby(level="date").mean()
    return (top - bottom).rename("decile_spread")


def r2_oos_zero(pred_return: pd.Series, realised: pd.Series) -> float:
    """Out-of-sample R^2 against a zero forecast (Gu-Kelly-Xiu), on demeaned returns."""
    df = pd.DataFrame({"p": pred_return, "r": realised}).dropna()
    if df.empty:
        return math.nan
    return float(1.0 - ((df["r"] - df["p"]) ** 2).sum() / (df["r"] ** 2).sum())


def pesaran_timmermann(pred_sign: pd.Series, realised_sign: pd.Series) -> tuple[float, float]:
    """PT (1992) directional test. Returns (statistic ~ N(0,1), one-sided p-value)."""
    df = pd.DataFrame({"x": pred_sign, "y": realised_sign}).dropna()
    t = len(df)
    if t < 10:
        return math.nan, math.nan
    x = (df["x"] > 0).astype(float)
    y = (df["y"] > 0).astype(float)
    p = float((x == y).mean())
    py, px = float(y.mean()), float(x.mean())
    p_star = py * px + (1 - py) * (1 - px)
    var_p = p_star * (1 - p_star) / t
    var_ps = (
        (2 * py - 1) ** 2 * px * (1 - px) / t
        + (2 * px - 1) ** 2 * py * (1 - py) / t
        + 4 * py * px * (1 - py) * (1 - px) / t**2
    )
    denom = var_p - var_ps
    if denom <= 0:
        return math.nan, math.nan
    stat = (p - p_star) / math.sqrt(denom)
    return float(stat), float(1 - stats.norm.cdf(stat))


def brier_score(prob: pd.Series, outcome: pd.Series) -> float:
    df = pd.DataFrame({"p": prob, "y": outcome}).dropna()
    return float(((df["p"] - df["y"]) ** 2).mean()) if len(df) else math.nan


def reliability_table(prob: pd.Series, outcome: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": prob, "y": outcome}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "mean_prob", "frac_positive", "n"])
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    out = (
        df.groupby("bin")
        .agg(mean_prob=("p", "mean"), frac_positive=("y", "mean"), n=("y", "size"))
        .reset_index()
    )
    return out


def family_a(
    pred: pd.Series,
    realised: pd.Series,
    *,
    expected_return: pd.Series | None = None,
    prob_beat: pd.Series | None = None,
    prob_green: pd.Series | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
) -> tuple[dict[str, float], pd.DataFrame]:
    """All Family A metrics: summary dict plus the per-date frame the significance tools need."""
    per_date = pd.DataFrame({"ic": daily_ic(pred, realised)})
    for k in ks:
        per_date[f"p_at_{k}_green"] = precision_at_k(pred, realised, k, "green")
        per_date[f"p_at_{k}_beat"] = precision_at_k(pred, realised, k, "beat")
        per_date[f"lift_at_{k}"] = lift_at_k(pred, realised, k)
    per_date["ndcg_at_5"] = ndcg_at_k(pred, realised, 5)
    per_date["mrr"] = mrr(pred, realised)
    per_date["decile_spread"] = decile_spread(pred, realised)
    per_date["green_frac"] = daily_green_fraction(realised)
    summary: dict[str, float] = ic_stats(per_date["ic"])
    for c in per_date.columns:
        if c != "ic":
            summary[f"{c}_mean"] = float(per_date[c].mean())
    df = _align(pred, realised)
    demeaned_r = df["r"] - df.groupby(level="date")["r"].transform("mean")
    if expected_return is not None:
        e = expected_return.reindex(df.index)
        summary["r2_oos_zero"] = r2_oos_zero(e, demeaned_r)
        pt_stat, pt_p = pesaran_timmermann(e, demeaned_r)
    else:
        demeaned_p = df["p"] - df.groupby(level="date")["p"].transform("mean")
        summary["r2_oos_zero"] = math.nan
        pt_stat, pt_p = pesaran_timmermann(demeaned_p, demeaned_r)
    summary["pt_stat"], summary["pt_pvalue"] = pt_stat, pt_p
    if prob_beat is not None:
        beat = (demeaned_r > 0).astype(float)
        summary["brier_beat"] = brier_score(prob_beat.reindex(df.index), beat)
    if prob_green is not None:
        summary["brier_green"] = brier_score(prob_green.reindex(df.index), (df["r"] > 0).astype(float))
    return summary, per_date
