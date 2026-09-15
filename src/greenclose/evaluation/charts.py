"""Charts for the HTML report, rendered to base64-embedded PNG so the report needs no external assets.

Matplotlib only, Agg backend, no seaborn, no CDN. One figure per function, each returning an
`<img>` tag ready to drop into the page.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#1b1b1b"
GRID = "#d8d8d8"
ACCENT = "#1f6f8b"
WARN = "#b3452f"
BAND = "#bcd6e0"
FIGSIZE = (9.0, 3.4)
DPI = 110


def _style(ax: Any, title: str, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def _to_img(fig: Any, alt: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img alt="{alt}" src="data:image/png;base64,{data}" />'


def ic_timeseries(ic: pd.Series, *, window: int = 60, title: str = "Daily rank IC") -> str:
    """Rolling-mean IC with a +/-2 standard-error band, plus the full-sample mean."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ic = ic.dropna()
    roll = ic.rolling(window, min_periods=window // 2)
    mean, se = roll.mean(), roll.std() / np.sqrt(window)
    ax.axhline(0.0, color=INK, linewidth=0.8)
    ax.fill_between(mean.index, mean - 2 * se, mean + 2 * se, color=BAND, alpha=0.7, linewidth=0)
    ax.plot(mean.index, mean, color=ACCENT, linewidth=1.3, label=f"{window}-day mean")
    ax.axhline(
        float(ic.mean()), color=WARN, linewidth=1.0, linestyle="--", label=f"full sample {ic.mean():+.4f}"
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax, title, "rank IC")
    return _to_img(fig, "IC time series")


def equity_curves(curves: dict[str, pd.Series], *, title: str = "Equity, net of all costs") -> str:
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    for i, (name, eq) in enumerate(curves.items()):
        if eq is None or not len(eq):
            continue
        style = "-" if i == 0 else "--"
        ax.plot(eq.index, eq / eq.iloc[0], linestyle=style, linewidth=1.3, label=name)
    ax.axhline(1.0, color=INK, linewidth=0.8)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    _style(ax, title, "growth of 1")
    return _to_img(fig, "equity curves")


def cost_multiplier_curve(df: pd.DataFrame, *, title: str = "Net Sharpe against the cost multiplier") -> str:
    """The brief's decisive plot: a strategy that dies between 1x and 1.5x is a cost assumption."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = df["multiplier"].to_numpy(dtype="float64")
    y = df["sharpe"].to_numpy(dtype="float64")
    ax.axhline(0.0, color=INK, linewidth=0.8)
    ax.axvline(1.0, color=WARN, linewidth=1.0, linestyle=":", label="modelled costs")
    ax.plot(x, y, marker="o", color=ACCENT, linewidth=1.4)
    ax.legend(fontsize=8, frameon=False)
    _style(ax, title, "net Sharpe", "cost multiplier")
    return _to_img(fig, "cost multiplier curve")


def reliability_curve(table: pd.DataFrame, *, title: str = "Calibration (decile-binned)") -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=INK, linewidth=0.9, linestyle="--", label="perfect")
    if len(table):
        ax.plot(
            table["mean_prob"], table["frac_positive"], marker="o", color=ACCENT, linewidth=1.4, label="model"
        )
    ax.legend(fontsize=8, frameon=False)
    _style(ax, title, "realised frequency", "predicted probability")
    return _to_img(fig, "reliability curve")


def null_distribution(values: np.ndarray, observed: float, *, title: str, xlabel: str) -> str:
    """The B0 histogram with the 95th percentile and the observed value marked."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.hist(values, bins=30, color=BAND, edgecolor=ACCENT, linewidth=0.6)
    p95 = float(np.nanpercentile(values, 95))
    ax.axvline(p95, color=INK, linewidth=1.1, linestyle="--", label=f"random 95th pct {p95:+.4f}")
    if observed == observed:
        ax.axvline(observed, color=WARN, linewidth=1.6, label=f"model {observed:+.4f}")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, title, "simulations", xlabel)
    return _to_img(fig, "null distribution")


def per_year_bars(df: pd.DataFrame, value_col: str, *, title: str, ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    colors = [ACCENT if v >= 0 else WARN for v in df[value_col]]
    pos = np.arange(len(df))
    ax.bar(pos, df[value_col], color=colors, width=0.65)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(y) for y in df["year"]])
    ax.axhline(0.0, color=INK, linewidth=0.8)
    _style(ax, title, ylabel)
    return _to_img(fig, "per-year bars")


def importance_stability(
    imp: pd.DataFrame, *, top: int = 20, title: str = "Feature importance by fold"
) -> str:
    """Mean importance with the min-max range across folds: a wide range means fold-specific noise."""
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.22 * min(top, len(imp)))))
    d = imp.head(top).iloc[::-1]
    ax.barh(d.index, d["mean"], color=BAND, edgecolor=ACCENT, linewidth=0.6)
    ax.errorbar(
        d["mean"],
        range(len(d)),
        xerr=[d["mean"] - d["min"], d["max"] - d["mean"]],
        fmt="none",
        ecolor=INK,
        elinewidth=0.8,
        capsize=2,
    )
    _style(ax, title, "", "mean |SHAP| across folds")
    ax.tick_params(axis="y", labelsize=7)
    return _to_img(fig, "importance stability")
