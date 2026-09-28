"""Single self-contained HTML report (brief section 14).

It opens with the verdict block, and the verdict is written mechanically from the thresholds in
config/base.yaml. Nothing here softens a number in prose. Every chart is an embedded base64 PNG, so
the file has no external assets and can be opened from anywhere.

The generator reads only artefacts already on disk (run summaries, ladder.json, sweep parquets, the
research log, the leakage suite result), so it is cheap to regenerate after a long run finishes.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from greenclose.backtest.costs import CostModel, break_even_gross_alpha_bps
from greenclose.config import AppConfig, Paths
from greenclose.domain.models import Product
from greenclose.evaluation import charts
from greenclose.evaluation.baselines import PLANNING_SLIPPAGE_BPS
from greenclose.evaluation.significance import deflated_sharpe
from greenclose.ops.provenance import git_sha, utc_now_iso
from greenclose.ops.research_log import ResearchLog, distinct

TICK_PASS = "&#10003;"
TICK_FAIL = "&#10007;"


@dataclass
class Criterion:
    label: str
    value: str
    passed: bool
    note: str = ""


@dataclass
class ReportInputs:
    cfg: AppConfig
    paths: Paths
    snapshot_id: str
    run_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    ladder: dict[str, Any] = field(default_factory=dict)
    leakage: dict[str, Any] = field(default_factory=dict)
    sweeps: pd.DataFrame = field(default_factory=pd.DataFrame)
    b0: pd.DataFrame = field(default_factory=pd.DataFrame)
    research_log: list[dict[str, Any]] = field(default_factory=list)


# ----------------------------------------------------------------------------- loading


def load_run_summaries(paths: Paths) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    root = paths.data_dir / "runs"
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        f = d / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        out[s["name"]] = s | {"_dir": str(d)}  # later runs of the same name win
    return out


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def load_b0(paths: Paths, snapshot_id: str) -> pd.DataFrame:
    files = sorted((paths.data_dir / "runs").glob(f"B0_distribution__{snapshot_id}__n*.parquet"))
    if not files:
        return pd.DataFrame()
    best = max(files, key=lambda p: int(p.stem.rsplit("__n", 1)[-1]))
    return pd.read_parquet(best)


def load_sweeps(paths: Paths) -> pd.DataFrame:
    files = (
        sorted((paths.data_dir / "sweeps").glob("*.parquet")) if (paths.data_dir / "sweeps").exists() else []
    )
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def gather(cfg: AppConfig, paths: Paths, snapshot_id: str) -> ReportInputs:
    return ReportInputs(
        cfg=cfg,
        paths=paths,
        snapshot_id=snapshot_id,
        run_summaries=load_run_summaries(paths),
        ladder=load_json(paths.research_dir / "ladder.json"),
        leakage=load_json(paths.research_dir / "leakage_suite.json"),
        sweeps=load_sweeps(paths),
        b0=load_b0(paths, snapshot_id),
        research_log=ResearchLog(paths.research_dir / "research_log.jsonl").rows(),
    )


# ----------------------------------------------------------------------------- verdict


def pick_headline(inp: ReportInputs) -> tuple[str, dict[str, Any]]:
    """The best model rung by mean out-of-sample rank IC. Chosen mechanically, not by eye."""
    table = inp.ladder.get("table") or []
    best_name, best = "", {}
    best_ic = -math.inf
    for row in table:
        ic = row.get("ic_mean")
        if ic is not None and ic == ic and ic > best_ic:
            best_ic, best_name, best = ic, str(row.get("name", "")), row
    return best_name, best


def build_criteria(inp: ReportInputs, headline: dict[str, Any]) -> list[Criterion]:
    cfg = inp.cfg
    crit: list[Criterion] = []
    ic = headline.get("ic_mean", math.nan)
    t = headline.get("ic_t_nw", math.nan)
    lo, hi = headline.get("ic_ci_lo", math.nan), headline.get("ic_ci_hi", math.nan)
    bar = cfg.base.report.ic_t_bar
    crit.append(
        Criterion(
            "Mean daily rank IC",
            f"{ic:+.4f}   (NW t = {t:+.2f}, 95% CI [{lo:+.4f}, {hi:+.4f}])",
            bool(t == t and t > bar),
            f"bar is t &gt; {bar} (Harvey-Liu-Zhu), not 2.0",
        )
    )
    lift = headline.get("lift_at_3", math.nan)
    lift_lo = headline.get("lift_at_3_ci_lo", math.nan)
    lift_hi = headline.get("lift_at_3_ci_hi", math.nan)
    crit.append(
        Criterion(
            "Precision@3 lift over the daily base rate",
            f"{lift * 100:+.2f} pp"
            + (f"   (95% CI [{lift_lo * 100:+.2f}, {lift_hi * 100:+.2f}] pp)" if lift_lo == lift_lo else ""),
            bool(lift_lo == lift_lo and lift_lo > 0),
            "raw accuracy without this is meaningless",
        )
    )
    b2 = inp.run_summaries.get("B2_equal_weight", {}).get("family_b", {})
    net = headline.get("net_pa", math.nan)
    b2_net = b2.get("net_return_pa", math.nan)
    diff = net - b2_net if (net == net and b2_net == b2_net) else math.nan
    crit.append(
        Criterion(
            "Net return vs B2 (equal-weight universe)",
            f"{diff * 100:+.1f} pp p.a."
            + (f"   (model {net * 100:+.1f}%, B2 {b2_net * 100:+.1f}%)" if net == net else ""),
            bool(diff == diff and diff > 0),
            "the toughest and most important baseline. &quot;p.a.&quot; here is total P&amp;L over "
            "initial capital divided by years, not a compound growth rate, so it is not comparable to "
            "a published CAGR" + _b2_cagr_note(b2),
        )
    )
    cog, n_neg, n_seeds = _cost_share(inp, headline)
    seed_note = f"; {n_neg} of {n_seeds} seeds made no gross profit at all" if n_neg else ""
    crit.append(
        Criterion(
            "Costs as a share of gross P&amp;L",
            ("infinite (gross P&amp;L is not positive)" if cog == math.inf else f"{cog * 100:.0f}%")
            + (f"   ({n_neg}/{n_seeds} seeds gross-negative)" if n_neg else "")
            if cog == cog
            else "n/a",
            bool(cog == cog and cog != math.inf and cog < cfg.base.report.costs_gross_max),
            f"above {cfg.base.report.costs_gross_max:.0%} the design is wrong regardless of the net "
            f"result. Pooled over seeds in rupees, not averaged as a ratio{seed_note}",
        )
    )
    dsr = compute_dsr(inp, headline)
    crit.append(
        Criterion(
            "Deflated Sharpe Ratio",
            f"{dsr['dsr']:.3f}   (N = {int(dsr['n_trials'])} configurations tried, "
            f"{int(dsr.get('n_sharpes', 0))} with a Sharpe, SR0 = {dsr['sr0']:.4f})"
            if dsr["dsr"] == dsr["dsr"]
            else f"n/a (N = {int(dsr['n_trials'])} configurations tried)",
            bool(dsr["dsr"] == dsr["dsr"] and dsr["dsr"] > cfg.base.report.dsr_bar),
            f"bar is DSR &gt; {cfg.base.report.dsr_bar}{_seed_spread_note(inp, headline)}",
        )
    )
    if len(inp.b0):
        p95 = float(inp.b0["net_pa"].quantile(0.95)) if "net_pa" in inp.b0 else math.nan
        crit.append(
            Criterion(
                "Beats the 95th percentile of random (B0)",
                f"model {net * 100:+.1f}% p.a. vs random p95 {p95 * 100:+.1f}% p.a."
                f" over {len(inp.b0)} simulations",
                bool(net == net and p95 == p95 and net > p95),
                "beating the mean is not enough",
            )
        )
    lk = inp.leakage
    if lk:
        crit.append(
            Criterion(
                "Leakage suite",
                f"{lk.get('n_passed', 0)}/{lk.get('n_checks', 6)} PASS on the real snapshot",
                bool(lk.get("n_passed") == lk.get("n_checks")),
                "",
            )
        )
    return crit


def _b2_cagr_note(b2: dict[str, Any]) -> str:
    cagr = b2.get("cagr", math.nan)
    return f" (B2 compounds at {cagr * 100:.1f}% a year)" if cagr == cagr else ""


def _cost_share(inp: ReportInputs, headline: dict[str, Any]) -> tuple[float, int, int]:
    """Costs as a share of gross P&L, pooled in rupees over the seeds.

    Prefers the value the ladder already pooled. Falls back to the per-seed run summaries so a
    ladder.json written before pooling existed still reports the honest ratio rather than infinity.
    """
    if "n_seeds_gross_nonpositive" in headline:
        return (
            float(headline.get("costs_over_gross", math.nan)),
            int(headline["n_seeds_gross_nonpositive"] or 0),
            int(headline.get("n_seeds", 0) or 0),
        )
    prefix = str(headline.get("name", "")).replace(" ", "_")
    runs = [s.get("family_b", {}) for k, s in inp.run_summaries.items() if k.startswith(prefix)]
    if not runs:
        return math.nan, 0, int(headline.get("n_seeds", 0) or 0)
    gross = sum(float(r.get("gross_pnl", 0.0) or 0.0) for r in runs)
    costs = sum(float(r.get("costs_total", 0.0) or 0.0) for r in runs)
    n_neg = sum(1 for r in runs if float(r.get("gross_pnl", 0.0) or 0.0) <= 0.0)
    share = costs / gross if gross > 0.0 else (math.inf if costs > 0.0 else math.nan)
    return share, n_neg, len(runs)


def _seed_spread_note(inp: ReportInputs, headline: dict[str, Any]) -> str:
    """Say so when the spread across seeds is as large as the effect. That is itself the finding."""
    sr, sd = headline.get("sharpe_per_period", math.nan), headline.get("sharpe_sd", math.nan)
    if sr != sr or sd != sd:
        sr, sd = _seed_sharpe_spread(inp, str(headline.get("name", "")))
    if sr != sr or sd != sd or sd <= 0:
        return ""
    note = f". Sharpe across seeds {sr:.4f} +/- {sd:.4f}"
    return note + (" -- the spread is as large as the effect" if sd >= abs(sr) * 0.9 else "")


def _seed_sharpe_spread(inp: ReportInputs, name: str) -> tuple[float, float]:
    """Per-period Sharpe of one rung across seeds: mean and sample standard deviation."""
    prefix = name.replace(" ", "_")
    vals = [
        float(s["family_b"]["sharpe_per_period"])
        for k, s in inp.run_summaries.items()
        if k.startswith(prefix) and (s.get("family_b", {}).get("sharpe_per_period") is not None)
    ]
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return math.nan, math.nan
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def _seed_mean_family_b(inp: ReportInputs, name: str) -> dict[str, float]:
    """Family B for one rung, averaged over every seed whose run summary is on disk."""
    prefix = name.replace(" ", "_")
    runs = [s.get("family_b", {}) for k, s in inp.run_summaries.items() if k.startswith(prefix)]
    if not runs:
        return {}
    out: dict[str, float] = {}
    for key in ("sharpe_per_period", "skew", "kurtosis", "n_days"):
        vals = [float(r[key]) for r in runs if r.get(key) is not None and r[key] == r[key]]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def compute_dsr(inp: ReportInputs, headline: dict[str, Any]) -> dict[str, float]:
    """Deflated Sharpe with the honest trial count read from research_log (brief invariant 9)."""
    tried = distinct(inp.research_log)  # one row per configuration, not per time it was run
    sharpes = [
        float(r["oos_sharpe"])
        for r in tried
        if r.get("oos_sharpe") is not None
        and r.get("counts_for_dsr", True)
        and r["oos_sharpe"] == r["oos_sharpe"]
    ]
    # The headline row is the mean over seeds, which is what the brief asks a headline to be. Reading
    # one seed's summary instead -- the old behaviour -- picked whichever run sorted first, and for the
    # rank ensemble that was the single seed out of five that lost money gross.
    sr_hat = headline.get("sharpe_per_period", math.nan)
    n_obs = int(headline.get("n_days", 0) or 0)
    skew = headline.get("skew", 0.0) or 0.0
    kurt = headline.get("kurtosis", 3.0) or 3.0
    if sr_hat != sr_hat or n_obs < 2:  # an older ladder.json without the aggregated Family B columns
        fb = _seed_mean_family_b(inp, str(headline.get("name", "")))
        sr_hat = fb.get("sharpe_per_period", math.nan)
        n_obs = int(fb.get("n_days", 0) or 0)
        skew = fb.get("skew", 0.0) or 0.0
        kurt = fb.get("kurtosis", 3.0) or 3.0
    n_trials = sum(1 for r in tried if r.get("counts_for_dsr", True))
    if sr_hat != sr_hat or n_obs < 2:
        return {"dsr": math.nan, "sr0": 0.0, "n_trials": float(n_trials), "n_sharpes": float(len(sharpes))}
    return deflated_sharpe(
        sr_hat, n_obs=n_obs, skew=skew, kurt=kurt, sharpes_tried=sharpes, n_trials=n_trials
    )


def overall_verdict(crit: list[Criterion], inp: ReportInputs, headline: dict[str, Any]) -> str:
    """Written mechanically from the criteria. Never softened."""
    leak = next((c for c in crit if c.label == "Leakage suite"), None)
    if leak is not None and not leak.passed:
        return (
            "LEAKAGE SUITE FAILED. No result on this page is trustworthy. "
            "Fix the harness before reading further."
        )
    ic_ok = next((c.passed for c in crit if c.label.startswith("Mean daily rank IC")), False)
    money_ok = next((c.passed for c in crit if c.label.startswith("Net return vs B2")), False)
    ic = headline.get("ic_mean", math.nan)
    if ic == ic and ic > 0.06:
        return (
            f"RED FLAG: rank IC of {ic:+.4f} is above the 0.06 line the brief treats as a leak until proven "
            "otherwise. Do not act on this number; re-run the leakage suite and bisect the feature registry."
        )
    if ic_ok and money_ok:
        return (
            "Signal present and net-positive against the equal-weight benchmark. "
            "Treat it with suspicion and re-check the trial count."
        )
    if ic_ok and not money_ok:
        return (
            "Signal present and statistically significant, but not tradeable at this turnover: the model "
            "predicts, the strategy does not pay. This is a genuine result, and the expected one."
        )
    if not ic_ok and ic == ic and ic > 0:
        return (
            "Signal is positive but not significant at the t &gt; 3 bar. Not tradeable, and not "
            "from the reversal effect already documented in the baselines."
        )
    return (
        "No detectable cross-sectional predictability in this universe with these features. "
        "The harness works; the edge is absent."
    )


# ----------------------------------------------------------------------------- rendering


def _table(df: pd.DataFrame | None, title: str, note: str = "", max_rows: int = 200) -> str:
    if df is None or not len(df):
        return f"<h3>{html.escape(title)}</h3><p class='muted'>not available in this run</p>"
    body = df.head(max_rows).to_html(index=False, border=0, na_rep="", float_format=lambda v: f"{v:,.4f}")
    extra = f"<p class='muted'>{html.escape(note)}</p>" if note else ""
    more = f"<p class='muted'>showing {max_rows} of {len(df)} rows</p>" if len(df) > max_rows else ""
    return f"<h3>{html.escape(title)}</h3>{extra}{body}{more}"


def _fmt_pct(v: Any) -> str:
    return f"{float(v) * 100:+.1f}%" if isinstance(v, (int, float)) and v == v else ""


def baseline_table(inp: ReportInputs) -> pd.DataFrame:
    rows = []
    for name, s in sorted(inp.run_summaries.items()):
        if not name.startswith(("B1", "B2", "B3", "B4", "B5")):
            continue
        a, b = s.get("family_a", {}), s.get("family_b", {})
        rows.append(
            {
                "baseline": name,
                "IC": a.get("ic_mean"),
                "IC t (NW)": a.get("ic_t_nw"),
                "gross % p.a.": b.get("gross_return_pa"),
                "net % p.a.": b.get("net_return_pa"),
                "net CI lo": b.get("net_pa_ci_lo"),
                "net CI hi": b.get("net_pa_ci_hi"),
                "Sharpe": b.get("sharpe"),
                "max DD": b.get("max_drawdown"),
                "costs/gross": b.get("costs_over_gross_pnl"),
                "turnover p.a.": b.get("turnover_pa"),
                "gross bps/day invested": b.get("gross_bps_per_day_invested"),
                "net bps/day invested": b.get("net_bps_per_day_invested"),
            }
        )
    if len(inp.b0) and "net_pa" in inp.b0:
        for q, lab in ((0.05, "p05"), (0.50, "p50"), (0.95, "p95")):
            rows.append(
                {
                    "baseline": f"B0 random {lab} (n={len(inp.b0)})",
                    "IC": float(inp.b0["ic_mean"].quantile(q)) if "ic_mean" in inp.b0 else None,
                    "gross % p.a.": float(inp.b0["gross_pa"].quantile(q)) if "gross_pa" in inp.b0 else None,
                    "net % p.a.": float(inp.b0["net_pa"].quantile(q)),
                    "Sharpe": float(inp.b0["sharpe"].quantile(q)) if "sharpe" in inp.b0 else None,
                }
            )
    return pd.DataFrame(rows)


def render(inp: ReportInputs, *, run_id: str) -> str:
    cfg = inp.cfg
    headline_name, headline = pick_headline(inp)
    crit = build_criteria(inp, headline)
    verdict = overall_verdict(crit, inp, headline)
    cm = CostModel(
        cfg.charges,
        product=Product(cfg.base.simulation.product),
        cost_multiplier=cfg.base.simulation.cost_multiplier,
    )
    rt_bps = cm.round_trip_bps(cfg.base.simulation.position_size, slippage_bps_per_leg=PLANNING_SLIPPAGE_BPS)
    break_even = break_even_gross_alpha_bps(round_trip_bps=rt_bps)

    rows = "".join(
        f"<tr><td>{html.escape(c.label)}</td><td class='num'>{c.value}</td>"
        f"<td class='{'ok' if c.passed else 'bad'}'>{TICK_PASS if c.passed else TICK_FAIL}</td>"
        f"<td class='muted'>{c.note}</td></tr>"
        for c in crit
    )
    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>GREENCLOSE {html.escape(run_id)}</title>",
        _CSS,
        "</head><body>",
        "<h1>GREENCLOSE</h1>",
        "<p class='muted'>Cross-sectional next-day return prediction for Indian equities. "
        "<b>Paper only: no broker, no orders, no money.</b></p>",
        f"<p class='meta'>run {html.escape(run_id)} &middot; "
        f"model {html.escape(headline_name or 'n/a')} &middot; "
        f"snapshot {html.escape(inp.snapshot_id)} &middot; "
        f"git {html.escape(git_sha(inp.paths.root))} &middot; "
        f"config {html.escape(cfg.hash())} &middot; generated {html.escape(utc_now_iso())}</p>",
        f"<div class='breakeven'><b>BREAK-EVEN CONTEXT:</b> at a modelled round trip of {rt_bps:.1f} bps, "
        f"this strategy needs about <b>{break_even:.0f} bps of gross daily alpha</b> per trade "
        "to reach a net "
        "Sharpe of 1.0. Published cross-sectional machine-learning decile spreads run 5&ndash;15 bps per day "
        "gross. Read every number below against that.</div>",
        "<table class='verdict'><thead><tr><th>Criterion</th><th>Value</th><th></th>"
        "<th>Note</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        f"<div class='verdictline'><b>VERDICT:</b> {verdict}</div>",
    ]

    parts.append("<h2>Question A &mdash; did the model predict?</h2>")
    ic_series = _headline_ic_series(inp, headline_name)
    if ic_series is not None and len(ic_series):
        parts.append(charts.ic_timeseries(ic_series))
    ladder_tab = pd.DataFrame(inp.ladder.get("table") or [])
    parts.append(
        _table(
            ladder_tab,
            "Model ladder (mean over seeds)",
            "A rung counts as beaten only if the paired bootstrap excludes zero AND the "
            "difference exceeds the seed spread.",
        )
    )
    comps = inp.ladder.get("comparisons") or {}
    if comps:
        cdf = pd.DataFrame([{"comparison": k, **{kk: vv for kk, vv in v.items()}} for k, v in comps.items()])
        parts.append(_table(cdf, "Rung comparisons (paired stationary bootstrap on daily IC)"))
    h2 = pd.DataFrame(inp.ladder.get("h2") or [])
    parts.append(
        _table(h2, "H2: ranking versus classification", "Same features, same folds, same seeds, three heads.")
    )
    if inp.ladder.get("h2_comparisons"):
        parts.append(
            _table(
                pd.DataFrame([{"comparison": k, **v} for k, v in inp.ladder["h2_comparisons"].items()]),
                "H2 differences",
            )
        )
    per_year = _per_year_ic(inp, headline_name)
    if per_year is not None and len(per_year):
        parts.append(
            charts.per_year_bars(per_year, "ic_mean", title="Mean rank IC by calendar year", ylabel="IC")
        )
        parts.append(_table(per_year, "Rank IC by calendar year"))

    parts.append("<h2>Question B &mdash; would it have made money?</h2>")
    curves = _equity_curves(inp, headline_name)
    if curves:
        parts.append(charts.equity_curves(curves))
    parts.append(
        _table(
            baseline_table(inp),
            "Baselines on identical dates, universe and cost model",
            "B2 (equal-weight) is the baseline that matters: it tests whether selection adds "
            "anything at all.",
        )
    )
    if len(inp.b0) and "net_pa" in inp.b0:
        parts.append(
            charts.null_distribution(
                inp.b0["net_pa"].to_numpy(dtype="float64"),
                float(headline.get("net_pa", math.nan)),
                title=f"B0 null distribution of net return ({len(inp.b0)} random simulations)",
                xlabel="net return p.a.",
            )
        )

    if len(inp.sweeps):
        parts.append("<h2>Sweeps</h2>")
        cm_rows = inp.sweeps[inp.sweeps["group"] == "cost_multiplier"].copy()
        if len(cm_rows):
            cm_rows["multiplier"] = cm_rows["cell"].str.extract(r"x([0-9.]+)").astype(float)
            cm_rows = cm_rows.sort_values("multiplier")
            parts.append(charts.cost_multiplier_curve(cm_rows[["multiplier", "sharpe"]]))
        for group in ("hold", "k", "cost_multiplier", "product", "rebalance", "retrain"):
            g = inp.sweeps[inp.sweeps["group"] == group]
            if len(g):
                parts.append(_table(g.drop(columns=["group"]), f"Sweep: {group}"))

    parts.append("<h2>Provenance and honesty</h2>")
    lk = inp.leakage
    if lk:
        lk_df = pd.DataFrame(lk.get("checks", []))[["name", "passed", "detail"]]
        parts.append(_table(lk_df, f"Leakage suite on the real snapshot ({lk.get('ran_at', '')})"))
    counts: dict[str, int] = {}
    for r in inp.research_log:
        counts[r.get("family", "?")] = counts.get(r.get("family", "?"), 0) + 1
    parts.append(
        _table(
            pd.DataFrame([{"family": k, "trials": v} for k, v in sorted(counts.items())]),
            "research_log: out-of-sample configurations evaluated",
            "The Deflated Sharpe Ratio above reads its trial count N from this file. If a configuration was "
            "evaluated and is not here, no result that depends on it is valid.",
        )
    )
    parts.append(_caveats(inp))
    parts.append("</body></html>")
    return "\n".join(parts)


def _headline_ic_series(inp: ReportInputs, name: str) -> pd.Series | None:
    d = _run_dir_for(inp, name)
    if d is None:
        return None
    f = Path(d) / "per_date.parquet"
    if not f.exists():
        return None
    return pd.read_parquet(f)["ic"]


def _per_year_ic(inp: ReportInputs, name: str) -> pd.DataFrame | None:
    ic = _headline_ic_series(inp, name)
    if ic is None or not len(ic):
        return None
    ic = ic.dropna()
    g = ic.groupby(ic.index.year).agg(["mean", "std", "size"])
    g.columns = ["ic_mean", "ic_std", "n_days"]
    g["icir"] = g["ic_mean"] / g["ic_std"]
    return g.reset_index(names="year")


def _run_dir_for(inp: ReportInputs, name: str) -> str | None:
    if not name:
        return None
    key = name.replace(" ", "_")
    for run_name, s in inp.run_summaries.items():
        if run_name.startswith(key):
            return str(s.get("_dir"))
    return None


def _equity_curves(inp: ReportInputs, headline_name: str) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    wanted = [
        (headline_name, headline_name),
        ("B2_equal_weight", "B2 equal-weight"),
        ("B1_nifty_buy_hold", "B1 NIFTY"),
        ("B5_reversal_1d", "B5 reversal"),
    ]
    for key, label in wanted:
        d = _run_dir_for(inp, key)
        if d is None:
            continue
        f = Path(d) / "daily.parquet"
        if f.exists():
            daily = pd.read_parquet(f)
            out[label] = daily.set_index("date")["equity"]
    return out


def _caveats(inp: ReportInputs) -> str:
    cfg = inp.cfg
    dropped = ", ".join(f"{k} ({v})" for k, v in cfg.universe.dropped.items()) or "none"
    return (
        "<h3>Caveats that apply to every number on this page</h3><ul>"
        f"<li><b>Survivorship.</b> Universe fixed as of {html.escape(cfg.universe.fixed_as_of)}; backtest "
        "results are subject to survivorship bias of unquantified magnitude. These are "
        "today's survivors.</li>"
        f"<li><b>Closing-auction proxy.</b> {html.escape(cfg.calendar.close_proxy_note.strip())}</li>"
        "<li><b>Dividends.</b> Positions are valued on the adjusted close, which treats every dividend as "
        "reinvested at the close. Real cash dividends arrive later and are taxed differently.</li>"
        f"<li><b>Dropped symbols.</b> {html.escape(dropped)}</li>"
        "<li><b>Effective sample size.</b> Fifty stocks on one day are not fifty independent observations. "
        "With an average pairwise correlation near 0.3 the effective cross-sectional sample is closer to 3, "
        "so every t-statistic here is computed on the daily series, never on the pooled rows.</li>"
        f"<li><b>Charges.</b> Rates dated {html.escape(cfg.charges.as_of)} from "
        f"<a href='{html.escape(cfg.charges.source)}'>{html.escape(cfg.charges.source)}</a>. "
        "Re-verify before reuse.</li>"
        "<li><b>Paper only.</b> This repository contains no broker integration and no "
        "order-placement code.</li>"
        "</ul>"
    )


_CSS = """<style>
:root { color-scheme: light; }
body { font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif; margin: 0 auto;
       padding: 2rem 1.25rem 4rem; max-width: 1080px; color: #1b1b1b; background: #fbfbfa; line-height: 1.5; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; letter-spacing: -0.01em; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .75rem; padding-bottom: .3rem; border-bottom: 2px solid #1b1b1b; }
h3 { font-size: .95rem; margin: 1.6rem 0 .4rem; }
p { margin: .4rem 0; }
.muted { color: #63635e; font-size: .82rem; }
.meta { color: #63635e; font-size: .78rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        word-break: break-word; }
.breakeven { background: #fff5e6; border-left: 5px solid #d08700; padding: .8rem 1rem; margin: 1.2rem 0;
             font-size: .88rem; border-radius: 3px; }
.verdictline { background: #1b1b1b; color: #fbfbfa; padding: 1rem 1.15rem; margin: 1rem 0 2rem;
               font-size: .95rem; border-radius: 3px; }
table { border-collapse: collapse; font-size: .8rem; width: 100%; margin: .4rem 0 1rem; background: #fff; }
th, td { padding: .35rem .55rem; border-bottom: 1px solid #e6e6e2; text-align: right; vertical-align: top; }
th { text-align: left; background: #f2f2ee; font-weight: 600; position: sticky; top: 0; }
td:first-child, th:first-child { text-align: left; }
table.verdict td { font-size: .86rem; }
table.verdict td.num { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.ok { color: #1a7f45; font-weight: 700; text-align: center; }
.bad { color: #b3452f; font-weight: 700; text-align: center; }
img { max-width: 100%; height: auto; display: block; margin: .6rem 0 1.2rem; }
ul { font-size: .85rem; padding-left: 1.1rem; }
li { margin: .3rem 0; }
a { color: #1f6f8b; }
</style>"""


def write_report(inp: ReportInputs, *, run_id: str) -> Path:
    out = inp.paths.reports_dir / f"{run_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(inp, run_id=run_id), encoding="utf-8")
    return out


def verdict_lines(inp: ReportInputs) -> list[str]:
    """The same verdict block as plain text, for the terminal and for the phase report."""
    name, headline = pick_headline(inp)
    crit = build_criteria(inp, headline)
    lines = [f"GREENCLOSE verdict  model {name or 'n/a'}  snapshot {inp.snapshot_id}"]
    for c in crit:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"  [{mark}] {_strip(c.label)}: {_strip(c.value)}")
    lines.append(f"VERDICT: {_strip(overall_verdict(crit, inp, headline))}")
    return lines


def _strip(s: str) -> str:
    return html.unescape(s)


__all__ = ["ReportInputs", "gather", "render", "verdict_lines", "write_report"]


def _unused_guard() -> None:  # pragma: no cover - keeps numpy import meaningful for type checkers
    _ = np.nan
