"""Data validation report (brief section 4.4). Mandatory before any feature is built.

Produces `reports/data_quality.html` and a verdict: "N symbols pass, M flagged, K should be dropped".
Symbols that should be dropped are moved to `dropped:` in config/universe.yaml with the reason.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from greenclose.ingest.clean import remove_phantom_days
from greenclose.ingest.yahoo import canonicalise_bars

BIG_MOVE = 0.20
HUGE_MOVE = 0.50  # a single unadjusted move this large is a corporate action the source did not adjust
SPLIT_RATIOS = (2.0, 5.0, 10.0, 0.5, 0.2, 0.1)
SPLIT_TOL = 0.03


@dataclass(slots=True)
class ValidationResult:
    snapshot_id: str
    calendar: pd.DataFrame
    coverage: pd.DataFrame
    suspicious: pd.DataFrame
    splits: pd.DataFrame
    index_coverage: pd.DataFrame
    vix_usable: bool
    phantom_days: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)
    flagged: dict[str, list[str]] = field(default_factory=dict)
    drop: dict[str, str] = field(default_factory=dict)

    @property
    def verdict_line(self) -> str:
        n_pass, n_flag, n_drop = len(self.passed), len(self.flagged), len(self.drop)
        return f"{n_pass} symbols pass, {n_flag} symbols flagged, {n_drop} symbols should be dropped."


def reconstruct_calendar(bars: pd.DataFrame, min_fraction: float = 0.5) -> pd.DatetimeIndex:
    """Trading days = dates on which at least `min_fraction` of the symbols listed at that time have a bar."""
    counts = bars.groupby("date")["symbol"].nunique()
    first = bars.groupby("symbol")["date"].min()
    last = bars.groupby("symbol")["date"].max()
    dates = counts.index
    listed = np.array([int(((first <= d) & (last >= d)).sum()) for d in dates])
    keep = counts.to_numpy() >= np.maximum(1, np.floor(min_fraction * listed))
    return pd.DatetimeIndex(dates[keep])


def calendar_table(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    per_year = pd.Series(1, index=calendar).groupby(calendar.year).sum()
    df = pd.DataFrame({"year": per_year.index, "trading_days": per_year.to_numpy()})
    full_years = df.iloc[1:-1] if len(df) > 2 else df
    median = float(full_years["trading_days"].median()) if len(full_years) else float("nan")
    df["flag"] = ""
    is_partial = (df["year"] == df["year"].min()) | (df["year"] == df["year"].max())
    anomalous = (~is_partial) & ((df["trading_days"] < median - 12) | (df["trading_days"] > median + 12))
    df.loc[anomalous, "flag"] = "anomalous count"
    df.loc[is_partial, "flag"] = "partial year"
    return df


def per_symbol_coverage(bars: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    cal = calendar.to_numpy()
    for sym, g in bars.groupby("symbol"):
        present = g["date"].to_numpy()
        lo, hi = present.min(), present.max()
        expected = cal[(cal >= lo) & (cal <= hi)]
        missing_mask = ~np.isin(expected, present)
        n_missing = int(missing_mask.sum())
        longest = 0
        run = 0
        for m in missing_mask:
            run = run + 1 if m else 0
            longest = max(longest, run)
        years = (pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25
        rows.append(
            {
                "symbol": sym,
                "first": pd.Timestamp(lo).date(),
                "last": pd.Timestamp(hi).date(),
                "years": round(years, 2),
                "n_bars": int(len(present)),
                "n_expected": int(len(expected)),
                "n_missing": n_missing,
                "pct_missing": round(100.0 * n_missing / max(1, len(expected)), 3),
                "longest_gap": longest,
                "ends_early": bool(pd.Timestamp(hi) < calendar.max() - pd.Timedelta(days=7)),
            }
        )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def suspicious_prints(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.sort_values(["symbol", "date"]).copy()
    b["adj_ret"] = b.groupby("symbol")["adj_close"].pct_change()
    b["raw_ret"] = b.groupby("symbol")["close"].pct_change()
    out: list[pd.DataFrame] = []

    big = b.loc[b["adj_ret"].abs() > BIG_MOVE, ["symbol", "date", "adj_ret", "raw_ret"]].copy()
    big["kind"] = np.where(
        big["adj_ret"].abs() > HUGE_MOVE, "abs_return_gt_50pct_unadjusted", "abs_return_gt_20pct"
    )
    big["detail"] = big.apply(lambda r: f"adj_ret={r.adj_ret:+.1%} raw_ret={r.raw_ret:+.1%}", axis=1)
    out.append(big[["symbol", "date", "kind", "detail"]])

    zero = b.loc[b["volume"] <= 0, ["symbol", "date"]].copy()
    zero["kind"] = "zero_volume"
    zero["detail"] = ""
    out.append(zero)

    hl = b.loc[b["high"] < b["low"], ["symbol", "date", "high", "low"]].copy()
    hl["kind"] = "high_lt_low"
    hl["detail"] = hl.apply(lambda r: f"high={r.high} low={r.low}", axis=1)
    out.append(hl[["symbol", "date", "kind", "detail"]])

    tol = 1e-3
    outside = b.loc[
        (b["close"] > b["high"] * (1 + tol)) | (b["close"] < b["low"] * (1 - tol)),
        ["symbol", "date", "close", "high", "low"],
    ].copy()
    outside["kind"] = "close_outside_range"
    outside["detail"] = outside.apply(lambda r: f"close={r.close} range=[{r.low},{r.high}]", axis=1)
    out.append(outside[["symbol", "date", "kind", "detail"]])

    nonpos = b.loc[
        (b[["open", "high", "low", "close", "adj_close"]] <= 0).any(axis=1), ["symbol", "date"]
    ].copy()
    nonpos["kind"] = "non_positive_price"
    nonpos["detail"] = ""
    out.append(nonpos)

    res = pd.concat(out, ignore_index=True)
    return res.sort_values(["symbol", "date"]).reset_index(drop=True)


def split_artefacts(bars: pd.DataFrame) -> pd.DataFrame:
    """Raw-close ratio jumps near round split ratios, and adj_factor steps (corporate-action detector)."""
    b = bars.sort_values(["symbol", "date"]).copy()
    b["raw_ratio"] = b["close"] / b.groupby("symbol")["close"].shift(1)
    b["adj_factor"] = b["adj_close"] / b["close"]
    b["af_step"] = b["adj_factor"] / b.groupby("symbol")["adj_factor"].shift(1) - 1.0
    near = np.zeros(len(b), dtype=bool)
    for r in SPLIT_RATIOS:
        near |= (b["raw_ratio"] - r).abs() <= SPLIT_TOL * r
    rows = b.loc[near, ["symbol", "date", "raw_ratio"]].copy()
    rows["kind"] = "raw_close_ratio_near_split"
    rows["detail"] = rows["raw_ratio"].map(lambda x: f"ratio={x:.3f}")
    steps = b.loc[b["af_step"].abs() > 0.005, ["symbol", "date", "af_step"]].copy()
    steps["kind"] = "adj_factor_step"
    steps["detail"] = steps["af_step"].map(lambda x: f"adj_factor step={x:+.2%} (dividend/split adjustment)")
    res = pd.concat(
        [rows[["symbol", "date", "kind", "detail"]], steps[["symbol", "date", "kind", "detail"]]],
        ignore_index=True,
    )
    return res.sort_values(["symbol", "date"]).reset_index(drop=True)


def index_coverage(
    index: pd.DataFrame, calendar: pd.DatetimeIndex, vix_symbol: str | None, min_cov: float
) -> tuple[pd.DataFrame, bool]:
    rows = []
    vix_ok = False
    for sym, g in index.groupby("symbol"):
        present = g["date"].to_numpy()
        cov = float(np.isin(calendar.to_numpy(), present).mean()) if len(calendar) else 0.0
        rows.append(
            {
                "symbol": sym,
                "first": pd.Timestamp(present.min()).date(),
                "last": pd.Timestamp(present.max()).date(),
                "n_bars": len(present),
                "coverage_of_calendar": round(cov, 4),
            }
        )
        if vix_symbol and sym == vix_symbol:
            vix_ok = cov >= min_cov
    return pd.DataFrame(rows), vix_ok


def decide_verdict(
    coverage: pd.DataFrame,
    suspicious: pd.DataFrame,
    *,
    min_years: float = 3.0,
    max_pct_missing: float = 5.0,
    max_big_moves: int = 5,
) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    passed: list[str] = []
    flagged: dict[str, list[str]] = {}
    drop: dict[str, str] = {}
    counts = (
        suspicious.groupby(["symbol", "kind"]).size().unstack(fill_value=0)
        if len(suspicious)
        else pd.DataFrame()
    )
    for _, r in coverage.iterrows():
        sym = str(r["symbol"])
        reasons: list[str] = []
        fatal: list[str] = []
        if r["years"] < min_years:
            fatal.append(f"only {r['years']} years of history (< {min_years})")
        if r["pct_missing"] > max_pct_missing:
            fatal.append(f"{r['pct_missing']}% trading days missing (> {max_pct_missing}%)")
        if r["ends_early"]:
            fatal.append(f"series ends early on {r['last']} (delisted/renamed?)")
        c = counts.loc[sym] if sym in counts.index else pd.Series(dtype=int)
        huge = int(c.get("abs_return_gt_50pct_unadjusted", 0))
        if huge:
            fatal.append(
                f"{huge} single-day |return|>50% present in both raw and adjusted series "
                "(corporate action not adjusted by the source, e.g. a demerger)"
            )
        big = int(c.get("abs_return_gt_20pct", 0))
        if big > max_big_moves:
            fatal.append(f"{big} unexplained |return|>20% prints (> {max_big_moves})")
        elif big:
            reasons.append(f"{big} |return|>20% prints")
        for kind in ("zero_volume", "high_lt_low", "close_outside_range", "non_positive_price"):
            n = int(c.get(kind, 0))
            if n:
                reasons.append(f"{n} {kind}")
        if 0 < r["pct_missing"] <= max_pct_missing and r["pct_missing"] > 1.0:
            reasons.append(f"{r['pct_missing']}% days missing")
        if fatal:
            drop[sym] = "; ".join(fatal)
        elif reasons:
            flagged[sym] = reasons
        else:
            passed.append(sym)
    return passed, flagged, drop


def validate(
    bars: pd.DataFrame,
    index: pd.DataFrame,
    *,
    snapshot_id: str,
    vix_symbol: str | None,
    vix_min_coverage: float,
    expected_symbols: list[str] | None = None,
) -> ValidationResult:
    bars, phantom = remove_phantom_days(canonicalise_bars(bars))
    if len(index):
        index = canonicalise_bars(index)
        index = index.loc[~index["date"].isin(phantom)].reset_index(drop=True)
    calendar = reconstruct_calendar(bars)
    coverage = per_symbol_coverage(bars, calendar)
    suspicious = suspicious_prints(bars)
    splits = split_artefacts(bars)
    idx_cov, vix_ok = (
        index_coverage(index, calendar, vix_symbol, vix_min_coverage)
        if len(index)
        else (pd.DataFrame(), False)
    )
    passed, flagged, drop = decide_verdict(coverage, suspicious)
    for sym in expected_symbols or []:
        if sym not in set(coverage["symbol"]):
            drop[sym] = "no bars in snapshot: the source returned nothing for this ticker"
    return ValidationResult(
        snapshot_id=snapshot_id,
        calendar=calendar_table(calendar),
        coverage=coverage,
        suspicious=suspicious,
        splits=splits,
        index_coverage=idx_cov,
        vix_usable=vix_ok,
        phantom_days=[str(d.date()) for d in phantom],
        passed=passed,
        flagged=flagged,
        drop=drop,
    )


def render_html(res: ValidationResult) -> str:
    def table(df: pd.DataFrame, title: str, max_rows: int = 400) -> str:
        if df is None or len(df) == 0:
            return f"<h2>{html.escape(title)}</h2><p>none</p>"
        note = f"<p>showing first {max_rows} of {len(df)} rows</p>" if len(df) > max_rows else ""
        return f"<h2>{html.escape(title)}</h2>{note}{df.head(max_rows).to_html(index=False, border=0)}"

    vix_text = "yes" if res.vix_usable else "no (omitted, not approximated)"
    flagged_rows = pd.DataFrame([{"symbol": s, "reasons": "; ".join(r)} for s, r in res.flagged.items()])
    drop_rows = pd.DataFrame([{"symbol": s, "reason": r} for s, r in res.drop.items()])
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>GREENCLOSE data quality</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}"
        "table{border-collapse:collapse;font-size:12px}"
        "td,th{padding:3px 8px;border-bottom:1px solid #ddd;text-align:right}th{text-align:left}"
        ".verdict{font-size:1.2rem;font-weight:600;padding:1rem;background:#f4f4f4;"
        "border-left:6px solid #333}</style></head><body>",
        f"<h1>GREENCLOSE data quality report</h1><p>snapshot <code>{html.escape(res.snapshot_id)}</code></p>",
        f"<div class='verdict'>VERDICT: {html.escape(res.verdict_line)}</div>",
        f"<p>India VIX usable as a feature: <b>{vix_text}</b></p>",
        f"<p>Phantom holiday bars removed from the calendar ({len(res.phantom_days)} dates): "
        f"{html.escape(', '.join(res.phantom_days)) or 'none'}</p>",
        table(drop_rows, "Symbols to drop"),
        table(flagged_rows, "Symbols flagged (kept)"),
        table(res.calendar, "Trading-day calendar reconstructed from the data"),
        table(res.coverage, "Per-symbol coverage"),
        table(res.index_coverage, "Index series coverage"),
        table(res.suspicious, "Suspicious prints"),
        table(res.splits, "Split / bonus / dividend artefacts (adjustment detector)"),
        "</body></html>",
    ]
    return "\n".join(parts)


def apply_drops(universe_yaml: Path, drop: dict[str, str]) -> None:
    """Rewrite the `dropped:` block of config/universe.yaml, leaving everything else untouched."""
    text = universe_yaml.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("dropped:")), None)
    if start is None:
        raise ValueError("universe.yaml has no `dropped:` key")
    end = start + 1
    while end < len(lines) and (lines[end].startswith(" ") or lines[end].strip() == ""):
        end += 1
    block = (
        ["dropped: {}"]
        if not drop
        else ["dropped:"] + [f"  {s}: {_yaml_str(r)}" for s, r in sorted(drop.items())]
    )
    new_lines = lines[:start] + block + lines[end:]
    universe_yaml.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _yaml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
