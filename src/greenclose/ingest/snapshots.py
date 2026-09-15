"""Immutable, content-hashed data snapshots with drift detection (brief section 4.2).

data/snapshots/<YYYY-MM-DDTHHMMSS>/
    bars.parquet      canonical long frame for the universe
    index.parquet     benchmark (and optional VIX) rows in the same layout
    manifest.json     {snapshot_id, taken_at, source, content_sha256, row_count, symbols, ...}

Every new snapshot is diffed against the previous one. Any change to a *historical* bar is appended to
data/snapshots/drift.log and written in full to drift_<snapshot_id>.parquet. On a free source, retroactive
corporate-action adjustment shows up here, which makes the log a free corporate-action detector.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from greenclose.ingest.yahoo import BAR_COLUMNS, canonicalise_bars

SNAPSHOT_FMT = "%Y-%m-%dT%H%M%S"
NUMERIC = ["open", "high", "low", "close", "adj_close", "volume"]
RAW_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
# The source computes adj_close in float32: successive fetches differ at the ~1e-7 relative level
# (every difference is a multiple of 2^-14). Real adjustments (dividends, splits) move adj_close by
# >= 1e-3 relative, so this tolerance hides the quantisation noise without hiding corporate actions.
ADJ_CLOSE_RTOL = 1e-5


@dataclass(slots=True)
class Manifest:
    snapshot_id: str
    taken_at: str
    source: str
    content_sha256: str
    bars_sha256: str
    index_sha256: str
    row_count: int
    index_row_count: int
    symbols: list[str]
    failed_symbols: list[str]
    start: str
    end: str
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)
    raw_sha256: str = ""  # over date/symbol/OHLC/volume only; identical across re-fetches of unchanged data

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_path(cls, path: Path) -> Manifest:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def frame_sha256(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """SHA-256 over the canonical, sorted CSV serialisation. Full float precision so nothing hides."""
    canon = canonicalise_bars(df) if len(df) else pd.DataFrame(columns=BAR_COLUMNS)
    if columns is not None:
        canon = canon.loc[:, columns]
    payload = canon.to_csv(index=False, float_format="%.17g", lineterminator="\n", date_format="%Y-%m-%d")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def raw_sha256(df: pd.DataFrame) -> str:
    """Hash of the source's raw data only (no adj_close). This is the re-fetch identity check."""
    return frame_sha256(df, RAW_COLUMNS)


def list_snapshots(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def latest_snapshot(root: Path) -> Path | None:
    snaps = list_snapshots(root)
    return snaps[-1] if snaps else None


def resolve_snapshot(root: Path, snapshot_id: str | None) -> Path:
    if snapshot_id is None or snapshot_id == "latest":
        latest = latest_snapshot(root)
        if latest is None:
            raise FileNotFoundError(f"No snapshots under {root}. Run `greenclose fetch` first.")
        return latest
    path = root / snapshot_id
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"Snapshot {snapshot_id} not found under {root}")
    return path


def read_manifest(snapshot_dir: Path) -> Manifest:
    return Manifest.from_path(snapshot_dir / "manifest.json")


def read_bars(snapshot_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(snapshot_dir / "bars.parquet")


def read_index(snapshot_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(snapshot_dir / "index.parquet")


def write_snapshot(
    root: Path,
    bars: pd.DataFrame,
    index: pd.DataFrame,
    *,
    source: str,
    failed_symbols: list[str],
    notes: str = "",
    extra: dict[str, str] | None = None,
    taken_at: datetime | None = None,
) -> Path:
    bars = canonicalise_bars(bars)
    index = canonicalise_bars(index) if len(index) else pd.DataFrame(columns=BAR_COLUMNS)
    taken = taken_at or datetime.now(UTC)
    snapshot_id = taken.strftime(SNAPSHOT_FMT)
    target = root / snapshot_id
    if target.exists():
        raise FileExistsError(f"Snapshot {snapshot_id} already exists; snapshots are immutable")
    target.mkdir(parents=True)
    bars_hash = frame_sha256(bars)
    index_hash = frame_sha256(index)
    content = hashlib.sha256((bars_hash + index_hash).encode()).hexdigest()
    bars.to_parquet(target / "bars.parquet", index=False)
    index.to_parquet(target / "index.parquet", index=False)
    manifest = Manifest(
        snapshot_id=snapshot_id,
        taken_at=taken.replace(microsecond=0).isoformat(),
        source=source,
        content_sha256=content,
        bars_sha256=bars_hash,
        index_sha256=index_hash,
        row_count=int(len(bars)),
        raw_sha256=raw_sha256(bars),
        index_row_count=int(len(index)),
        symbols=sorted(bars["symbol"].unique().tolist()),
        failed_symbols=sorted(failed_symbols),
        start=str(bars["date"].min().date()) if len(bars) else "",
        end=str(bars["date"].max().date()) if len(bars) else "",
        notes=notes,
        extra=extra or {},
    )
    (target / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return target


@dataclass(slots=True)
class DriftReport:
    old_id: str
    new_id: str
    summary: pd.DataFrame  # one row per (symbol, column): n_changed, first/last date, max_rel_change
    detail: pd.DataFrame  # one row per changed (symbol, date, column) with old/new values
    removed_bars: pd.DataFrame  # (symbol, date) present before, absent now
    added_symbols: list[str]
    removed_symbols: list[str]

    @property
    def changed(self) -> bool:
        return bool(len(self.detail) or len(self.removed_bars) or self.removed_symbols)

    def lines(self) -> list[str]:
        out = [f"drift {self.old_id} -> {self.new_id}: {len(self.detail)} changed historical values"]
        for _, r in self.summary.iterrows():
            sym, col, n = r["symbol"], r["column"], int(r["n_changed"])
            first, last, rel = r["first_date"], r["last_date"], float(r["max_rel_change"])
            out.append(f"  {sym:<12} {col:<9} n={n:>5} {first}..{last} max_rel={rel:.3e}")
        if len(self.removed_bars):
            out.append(f"  removed bars: {len(self.removed_bars)} (see detail parquet)")
        if self.added_symbols:
            out.append(f"  added symbols: {self.added_symbols}")
        if self.removed_symbols:
            out.append(f"  removed symbols: {self.removed_symbols}")
        return out


def diff_bars(old: pd.DataFrame, new: pd.DataFrame, *, old_id: str, new_id: str) -> DriftReport:
    """Compare bars on the dates both frames contain. Only *historical* changes are reported."""
    old = canonicalise_bars(old)
    new = canonicalise_bars(new)
    keys = ["symbol", "date"]
    merged = old.merge(new, on=keys, how="left", suffixes=("_old", "_new"), indicator=True)
    removed = merged.loc[merged["_merge"] == "left_only", keys].reset_index(drop=True)
    both = merged.loc[merged["_merge"] == "both"]
    details: list[pd.DataFrame] = []
    for col in NUMERIC:
        a = both[f"{col}_old"].to_numpy(dtype="float64")
        b = both[f"{col}_new"].to_numpy(dtype="float64")
        if col == "volume":
            changed = a != b
        elif col == "adj_close":
            changed = ~np.isclose(a, b, rtol=ADJ_CLOSE_RTOL, atol=0.0)
        else:
            changed = ~np.isclose(a, b, rtol=1e-9, atol=1e-9)
        if changed.any():
            d = both.loc[changed, keys].copy()
            d["column"] = col
            d["old"] = a[changed]
            d["new"] = b[changed]
            denom = np.where(a[changed] == 0, np.nan, np.abs(a[changed]))
            with np.errstate(divide="ignore", invalid="ignore"):
                d["rel_change"] = np.abs(b[changed] - a[changed]) / denom
            details.append(d)
    if details:
        detail = pd.concat(details, ignore_index=True)
        summary = (
            detail.groupby(["symbol", "column"], sort=True)
            .agg(
                n_changed=("date", "size"),
                first_date=("date", "min"),
                last_date=("date", "max"),
                max_rel_change=("rel_change", "max"),
            )
            .reset_index()
        )
        summary["first_date"] = summary["first_date"].dt.strftime("%Y-%m-%d")
        summary["last_date"] = summary["last_date"].dt.strftime("%Y-%m-%d")
        summary["max_rel_change"] = summary["max_rel_change"].fillna(0.0)
    else:
        detail = pd.DataFrame(columns=[*keys, "column", "old", "new", "rel_change"])
        summary = pd.DataFrame(
            columns=["symbol", "column", "n_changed", "first_date", "last_date", "max_rel_change"]
        )
    old_syms, new_syms = set(old["symbol"]), set(new["symbol"])
    return DriftReport(
        old_id=old_id,
        new_id=new_id,
        summary=summary,
        detail=detail,
        removed_bars=removed,
        added_symbols=sorted(new_syms - old_syms),
        removed_symbols=sorted(old_syms - new_syms),
    )


def log_drift(root: Path, report: DriftReport) -> Path:
    log_path = root / "drift.log"
    stamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] " + "\n".join(report.lines()) + "\n")
    if report.changed:
        report.detail.to_parquet(root / f"drift_{report.new_id}.parquet", index=False)
    return log_path
