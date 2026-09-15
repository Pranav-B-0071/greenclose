"""Fetch -> snapshot -> drift diff. Shared by the `fetch` CLI and the forward runner."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from greenclose.config import AppConfig, Paths
from greenclose.ingest.csv_source import load_csv_dir
from greenclose.ingest.snapshots import (
    DriftReport,
    diff_bars,
    latest_snapshot,
    log_drift,
    read_bars,
    read_manifest,
    write_snapshot,
)
from greenclose.ingest.yahoo import BAR_COLUMNS, DataUnavailableError, fetch_bars
from greenclose.ops.logging import get_logger

log = get_logger(__name__)


def take_snapshot(
    cfg: AppConfig,
    paths: Paths,
    *,
    from_csv: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    notes: str = "",
    max_failed: int = 2,
) -> tuple[Path, DriftReport | None]:
    """Acquire bars for the whole universe (dropped symbols included, so validation can see them).

    Refuses to write a snapshot when more than `max_failed` symbols are missing: a partial snapshot
    must never become the baseline that later snapshots are diffed against.
    """
    symbols = cfg.universe.all_symbols
    index_symbols = [cfg.base.data.benchmark] + ([cfg.base.data.vix] if cfg.base.data.vix else [])
    start = start or cfg.base.data.start
    end = end or str(date.today() + timedelta(days=1))  # yfinance `end` is exclusive: include today

    if from_csv is not None:
        bars, failed = load_csv_dir(from_csv, symbols)
        try:
            index, _ = load_csv_dir(from_csv, index_symbols)
        except Exception:  # index series are optional in CSV mode
            index = pd.DataFrame(columns=BAR_COLUMNS)
        source = f"csv:{from_csv}"
    else:
        t2s = {cfg.universe.ticker(s): s for s in symbols}
        bars, failed = fetch_bars(list(t2s), t2s, start=start, end=end)
        i2s = {s: s for s in index_symbols}
        try:
            index, _ = fetch_bars(index_symbols, i2s, start=start, end=end)
        except Exception as exc:  # benchmark failure is loud but not fatal for the snapshot
            log.error("index_fetch_failed", error=str(exc))
            index = pd.DataFrame(columns=BAR_COLUMNS)
        source = "yfinance"

    if len(failed) > max_failed:
        raise DataUnavailableError(
            f"{len(failed)} of {len(symbols)} symbols returned no data ({failed}); refusing to write a "
            f"partial snapshot. Retry later or supply the missing series with --from-csv."
        )

    root = paths.data_dir / "snapshots"
    previous = latest_snapshot(root)
    target = write_snapshot(
        root,
        bars,
        index,
        source=source,
        failed_symbols=failed,
        notes=notes,
        extra={"start": start, "end": end},
    )
    report: DriftReport | None = None
    if previous is not None:
        report = diff_bars(
            read_bars(previous),
            bars,
            old_id=read_manifest(previous).snapshot_id,
            new_id=read_manifest(target).snapshot_id,
        )
        log_drift(root, report)
    return target, report


def coverage_table(bars: pd.DataFrame) -> pd.DataFrame:
    g = bars.groupby("symbol")["date"]
    out = pd.DataFrame({"first": g.min(), "last": g.max(), "n_bars": g.size()})
    out["years"] = (out["last"] - out["first"]).dt.days / 365.25
    return out.sort_values("first")
