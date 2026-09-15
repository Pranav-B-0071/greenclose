"""`--from-csv` fallback: per-symbol CSVs with columns date,open,high,low,close,adj_close,volume.

File names use `safe_name(symbol)` (M&M -> M_AND_M.csv, ^NSEI -> IDX_NSEI.csv).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from greenclose.ingest.universe import safe_name
from greenclose.ingest.yahoo import BAR_COLUMNS, DataUnavailableError, canonicalise_bars

REQUIRED = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def load_csv_dir(directory: Path, symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Load the listed symbols from `directory`. Returns (canonical frame, symbols with no file)."""
    if not directory.is_dir():
        raise DataUnavailableError(f"--from-csv directory does not exist: {directory}")
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for sym in symbols:
        path = directory / f"{safe_name(sym)}.csv"
        if not path.exists():
            missing.append(sym)
            continue
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        absent = [c for c in REQUIRED if c not in df.columns]
        if absent:
            raise DataUnavailableError(f"{path}: missing columns {absent}; required {REQUIRED}")
        df["symbol"] = sym
        frames.append(df[BAR_COLUMNS])
    if not frames:
        raise DataUnavailableError(f"No CSV files for the requested symbols found in {directory}")
    return canonicalise_bars(pd.concat(frames, ignore_index=True)), missing
