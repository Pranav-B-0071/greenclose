"""Primary data source: yfinance daily OHLCV with `auto_adjust=False` so both Close and Adj Close arrive.

Conventions (brief section 4.1):
- returns are computed from `adj_close` (splits, bonuses, dividends);
- price levels and cost calculations use raw `close`. Note Yahoo's raw Close is split-adjusted but not
  dividend-adjusted, which is what percentage-based charges need;
- both columns are kept and never mixed.

If the source is unreachable this module fails loudly. It never produces an empty dataset silently and
never tries to route around a network policy.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import pandas as pd

BAR_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]


class DataUnavailableError(RuntimeError):
    """Raised when the primary source cannot deliver data. The message names the CSV fallback."""


def canonicalise_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical long frame: one row per (symbol, trading date), sorted, typed, de-duplicated.

    `date` is the trading date in Asia/Kolkata stored as a tz-naive midnight timestamp. Only event
    timestamps (snapshot manifests, logs) are stored as tz-aware UTC.
    """
    out = df.loc[:, BAR_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], utc=False)
    if getattr(out["date"].dt, "tz", None) is not None:
        out["date"] = out["date"].dt.tz_localize(None)
    out["date"] = out["date"].dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    for c in ["open", "high", "low", "close", "adj_close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).round().astype("int64")
    out = out.dropna(subset=["close", "adj_close"])
    out = out.drop_duplicates(subset=["symbol", "date"], keep="last")
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    return out


def _wide_to_long(
    raw: pd.DataFrame, tickers: Sequence[str], ticker_to_symbol: dict[str, str]
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    if not isinstance(raw.columns, pd.MultiIndex):
        # single ticker without a ticker level
        raw = pd.concat({tickers[0]: raw}, axis=1)
    present = set(raw.columns.get_level_values(0))
    frames: list[pd.DataFrame] = []
    for t in tickers:
        if t not in present:
            continue
        sub = raw[t].copy()
        sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
        if "adj_close" not in sub.columns:
            raise DataUnavailableError(
                f"{t}: source returned no 'Adj Close' column; fetch must use auto_adjust=False"
            )
        sub = sub.dropna(subset=["close"])
        if sub.empty:
            continue
        sub = sub.reset_index()
        sub = sub.rename(columns={sub.columns[0]: "date"})
        sub["symbol"] = ticker_to_symbol[t]
        frames.append(sub[BAR_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return canonicalise_bars(pd.concat(frames, ignore_index=True))


def fetch_bars(
    tickers: Sequence[str],
    ticker_to_symbol: dict[str, str],
    *,
    start: str,
    end: str,
    chunk_size: int = 10,
    pause_s: float = 1.0,
    retries: int = 4,
) -> tuple[pd.DataFrame, list[str]]:
    """Download daily bars. Returns (canonical long frame, symbols that returned nothing)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise DataUnavailableError("yfinance is not installed") from exc

    frames: list[pd.DataFrame] = []
    got: set[str] = set()
    last_error: Exception | None = None
    remaining = list(tickers)
    size = chunk_size
    # yfinance does not raise for a ticker that fails (rate limits, connection resets); it logs and
    # returns nothing for it. So: fetch, see what came back, retry the rest in ever smaller batches.
    for attempt in range(retries + 1):
        for i in range(0, len(remaining), size):
            batch = remaining[i : i + size]
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                    actions=False,
                )
                long = _wide_to_long(raw, batch, ticker_to_symbol)
                if not long.empty:
                    frames.append(long)
                    got.update(long["symbol"].unique().tolist())
            except DataUnavailableError:
                raise
            except Exception as exc:  # network / parsing errors: retry, then report
                last_error = exc
            time.sleep(pause_s)
        remaining = [t for t in tickers if ticker_to_symbol[t] not in got]
        if not remaining:
            break
        size = max(1, size // 2)
        time.sleep(pause_s * 4 * (attempt + 1))

    if not frames:
        raise DataUnavailableError(
            "yfinance returned no data for any ticker. The network may block Yahoo Finance. "
            "Supply data with `greenclose fetch --from-csv <dir>` "
            f"(last error: {last_error!r})"
        )
    bars = canonicalise_bars(pd.concat(frames, ignore_index=True))
    failed = [ticker_to_symbol[t] for t in tickers if ticker_to_symbol[t] not in got]
    return bars, failed
