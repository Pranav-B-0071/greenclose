"""Deterministic cleaning applied to every snapshot when it is loaded.

Free daily sources emit *phantom bars* on some exchange holidays: every symbol gets a row with
open == close == previous close and zero volume. Left in place they would become fake trading days
with zero returns. A date is a phantom day when at least `zero_volume_fraction` of the symbols that
have a bar on it show zero volume. Genuine thin sessions (Muhurat trading) never come close to that.
"""

from __future__ import annotations

import pandas as pd

ZERO_VOLUME_FRACTION = 0.9
MIN_SYMBOLS = 5


def phantom_days(bars: pd.DataFrame, zero_volume_fraction: float = ZERO_VOLUME_FRACTION) -> pd.DatetimeIndex:
    if bars.empty:
        return pd.DatetimeIndex([])
    g = bars.groupby("date")
    n = g["symbol"].nunique()
    zero = g["volume"].apply(lambda v: int((v <= 0).sum()))
    frac = zero / n
    mask = (frac >= zero_volume_fraction) & (n >= MIN_SYMBOLS)
    return pd.DatetimeIndex(frac.index[mask.to_numpy()])


def remove_phantom_days(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Return (bars without phantom days, the dates removed)."""
    days = phantom_days(bars)
    if len(days) == 0:
        return bars, days
    return bars.loc[~bars["date"].isin(days)].reset_index(drop=True), days
