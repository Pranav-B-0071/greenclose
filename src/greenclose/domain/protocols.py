"""Structural interfaces. The *shape* of these signatures is an architectural invariant.

`BarStore.get_bars` takes `cutoff` as a keyword-only parameter with no default: omitting it is a
`TypeError`, so look-ahead cannot be written by accident (brief section 2, invariant 1).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Protocol

import pandas as pd

from greenclose.domain.models import Fill, Intent


class BarStore(Protocol):
    @property
    def snapshot_id(self) -> str: ...

    def get_bars(
        self,
        symbols: Sequence[str],
        *,
        cutoff: pd.Timestamp,
        lookback_days: int | None = None,
    ) -> pd.DataFrame:
        """Bars with date <= cutoff. `cutoff` is keyword-only and REQUIRED."""
        ...

    def get_index(self, *, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Benchmark (and optional VIX) rows with date <= cutoff."""
        ...


class Broker(Protocol):
    """A fill simulator. There is no other implementation and there never will be."""

    def execute(
        self,
        intents: Sequence[Intent],
        *,
        bars_today: pd.DataFrame,
        asof: date,
        cost_multiplier: float = 1.0,
    ) -> list[Fill]: ...


class Clock(Protocol):
    def now(self) -> datetime: ...

    def today(self) -> date: ...
