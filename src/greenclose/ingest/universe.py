"""Symbol helpers: universe symbol <-> data-source ticker <-> file-safe name."""

from __future__ import annotations


def safe_name(symbol: str) -> str:
    """File-system-safe name for a symbol: M&M -> M_AND_M, ^NSEI -> IDX_NSEI, BAJAJ-AUTO -> BAJAJ_AUTO."""
    return symbol.replace("&", "_AND_").replace("^", "IDX_").replace("-", "_")


def safe_name_map(symbols: list[str]) -> dict[str, str]:
    """safe name -> symbol, so a directory of CSVs can be mapped back to the universe."""
    return {safe_name(s): s for s in symbols}
