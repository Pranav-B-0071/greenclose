"""The multiple-testing counter (brief invariant 9).

One JSON line per configuration evaluated out of sample. The row count is `N` in the Deflated Sharpe
Ratio. If a configuration was evaluated and is not here, no result that depends on it is valid.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from greenclose.ops.provenance import utc_now_iso


@dataclass(slots=True)
class Trial:
    description: str
    family: str  # baseline | ladder | grid | sweep | h2 | leakage | holdout | forward
    config_hash: str
    snapshot_id: str
    git_sha: str
    seed: int
    oos_ic: float | None = None
    oos_sharpe: float | None = None
    oos_net_return_pa: float | None = None
    n_days: int | None = None
    touched_holdout: bool = False
    counts_for_dsr: bool = True
    extra: dict[str, Any] = field(default_factory=dict)
    logged_at: str = field(default_factory=utc_now_iso)


# A configuration is identified by what was tried, not by how many times it was run. Re-evaluating
# the same rung, seed and snapshot -- after a code change, or because a run was interrupted -- is not
# a new trial, and counting it twice would deflate the Sharpe ratio for work nobody did. The file
# stays append-only so the audit trail is complete; only the counting is de-duplicated.
IDENTITY_FIELDS = ("description", "family", "config_hash", "snapshot_id", "seed", "touched_holdout")


def distinct(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The most recent row for each distinct configuration, in first-seen order."""
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        out[tuple(str(row.get(k)) for k in IDENTITY_FIELDS)] = row
    return list(out.values())


class ResearchLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trial: Trial) -> int:
        record = asdict(trial)
        record["trial_id"] = self.count() + 1
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return int(record["trial_id"])

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def count(self, *, dsr_only: bool = False) -> int:
        """Rows written, or -- for DSR -- distinct configurations tried."""
        rows = self.rows()
        if dsr_only:
            return sum(1 for r in distinct(rows) if r.get("counts_for_dsr", True))
        return len(rows)

    def sharpes(self) -> list[float]:
        return [
            float(r["oos_sharpe"])
            for r in distinct(self.rows())
            if r.get("oos_sharpe") is not None and r.get("counts_for_dsr", True)
        ]
