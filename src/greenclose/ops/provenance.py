"""Traceability: git SHA, config hash, snapshot id and seed on every result (brief invariant 6)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def _git(args: list[str], root: Path | None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def git_sha(root: Path | None = None) -> str:
    out = _git(["rev-parse", "--short=10", "HEAD"], root)
    return out.strip() if out and out.strip() else "no-git"


def git_dirty(root: Path | None = None) -> bool:
    out = _git(["status", "--porcelain", "--untracked-files=no"], root)
    return bool(out and out.strip())


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class Provenance:
    run_id: str
    git_sha: str
    git_dirty: bool
    config_hash: str
    snapshot_id: str
    seed: int
    created_at: str

    @classmethod
    def create(
        cls, *, config_hash: str, snapshot_id: str, seed: int, root: Path | None = None, label: str = ""
    ) -> Provenance:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        tag = f"-{label}" if label else ""
        return cls(
            run_id=f"{ts}{tag}-{config_hash[:8]}-s{seed}",
            git_sha=git_sha(root),
            git_dirty=git_dirty(root),
            config_hash=config_hash,
            snapshot_id=snapshot_id,
            seed=seed,
            created_at=utc_now_iso(),
        )

    def as_dict(self) -> dict[str, str | int | bool]:
        return {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "config_hash": self.config_hash,
            "snapshot_id": self.snapshot_id,
            "seed": self.seed,
            "created_at": self.created_at,
        }
