"""Small helpers for locating cached artefacts (features, labels, predictions) by snapshot id."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from greenclose.config import Paths
from greenclose.features.registry import REGISTRY
from greenclose.ingest.store import ParquetBarStore


def open_store(
    paths: Paths, snapshot_id: str | None = None, *, record_access: bool = False
) -> ParquetBarStore:
    return ParquetBarStore.open(paths.data_dir / "snapshots", snapshot_id, record_access=record_access)


def features_path(paths: Paths, snapshot_id: str) -> Path:
    import greenclose.features.build  # noqa: F401  (registers features so the version hash is complete)

    return paths.data_dir / "features" / f"{snapshot_id}__{REGISTRY.version_hash()}.parquet"


def load_features(paths: Paths, snapshot_id: str) -> pd.DataFrame:
    p = features_path(paths, snapshot_id)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `greenclose features` first.")
    return pd.read_parquet(p)


def labels_path(paths: Paths, snapshot_id: str, horizon_days: int, base: str) -> Path:
    return paths.data_dir / "labels" / f"{snapshot_id}__h{horizon_days}_{base}.parquet"


def load_labels(paths: Paths, snapshot_id: str, horizon_days: int, base: str) -> pd.DataFrame:
    p = labels_path(paths, snapshot_id, horizon_days, base)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `greenclose labels` first.")
    return pd.read_parquet(p)
