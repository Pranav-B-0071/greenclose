"""Shared fixtures. Everything here is offline and deterministic."""

from __future__ import annotations

from pathlib import Path

import pytest

from greenclose.config import AppConfig, Paths, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    return load_config(Paths(root=ROOT))
