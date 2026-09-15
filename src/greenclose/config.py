"""Typed configuration.

Every YAML key is validated against a model with `extra="forbid"`: a typo is an error, never a silent
behaviour change (brief section 3). `AppConfig.hash()` is the config hash cited by every result.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- base.yaml


class DataCfg(Strict):
    start: str
    source: Literal["yfinance", "csv"] = "yfinance"
    suffix: str = ".NS"
    benchmark: str = "^NSEI"
    vix: str | None = "^INDIAVIX"
    min_symbols_per_day: int = 40
    vix_min_coverage: float = 0.99


class HoldoutCfg(Strict):
    start: str


class FeaturesCfg(Strict):
    winsor_pct: tuple[float, float] = (0.01, 0.99)
    n_clusters: int = 8
    corr_window: int = 60


class LabelsCfg(Strict):
    horizon_days: int = 1
    horizon: Literal["c2c", "c2o", "o2c"] = "c2c"
    primary: str = "y_demeaned"
    winsor_sigma: float = 5.0


class WalkForwardCfg(Strict):
    min_train_years: int = 3
    retrain: Literal["monthly"] = "monthly"
    rolling_years: int | None = None
    purge_days: int = 1
    embargo_days: int = 5
    cv_folds: int = 3
    hparam_reselect_months: int = 12


class SimulationCfg(Strict):
    capital: float
    position_size: float
    product: Literal["CNC", "MIS"] = "CNC"
    cost_multiplier: float = 1.0
    slippage_override_bps: float | None = None


class AbstainCfg(Strict):
    enabled: bool = True
    buffer_bps: float = 5.0


class StrategyCfg(Strict):
    k: int = 3
    select_from: Literal["top", "bottom"] = "top"
    sizing: Literal["inverse_vol", "equal"] = "inverse_vol"
    min_position: float = 50_000
    max_position_pct: float = 0.40
    liquidity_cap_bps_adv: float = 10.0
    sector_max: int = 1
    cluster_max: int = 1
    rebalance_mode: Literal["overlap", "fixed"] = "overlap"
    rebalance_frequency: Literal["daily", "monthly"] = "daily"
    hold_days: int = 1
    abstain: AbstainCfg = AbstainCfg()


class ReportCfg(Strict):
    ic_t_bar: float = 3.0
    costs_gross_max: float = 0.50
    dsr_bar: float = 0.95


class BaseConfig(Strict):
    project: str
    mode: Literal["paper_only"]
    data: DataCfg
    holdout: HoldoutCfg
    features: FeaturesCfg
    labels: LabelsCfg
    walk_forward: WalkForwardCfg
    seeds: list[int]
    simulation: SimulationCfg
    strategy: StrategyCfg
    report: ReportCfg


# --------------------------------------------------------------------------- charges.yaml


class BrokeragePerOrder(Strict):
    pct: float
    cap: float


class CncCharges(Strict):
    brokerage_pct: float
    stt_buy_pct: float
    stt_sell_pct: float
    stamp_buy_pct: float
    exchange_txn_pct: float
    sebi_pct: float
    ipft_pct: float
    gst_pct: float
    dp_charge_flat: float


class MisCharges(Strict):
    brokerage_per_order: BrokeragePerOrder
    stt_buy_pct: float
    stt_sell_pct: float
    stamp_buy_pct: float
    exchange_txn_pct: float
    sebi_pct: float
    ipft_pct: float
    gst_pct: float
    dp_charge_flat: float


class SlippageCfg(Strict):
    base_bps: float
    impact_coef: float
    min_bps: float
    high_vol_multiplier: float


class ChargesCfg(Strict):
    as_of: str
    source: str
    delivery_cnc: CncCharges
    intraday_mis: MisCharges
    slippage: SlippageCfg


# --------------------------------------------------------------------------- universe.yaml


class UniverseCfg(Strict):
    fixed_as_of: str
    benchmark: str
    suffix: str
    universe: dict[str, list[str]]
    dropped: dict[str, str] = {}

    @property
    def sector_of(self) -> dict[str, str]:
        return {s: sec for sec, syms in self.universe.items() for s in syms if s not in self.dropped}

    @property
    def symbols(self) -> list[str]:
        return [s for syms in self.universe.values() for s in syms if s not in self.dropped]

    @property
    def all_symbols(self) -> list[str]:
        return [s for syms in self.universe.values() for s in syms]

    def ticker(self, symbol: str) -> str:
        return f"{symbol}{self.suffix}"


# --------------------------------------------------------------------------- calendar.yaml


class SessionCfg(Strict):
    continuous_open: str
    continuous_close: str
    cas_order_entry_start: str
    cas_close: str


class CalendarCfg(Strict):
    timezone: str
    session: SessionCfg
    information_cutoff: str
    close_proxy_note: str
    holidays: list[str] = []


# --------------------------------------------------------------------------- grid.yaml


class GridCfg(Strict):
    lightgbm: dict[str, list[float]]
    ridge: dict[str, list[float]]

    def lightgbm_cells(self) -> list[dict[str, float]]:
        """The 72 cells of brief section 10.1.

        `n_estimators` is not a grid axis: it is the early-stopping ceiling (the largest listed value),
        and the purged validation fold decides the effective tree count. 3*3*2*2*1*2 = 72.
        """
        keys = [k for k in self.lightgbm if k != "n_estimators"]
        ceiling = max(self.lightgbm.get("n_estimators", [600.0]))
        combos = itertools.product(*(self.lightgbm[k] for k in keys))
        return [{**dict(zip(keys, vals, strict=True)), "n_estimators": ceiling} for vals in combos]


# --------------------------------------------------------------------------- bundle


class AppConfig(Strict):
    base: BaseConfig
    charges: ChargesCfg
    universe: UniverseCfg
    calendar: CalendarCfg
    grid: GridCfg

    def override(self, updates: dict[str, Any]) -> AppConfig:
        """Return a copy with dotted-path overrides applied, e.g. {"base.strategy.k": 5}."""
        data = self.model_dump(mode="json")
        for dotted, value in updates.items():
            node: Any = data
            parts = dotted.split(".")
            for p in parts[:-1]:
                node = node[p]
            node[parts[-1]] = value
        return AppConfig.model_validate(data)


class Paths(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GREENCLOSE_", frozen=True)

    root: Path = PROJECT_ROOT

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def research_dir(self) -> Path:
        return self.root / "research"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return loaded


def load_config(paths: Paths | None = None) -> AppConfig:
    paths = paths or Paths()
    c = paths.config_dir
    return AppConfig(
        base=BaseConfig.model_validate(load_yaml(c / "base.yaml")),
        charges=ChargesCfg.model_validate(load_yaml(c / "charges.yaml")),
        universe=UniverseCfg.model_validate(load_yaml(c / "universe.yaml")),
        calendar=CalendarCfg.model_validate(load_yaml(c / "calendar.yaml")),
        grid=GridCfg.model_validate(load_yaml(c / "grid.yaml")),
    )
