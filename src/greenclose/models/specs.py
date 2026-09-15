"""Model heads for the ladder (brief section 10).

A head owns three things: how to fit on (X, y, groups) inside a fold, how to turn its raw output into
the columns the decision layer consumes, and which hyperparameter cells it is allowed to try. The
grid is fixed in config/grid.yaml and may not be expanded: every cell counts toward the
multiple-testing penalty and is logged.

Every head that needs an early-stopping set carves it out of the *training* rows by date, with the
same purge as the outer walk-forward, so the stopping decision never sees a future day.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

EARLY_STOP_FRACTION = 0.2  # last fifth of the training window, by date
EARLY_STOP_ROUNDS = 50


def _split_by_date(
    groups: np.ndarray, *, purge: int, fraction: float = EARLY_STOP_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into (fit, holdout) by date: the holdout is the last `fraction` of dates, purged."""
    uniq = np.unique(groups)
    if len(uniq) < 20:
        idx = np.arange(len(groups))
        return idx, idx[:0]
    cut = uniq[int(len(uniq) * (1.0 - fraction))]
    fit = np.flatnonzero(groups < cut - purge)
    hold = np.flatnonzero(groups >= cut)
    if len(fit) == 0 or len(hold) == 0:
        idx = np.arange(len(groups))
        return idx, idx[:0]
    return fit, hold


def _group_counts(groups: np.ndarray) -> np.ndarray:
    """Query-group sizes for a ranking objective. Rows must already be contiguous per date."""
    _, counts = np.unique(groups, return_counts=True)
    return counts


# ----------------------------------------------------------------------------- linear rungs


@dataclass
class RidgeHead:
    alpha: float = 1.0
    name: str = "ridge"
    target: str = "y_demeaned"
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    rung: int = 3

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int) -> Any:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=self.alpha).fit(X, y)

    def predict(self, model: Any, X: np.ndarray) -> pd.DataFrame:
        p = np.asarray(model.predict(X), dtype="float64")
        return pd.DataFrame({"score": p, "expected_return_bps": p * 1e4})

    def candidates(self) -> list[RidgeHead]:
        return [replace(self, alpha=a) for a in self.alphas]


@dataclass
class LogisticHead:
    c: float = 1.0
    name: str = "logistic_beat"
    target: str = "y_beat"
    prob_column: str = "prob_beat"
    rung: int = 3

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int) -> Any:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=self.c, max_iter=1000).fit(X, y.astype(int))

    def predict(self, model: Any, X: np.ndarray) -> pd.DataFrame:
        p = np.asarray(model.predict_proba(X)[:, 1], dtype="float64")
        return pd.DataFrame({"score": p, self.prob_column: p})

    def candidates(self) -> list[LogisticHead]:
        return [self]


# ----------------------------------------------------------------------------- gradient boosting


@dataclass
class LGBMRegressorHead:
    """Rung 4: LightGBM regression on y_demeaned with a Huber objective (heavy-tailed target)."""

    params: dict[str, Any] = field(default_factory=dict)
    name: str = "lgbm_huber"
    target: str = "y_demeaned"
    purge: int = 1
    grid: tuple[dict[str, Any], ...] = ()
    rung: int = 4

    def _full_params(self, seed: int) -> dict[str, Any]:
        p = {
            "objective": "huber",
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "n_estimators": 600,
            "verbosity": -1,
            "seed": seed,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 0,
        }
        p.update(self.params)
        return p

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int) -> Any:
        import lightgbm as lgb

        p = self._full_params(seed)
        n_estimators = int(p.pop("n_estimators"))
        fit_rows, stop_rows = _split_by_date(groups, purge=self.purge)
        train_set = lgb.Dataset(X[fit_rows], label=y[fit_rows], free_raw_data=False)
        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = []
        if len(stop_rows):
            valid_sets = [
                lgb.Dataset(X[stop_rows], label=y[stop_rows], reference=train_set, free_raw_data=False)
            ]
            callbacks.append(lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False))
        return lgb.train(
            p, train_set, num_boost_round=n_estimators, valid_sets=valid_sets, callbacks=callbacks
        )

    def predict(self, model: Any, X: np.ndarray) -> pd.DataFrame:
        best = getattr(model, "best_iteration", None)
        p = np.asarray(model.predict(X, num_iteration=best), dtype="float64")
        return pd.DataFrame({"score": p, "expected_return_bps": p * 1e4})

    def candidates(self) -> list[LGBMRegressorHead]:
        return [replace(self, params=dict(cell)) for cell in self.grid] or [self]


@dataclass
class LGBMRankerHead:
    """Rung 5: LightGBM lambdarank, one query group per date, integer within-day rank as the label.

    `label_gain` is linear, not the default 2^k - 1: with ~48 stocks a doubling gain overflows and
    makes the loss depend almost entirely on the single best name.
    """

    params: dict[str, Any] = field(default_factory=dict)
    name: str = "lgbm_lambdarank"
    target: str = "y_rank"
    purge: int = 1
    max_rank: int = 64
    grid: tuple[dict[str, Any], ...] = ()
    rung: int = 5

    def _full_params(self, seed: int) -> dict[str, Any]:
        p = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5],
            "label_gain": list(range(self.max_rank + 1)),
            "lambdarank_truncation_level": self.max_rank,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "n_estimators": 600,
            "verbosity": -1,
            "seed": seed,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 0,
        }
        p.update(self.params)
        return p

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int) -> Any:
        import lightgbm as lgb

        p = self._full_params(seed)
        n_estimators = int(p.pop("n_estimators"))
        label = np.clip(np.nan_to_num(y, nan=0.0).astype(int), 0, self.max_rank)
        fit_rows, stop_rows = _split_by_date(groups, purge=self.purge)
        train_set = lgb.Dataset(
            X[fit_rows], label=label[fit_rows], group=_group_counts(groups[fit_rows]), free_raw_data=False
        )
        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = []
        if len(stop_rows):
            valid_sets = [
                lgb.Dataset(
                    X[stop_rows],
                    label=label[stop_rows],
                    group=_group_counts(groups[stop_rows]),
                    reference=train_set,
                    free_raw_data=False,
                )
            ]
            callbacks.append(lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False))
        return lgb.train(
            p, train_set, num_boost_round=n_estimators, valid_sets=valid_sets, callbacks=callbacks
        )

    def predict(self, model: Any, X: np.ndarray) -> pd.DataFrame:
        best = getattr(model, "best_iteration", None)
        p = np.asarray(model.predict(X, num_iteration=best), dtype="float64")
        # a ranker emits an ordering score with no return units, so it carries no expected_return_bps
        return pd.DataFrame({"score": p})

    def candidates(self) -> list[LGBMRankerHead]:
        return [replace(self, params=dict(cell)) for cell in self.grid] or [self]


@dataclass
class CatBoostHead:
    """Rung 6a: CatBoost regression, for ensemble diversity rather than for its own IC."""

    params: dict[str, Any] = field(default_factory=dict)
    name: str = "catboost"
    target: str = "y_demeaned"
    purge: int = 1
    rung: int = 6

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int) -> Any:
        from catboost import CatBoostRegressor, Pool

        p = {
            "loss_function": "Huber:delta=0.02",
            "iterations": 600,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
        }
        p.update(self.params)
        fit_rows, stop_rows = _split_by_date(groups, purge=self.purge)
        model = CatBoostRegressor(**p)
        eval_set = Pool(X[stop_rows], y[stop_rows]) if len(stop_rows) else None
        model.fit(
            Pool(X[fit_rows], y[fit_rows]),
            eval_set=eval_set,
            early_stopping_rounds=EARLY_STOP_ROUNDS if eval_set else None,
        )
        return model

    def predict(self, model: Any, X: np.ndarray) -> pd.DataFrame:
        p = np.asarray(model.predict(X), dtype="float64")
        return pd.DataFrame({"score": p, "expected_return_bps": p * 1e4})

    def candidates(self) -> list[CatBoostHead]:
        return [self]


@dataclass
class XGBoostHead:
    """Rung 6b: XGBoost regression with a pseudo-Huber objective."""

    params: dict[str, Any] = field(default_factory=dict)
    name: str = "xgboost"
    target: str = "y_demeaned"
    purge: int = 1
    rung: int = 6

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int) -> Any:
        import xgboost as xgb

        p = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.02,
            "max_depth": 5,
            "eta": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "seed": seed,
            "nthread": 0,
            "verbosity": 0,
        }
        p.update(self.params)
        fit_rows, stop_rows = _split_by_date(groups, purge=self.purge)
        dtrain = xgb.DMatrix(X[fit_rows], label=y[fit_rows])
        evals = []
        if len(stop_rows):
            evals = [(xgb.DMatrix(X[stop_rows], label=y[stop_rows]), "valid")]
        return xgb.train(
            p,
            dtrain,
            num_boost_round=600,
            evals=evals,
            early_stopping_rounds=EARLY_STOP_ROUNDS if evals else None,
            verbose_eval=False,
        )

    def predict(self, model: Any, X: np.ndarray) -> pd.DataFrame:
        import xgboost as xgb

        best = getattr(model, "best_iteration", None)
        rng = (0, best + 1) if best is not None else None
        p = np.asarray(model.predict(xgb.DMatrix(X), iteration_range=rng), dtype="float64")
        return pd.DataFrame({"score": p, "expected_return_bps": p * 1e4})

    def candidates(self) -> list[XGBoostHead]:
        return [self]


def lightgbm_grid(cells: list[dict[str, float]]) -> tuple[dict[str, Any], ...]:
    """Translate config/grid.yaml cells into LightGBM parameter dicts, typed as LightGBM needs them."""
    out: list[dict[str, Any]] = []
    for cell in cells:
        d: dict[str, Any] = {}
        for k, v in cell.items():
            d[k] = int(v) if k in ("num_leaves", "min_data_in_leaf", "n_estimators") else float(v)
        out.append(d)
    return tuple(out)
