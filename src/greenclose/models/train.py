"""Walk-forward training and prediction.

Every transformation is fitted inside the fold (brief invariant 4): target winsorisation at +/-5 sigma,
median imputation and scaling are fitted on the training rows of each retrain only. The purged k-fold
inside the training window selects hyperparameters; the fixed grid lives in config/grid.yaml.

`walk_forward_predict` is the single loop every model rung uses. A `Head` knows how to fit itself on
(X, y, groups) and how to turn its raw output into the columns the decision layer understands:
`score` (ordering), `expected_return_bps` (magnitude, when the head has one) and probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from greenclose.config import AppConfig
from greenclose.models.splitters import PurgedGroupTimeSeriesSplit, training_mask, walk_forward_schedule
from greenclose.ops.logging import get_logger

log = get_logger(__name__)


class Head(Protocol):
    name: str
    target: str

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int) -> Any: ...

    def predict(self, model: Any, X: np.ndarray) -> pd.DataFrame: ...


@dataclass
class Preprocessor:
    """In-fold imputer + scaler (+ target winsorisation). Fitted on training rows only."""

    winsor_sigma: float = 5.0
    imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median"))
    scaler: StandardScaler = field(default_factory=StandardScaler)
    y_lo: float = -np.inf
    y_hi: float = np.inf

    def fit(self, X: np.ndarray, y: np.ndarray, *, continuous_target: bool) -> Preprocessor:
        self.imputer.fit(X)
        self.scaler.fit(self.imputer.transform(X))
        if continuous_target and self.winsor_sigma > 0:
            mu, sd = float(np.nanmean(y)), float(np.nanstd(y))
            self.y_lo, self.y_hi = mu - self.winsor_sigma * sd, mu + self.winsor_sigma * sd
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.scaler.transform(self.imputer.transform(X)), dtype="float64")

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return np.clip(y, self.y_lo, self.y_hi)


@dataclass
class RidgeHead:
    alpha: float = 1.0
    name: str = "ridge"
    target: str = "y_demeaned"
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int) -> Any:
        return Ridge(alpha=self.alpha).fit(X, y)

    def predict(self, model: Any, X: np.ndarray) -> pd.DataFrame:
        p = np.asarray(model.predict(X), dtype="float64")
        return pd.DataFrame({"score": p, "expected_return_bps": p * 1e4})

    def candidates(self) -> list[RidgeHead]:
        return [
            RidgeHead(alpha=a, name=self.name, target=self.target, alphas=self.alphas) for a in self.alphas
        ]


@dataclass
class LogisticHead:
    c: float = 1.0
    name: str = "logistic_beat"
    target: str = "y_beat"
    prob_column: str = "prob_beat"

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int) -> Any:
        return LogisticRegression(C=self.c, max_iter=500).fit(X, y.astype(int))

    def predict(self, model: Any, X: np.ndarray) -> pd.DataFrame:
        p = np.asarray(model.predict_proba(X)[:, 1], dtype="float64")
        return pd.DataFrame({"score": p, self.prob_column: p})

    def candidates(self) -> list[LogisticHead]:
        return [self]


@dataclass
class ModelRun:
    head_name: str
    target: str
    predictions: pd.DataFrame  # (date, symbol) -> score, expected_return_bps?, prob_*?
    retrains: list[dict[str, Any]]
    n_grid_evaluations: int


def _cv_score(
    head: Any,
    pre_cfg: float,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splitter: PurgedGroupTimeSeriesSplit,
    seed: int,
) -> float:
    """Mean daily rank IC on the purged validation folds inside the training window."""
    scores: list[float] = []
    for tr, va in splitter.split(groups):
        pre = Preprocessor(winsor_sigma=pre_cfg).fit(
            X[tr], y[tr], continuous_target=head.target not in ("y_beat", "y_green")
        )
        model = head.fit(pre.transform_X(X[tr]), pre.transform_y(y[tr]), groups[tr], seed=seed)
        pred = head.predict(model, pre.transform_X(X[va]))["score"].to_numpy()
        df = pd.DataFrame({"g": groups[va], "p": pred, "y": y[va]})
        ic = df.groupby("g").apply(
            lambda d: d["p"].rank().corr(d["y"].rank()) if len(d) > 2 else np.nan, include_groups=False
        )
        scores.append(float(np.nanmean(ic.to_numpy())))
    return float(np.nanmean(scores))


def walk_forward_predict(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    model_columns: list[str],
    head: Any,
    cfg: AppConfig,
    oos_dates: pd.DatetimeIndex,
    seed: int = 0,
    select_hparams: bool = True,
    extra_feature_frame: pd.DataFrame | None = None,
) -> ModelRun:
    """Expanding (or rolling) walk-forward: retrain monthly, predict daily, everything fitted in-fold."""
    wf = cfg.base.walk_forward
    cols = list(model_columns)
    X_frame = features[cols]
    if extra_feature_frame is not None:
        X_frame = X_frame.join(extra_feature_frame, how="left")
        cols = list(X_frame.columns)
    X_all = X_frame.to_numpy(dtype="float64")
    y_all = labels[head.target].reindex(features.index).to_numpy(dtype="float64")
    dates_all = features.index.get_level_values("date")
    all_dates = pd.DatetimeIndex(dates_all.unique()).sort_values()
    date_pos = {d: i for i, d in enumerate(all_dates)}
    t_idx = np.array([date_pos[d] for d in dates_all], dtype="int64")
    label_avail = labels["label_available_idx"].reindex(features.index).to_numpy(dtype="float64")
    # label_available_idx is on the store calendar; align it to the feature calendar via the offset
    label_avail_idx = (
        t_idx + (label_avail - labels["t_idx"].reindex(features.index).to_numpy(dtype="float64"))
    ).astype("int64")
    label_avail_idx = np.where(np.isnan(label_avail), 10**9, label_avail_idx)

    schedule = walk_forward_schedule(
        all_dates,
        oos_dates=oos_dates,
        min_train_years=wf.min_train_years,
        rolling_years=wf.rolling_years,
        hparam_reselect_months=wf.hparam_reselect_months,
    )
    splitter = PurgedGroupTimeSeriesSplit(
        wf.cv_folds,
        purge_days=wf.purge_days,
        embargo_days=wf.embargo_days,
        horizon_days=cfg.base.labels.horizon_days,
    )
    continuous = head.target not in ("y_beat", "y_green")
    current = head
    preds: list[pd.DataFrame] = []
    retrains: list[dict[str, Any]] = []
    n_grid = 0
    for point in schedule:
        mask = training_mask(
            t_idx,
            label_avail_idx,
            retrain_idx=point.retrain_idx,
            purge_days=wf.purge_days,
            rolling_start_idx=point.rolling_start_idx,
        )
        mask &= ~np.isnan(y_all)
        mask &= ~np.isnan(X_all).all(axis=1)
        X_tr, y_tr, g_tr = X_all[mask], y_all[mask], t_idx[mask]
        if len(np.unique(g_tr)) < 60:
            continue
        if select_hparams and point.reselect_hparams and hasattr(head, "candidates"):
            best, best_score = current, -np.inf
            for cand in head.candidates():
                s = _cv_score(cand, cfg.base.labels.winsor_sigma, X_tr, y_tr, g_tr, splitter, seed)
                n_grid += 1
                if s > best_score:
                    best, best_score = cand, s
            current = best
        pre = Preprocessor(winsor_sigma=cfg.base.labels.winsor_sigma).fit(
            X_tr, y_tr, continuous_target=continuous
        )
        model = current.fit(pre.transform_X(X_tr), pre.transform_y(y_tr), g_tr, seed=seed)
        pmask = dates_all.isin(point.predict_dates)
        out = current.predict(model, pre.transform_X(X_all[pmask]))
        out.index = features.index[pmask]
        preds.append(out)
        retrains.append(
            {
                "retrain_date": point.retrain_date,
                "n_train_rows": int(mask.sum()),
                "train_last_idx": int(t_idx[mask].max()),
                "retrain_idx": point.retrain_idx,
                "params": {k: v for k, v in current.__dict__.items() if isinstance(v, (int, float, str))},
                "pre_x_mean_first": float(pre.scaler.mean_[0]) if pre.scaler.mean_ is not None else np.nan,
            }
        )
    predictions = pd.concat(preds).sort_index() if preds else pd.DataFrame(columns=["score"])
    return ModelRun(
        head_name=head.name,
        target=head.target,
        predictions=predictions,
        retrains=retrains,
        n_grid_evaluations=n_grid,
    )
