"""
Gaussian anomaly detection for courier-day records.

Implements the same approach taught in DeepLearning.AI ML Specialization C3W1:
    p(x) = product of N(mu_i, sigma_i^2) over selected features
    flag day if p(x) < epsilon

Epsilon is selected by grid search to maximize F1 on a validation slice
labeled with rule-based suspicion flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class GaussianAnomalyModel:
    feature_cols: Sequence[str]
    mu: np.ndarray | None = None
    sigma2: np.ndarray | None = None
    epsilon: float | None = None

    def fit(self, X: np.ndarray) -> "GaussianAnomalyModel":
        self.mu = np.nanmean(X, axis=0)
        self.sigma2 = np.nanvar(X, axis=0)
        self.sigma2 = np.where(self.sigma2 < 1e-9, 1e-9, self.sigma2)
        return self

    def log_p(self, X: np.ndarray) -> np.ndarray:
        coef = -0.5 * np.log(2 * np.pi * self.sigma2)
        exponent = -((X - self.mu) ** 2) / (2 * self.sigma2)
        return np.nansum(coef + exponent, axis=1)

    def select_epsilon(self, X_val: np.ndarray, y_val: np.ndarray, n_steps: int = 1000):
        """Grid search epsilon (in log-p space) to maximize F1."""
        log_p = self.log_p(X_val)
        finite = log_p[np.isfinite(log_p)]
        if len(finite) == 0:
            self.epsilon = -np.inf
            return -np.inf, 0.0
        lo, hi = finite.min(), finite.max()
        thresholds = np.linspace(lo, hi, n_steps)
        best_f1, best_eps = 0.0, lo
        for eps in thresholds:
            preds = log_p < eps
            tp = int(np.sum((preds == 1) & (y_val == 1)))
            fp = int(np.sum((preds == 1) & (y_val == 0)))
            fn = int(np.sum((preds == 0) & (y_val == 1)))
            if tp == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1, best_eps = f1, eps
        self.epsilon = best_eps
        return best_eps, best_f1

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.log_p(X) < self.epsilon).astype(int)


def build_rule_based_labels(active: pd.DataFrame) -> np.ndarray:
    """
    Weak labels for epsilon tuning, derived from operational red flags
    a Future Link supervisor would already use:

        - Severely overdue task on the day                            (any)
        - On-time rate < 90% on a day with >=10 deliveries
        - Avg delivery time > 45 min on a day with >=10 deliveries
        - Acceptance rate < 80% on an otherwise active day
        - Logged 8+ online hours but delivered < 5 tasks
    """
    sev = active["severely_overdue"].fillna(0) > 0
    bad_ot = (active["ontime_rate"].fillna(1) < 0.9) & (active["delivered"].fillna(0) >= 10)
    slow = (active["avg_delivery_time_min"].fillna(0) > 45) & (active["delivered"].fillna(0) >= 10)
    low_acc = active["acceptance_rate"].fillna(1) < 0.8
    idle = (active["online_min"].fillna(0) >= 8 * 60) & (active["delivered"].fillna(0) < 5)
    return (sev | bad_ot | slow | low_acc | idle).astype(int).values


def fit_anomaly_model(
    active: pd.DataFrame,
    feature_cols: Sequence[str],
    val_frac: float = 0.3,
    seed: int = 42,
):
    """Fit Gaussian model on the 'clean' slice, tune epsilon on a held-out validation set."""
    labels = build_rule_based_labels(active)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(active))
    rng.shuffle(idx)
    val_n = int(len(idx) * val_frac)
    val_idx, fit_idx = idx[:val_n], idx[val_n:]

    fit_mask_clean = (labels[fit_idx] == 0)
    X_fit = active.iloc[fit_idx][list(feature_cols)].values[fit_mask_clean]
    X_val = active.iloc[val_idx][list(feature_cols)].values
    y_val = labels[val_idx]

    model = GaussianAnomalyModel(feature_cols=list(feature_cols))
    model.fit(X_fit)
    eps, f1 = model.select_epsilon(X_val, y_val)
    return model, {"val_f1": f1, "epsilon": eps, "n_fit": len(X_fit), "n_val": len(X_val)}
