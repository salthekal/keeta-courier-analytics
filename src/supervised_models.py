"""
Supervised models on courier-day records.

Two tasks:
  1. Productivity regression  -> predict `delivered` from upstream features
                                 (linear, polynomial+L2, XGBoost, small NN)
  2. Underperformer classification -> binary, rule-derived label
                                 (logistic regression, XGBoost)

Models are trained with Train / Validation / Test discipline (60/20/20)
to enable bias-variance diagnostics from C2W3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# XGBoost is the recommended boosted-tree backend (per C2W4) but adds an
# external dependency. We fall back to sklearn's GradientBoosting* which is
# functionally similar and ships with sklearn. If you have XGBoost installed,
# the import below will pick it up automatically.
try:
    from xgboost import XGBClassifier, XGBRegressor  # noqa: F401
    HAS_XGB = True
except Exception:
    HAS_XGB = False


REGRESSION_FEATURES = [
    "online_min",
    "valid_online_min",
    "peak_online_min",
    "total_qualified_min",
    "total_on_shift_min",
    "n_qualified_buckets",
    "attended_lunch_peak",
    "attended_dinner_peak",
    "shift_fragmentation",
    "dow",
    "is_weekend_ksa",
    "is_ramadan",
]


@dataclass
class RegressionResult:
    name: str
    mae_train: float
    mae_val: float
    mae_test: float
    r2_train: float
    r2_val: float
    r2_test: float


def split_active_data(active: pd.DataFrame, target: str, features: Sequence[str], seed=42):
    df = active.dropna(subset=[target] + list(features)).copy()
    X = df[list(features)].values.astype(float)
    y = df[target].values.astype(float)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=seed)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _eval_regressor(name, model, splits) -> RegressionResult:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
    model.fit(X_train, y_train)
    pred_train, pred_val, pred_test = model.predict(X_train), model.predict(X_val), model.predict(X_test)
    return RegressionResult(
        name=name,
        mae_train=mean_absolute_error(y_train, pred_train),
        mae_val=mean_absolute_error(y_val, pred_val),
        mae_test=mean_absolute_error(y_test, pred_test),
        r2_train=r2_score(y_train, pred_train),
        r2_val=r2_score(y_val, pred_val),
        r2_test=r2_score(y_test, pred_test),
    )


def fit_productivity_models(active: pd.DataFrame, features=REGRESSION_FEATURES, target="delivered"):
    splits = split_active_data(active, target=target, features=features)
    results = []

    results.append(_eval_regressor("linear", Pipeline([
        ("scaler", StandardScaler()), ("model", LinearRegression())
    ]), splits))

    results.append(_eval_regressor("poly2_ridge", Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", Ridge(alpha=10.0)),
    ]), splits))

    if HAS_XGB:
        results.append(_eval_regressor(
            "xgboost",
            XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=0),
            splits,
        ))
    else:
        results.append(_eval_regressor(
            "gradient_boosting",
            GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.9, random_state=42,
            ),
            splits,
        ))

    return results, splits


def build_underperformer_label(active: pd.DataFrame) -> np.ndarray:
    """
    Operational underperformer label:
      label = 1 if (severely_overdue > 0)
               or (avg_delivery_time_min > 35 and delivered >= 10)
               or (overdue >= 2)
               or (ontime_rate < 0.95 and delivered >= 10)
               or (delivered < 5 and online_min >= 8*60)
    """
    sev = active["severely_overdue"].fillna(0) > 0
    slow = (active["avg_delivery_time_min"].fillna(0) > 35) & (active["delivered"].fillna(0) >= 10)
    over = active["overdue"].fillna(0) >= 2
    bad_ot = (active["ontime_rate"].fillna(1) < 0.95) & (active["delivered"].fillna(0) >= 10)
    idle = (active["online_min"].fillna(0) >= 8 * 60) & (active["delivered"].fillna(0) < 5)
    return (sev | slow | over | bad_ot | idle).astype(int).values


CLASSIFIER_FEATURES = [
    "online_min",
    "valid_online_min",
    "peak_online_min",
    "total_qualified_min",
    "n_qualified_buckets",
    "attended_lunch_peak",
    "attended_dinner_peak",
    "shift_fragmentation",
    "dow",
    "is_weekend_ksa",
    "is_ramadan",
    "accepted",
    "acceptance_rate",
    "arrival_rate",
]


@dataclass
class ClassificationResult:
    name: str
    auc_train: float
    auc_val: float
    auc_test: float
    f1_test: float
    feature_importances: dict | None = None


def fit_underperformer_models(active: pd.DataFrame, features=CLASSIFIER_FEATURES):
    df = active.copy()
    df["label"] = build_underperformer_label(df)
    df = df.dropna(subset=features + ["label"])
    X = df[features].values.astype(float)
    y = df["label"].values.astype(int)

    if y.sum() < 5 or y.sum() == len(y):
        return [], df["label"].mean()

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if y_temp.sum() < 4:
        return [], y.mean()
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    out = []

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    logreg.fit(X_train, y_train)
    out.append(ClassificationResult(
        name="logistic_regression",
        auc_train=roc_auc_score(y_train, logreg.predict_proba(X_train)[:, 1]),
        auc_val=roc_auc_score(y_val, logreg.predict_proba(X_val)[:, 1]),
        auc_test=roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1]),
        f1_test=f1_score(y_test, logreg.predict(X_test)),
        feature_importances=dict(zip(features, logreg.named_steps["model"].coef_[0])),
    ))

    if HAS_XGB:
        scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        boost = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=scale_pos,
            random_state=42, verbosity=0, eval_metric="auc",
        )
        boost_name = "xgboost"
    else:
        boost = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, random_state=42,
        )
        boost_name = "gradient_boosting"

    boost.fit(X_train, y_train)
    out.append(ClassificationResult(
        name=boost_name,
        auc_train=roc_auc_score(y_train, boost.predict_proba(X_train)[:, 1]),
        auc_val=roc_auc_score(y_val, boost.predict_proba(X_val)[:, 1]),
        auc_test=roc_auc_score(y_test, boost.predict_proba(X_test)[:, 1]),
        f1_test=f1_score(y_test, boost.predict(X_test)),
        feature_importances=dict(zip(features, boost.feature_importances_.tolist())),
    ))

    return out, y.mean()
