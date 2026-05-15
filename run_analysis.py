"""
Master analysis script. Runs the entire pipeline and saves figures + findings.
This is the source of truth used to populate the notebook and the README.

Usage:
    python run_analysis.py [path/to/keeta_scorecard.xlsx]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_keeta_scorecard
from src.attendance_parser import add_attendance_features
from src.feature_engineering import derive_day_level_features, aggregate_to_courier_level
from src.anomaly_detection import build_rule_based_labels, fit_anomaly_model
from src.segmentation import run_segmentation, name_archetypes
from src.supervised_models import (
    fit_productivity_models, fit_underperformer_models,
    REGRESSION_FEATURES, CLASSIFIER_FEATURES, build_underperformer_label,
)


FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH_DEFAULT = "data/keeta_scorecard.xlsx"


def main(data_path: str = DATA_PATH_DEFAULT):
    np.random.seed(42)
    findings = {}

    # --- 1. Load + clean ---
    df = load_keeta_scorecard(data_path)
    df = add_attendance_features(df)
    df = derive_day_level_features(df)
    active = df[df.on_shift_bool].copy()
    courier = aggregate_to_courier_level(df)

    findings.update({
        "n_couriers_total": int(df.courier_id.nunique()),
        "n_couriers_active": int(df[df.on_shift_bool].courier_id.nunique()),
        "window_start": str(df.date.min().date()),
        "window_end": str(df.date.max().date()),
        "window_days": int((df.date.max() - df.date.min()).days + 1),
        "total_courier_days": int(len(df)),
        "active_courier_days": int(active.shape[0]),
    })
    findings["n_couriers_dormant"] = findings["n_couriers_total"] - findings["n_couriers_active"]
    findings["active_share"] = findings["active_courier_days"] / findings["total_courier_days"]

    # Figure 1: roster utilization
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    counts = df[df.on_shift_bool].groupby("courier_id").size().sort_values()
    all_couriers = df.groupby("courier_id").size().reset_index(name="roster_days")
    activity = df.groupby("courier_id")["on_shift_bool"].sum()
    activity_full = activity.reindex(all_couriers.courier_id).fillna(0).sort_values()
    ax[0].barh(range(len(activity_full)), activity_full.values, color="#3a7ca5")
    ax[0].set_xlabel("Active days in 142-day window")
    ax[0].set_ylabel("Courier (sorted)")
    ax[0].set_title(f"Roster utilization: {findings['n_couriers_active']}/{findings['n_couriers_total']} couriers ever active")
    ax[0].axvline(0.1, color="grey", ls=":", lw=1)
    ax[1].pie([findings['active_courier_days'],
               findings['total_courier_days'] - findings['active_courier_days']],
              labels=["Active", "No shift"],
              colors=["#3a7ca5", "#d9d9d9"],
              autopct="%1.1f%%", startangle=90, wedgeprops={"linewidth": 1, "edgecolor": "white"})
    ax[1].set_title("Of all courier-days in roster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_roster_utilization.png", dpi=140)
    plt.close()

    # --- 2. Day-of-week + Ramadan ---
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = active.groupby("dow").agg(
        n=("delivered", "size"),
        mean_delivered=("delivered", "mean"),
        mean_online_hr=("online_min", lambda s: s.mean() / 60),
        mean_avg_time=("avg_delivery_time_min", "mean"),
    )
    dow.index = [dow_names[i] for i in dow.index]
    findings["dow_table"] = dow.round(2).to_dict()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(dow.index, dow["mean_delivered"], color="#3a7ca5")
    ax[0].set_ylabel("Mean deliveries / active day")
    ax[0].set_title("Volume by day of week")
    ax2 = ax[1]
    ax2.bar(dow.index, dow["mean_avg_time"], color="#c46e3a")
    ax2.set_ylabel("Mean delivery time (min)")
    ax2.set_title("Speed by day of week")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_day_of_week.png", dpi=140)
    plt.close()

    # Ramadan effect
    ram = active.groupby("is_ramadan").agg(
        n=("delivered", "size"),
        mean_delivered=("delivered", "mean"),
        mean_online_hr=("online_min", lambda s: s.mean() / 60),
        mean_ontime=("ontime_rate", "mean"),
        mean_avg_time=("avg_delivery_time_min", "mean"),
        mean_overdue=("overdue", "mean"),
    )
    pre, rmd = ram.loc[False], ram.loc[True]
    findings["ramadan"] = {
        "volume_drop_pct": float((1 - rmd.mean_delivered / pre.mean_delivered) * 100),
        "hours_drop_pct": float((1 - rmd.mean_online_hr / pre.mean_online_hr) * 100),
        "avg_time_drop_pct": float((1 - rmd.mean_avg_time / pre.mean_avg_time) * 100),
        "n_pre": int(pre.n), "n_during": int(rmd.n),
    }

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [
        ("mean_delivered", "Deliveries / day", "#3a7ca5"),
        ("mean_online_hr", "Online hours", "#c46e3a"),
        ("mean_avg_time", "Avg delivery time (min)", "#5a9a3a"),
    ]
    for i, (m, lbl, c) in enumerate(metrics):
        ax[i].bar(["Non-Ramadan", "Ramadan"], [pre[m], rmd[m]], color=[c, "#d4a44a"])
        ax[i].set_title(lbl)
        for j, v in enumerate([pre[m], rmd[m]]):
            ax[i].text(j, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    plt.suptitle("Ramadan 2026 vs non-Ramadan operational metrics", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_ramadan_effect.png", dpi=140, bbox_inches="tight")
    plt.close()

    # --- 3. Anomaly detection ---
    anomaly_features = [
        "online_min", "total_qualified_min", "delivered",
        "avg_delivery_time_min", "overdue_rate", "acceptance_rate",
    ]
    clean = active.dropna(subset=anomaly_features).copy()
    weak = build_rule_based_labels(clean)
    model, info = fit_anomaly_model(clean, anomaly_features, val_frac=0.3, seed=42)
    X_all = clean[anomaly_features].values
    log_p = model.log_p(X_all)
    preds = (log_p < model.epsilon).astype(int)
    clean["anomaly_log_p"] = log_p
    clean["is_anomaly"] = preds

    findings["anomaly"] = {
        "features": anomaly_features,
        "n_records": int(len(clean)),
        "weak_label_rate": float(weak.mean()),
        "val_f1": float(info["val_f1"]),
        "epsilon_log_p": float(info["epsilon"]),
        "flagged_count": int(preds.sum()),
        "flagged_share": float(preds.mean()),
    }

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(log_p, bins=50, color="#3a7ca5", alpha=0.85)
    ax.axvline(model.epsilon, color="red", ls="--", label=f"epsilon = {model.epsilon:.2f}")
    ax.set_xlabel("log p(x)")
    ax.set_ylabel("Count")
    ax.set_title(f"Gaussian anomaly score distribution ({preds.sum()} of {len(clean)} flagged)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_anomaly_distribution.png", dpi=140)
    plt.close()

    top_anom = (
        clean.nsmallest(10, "anomaly_log_p")
        [["date", "courier_id", "online_min", "delivered", "ontime_rate",
          "avg_delivery_time_min", "overdue", "severely_overdue", "anomaly_log_p"]]
    )
    findings["top_anomalies"] = top_anom.assign(date=top_anom["date"].astype(str)).to_dict(orient="records")

    # --- 4. Segmentation (force k=3 for interpretable archetypes) ---
    seg = run_segmentation(courier, force_k=3)
    archetypes = name_archetypes(seg)
    findings["segmentation"] = {
        "k": int(seg.k),
        "silhouette": float(seg.silhouette),
        "pca_var": [float(x) for x in seg.pca_explained_variance],
        "archetypes": archetypes,
        "cluster_sizes": pd.Series(seg.labels).value_counts().sort_index().to_dict(),
    }
    findings["segmentation"]["centroids"] = seg.centroids_raw.round(3).to_dict()

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = ["#3a7ca5", "#c46e3a", "#5a9a3a", "#a3539b", "#d4a44a", "#8d6e63"]
    for c in sorted(set(seg.labels)):
        mask = seg.labels == c
        ax.scatter(seg.pca_coords[mask, 0], seg.pca_coords[mask, 1],
                   color=palette[c], label=f"Cluster {c}: {archetypes[c]}",
                   s=120, alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.set_xlabel(f"PC1 ({seg.pca_explained_variance[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({seg.pca_explained_variance[1]:.1%} variance)")
    ax.set_title(f"Courier segmentation (k={seg.k}, silhouette={seg.silhouette:.2f})")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_segmentation_pca.png", dpi=140)
    plt.close()

    # Heatmap of standardized centroids
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(seg.centroids_z, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, cbar_kws={"label": "z-score"})
    ax.set_title("Cluster centroids (standardized features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_centroid_heatmap.png", dpi=140)
    plt.close()

    # --- 5. Productivity regression ---
    reg_results, splits = fit_productivity_models(active)
    findings["regression"] = [
        {
            "model": r.name,
            "mae_train": float(r.mae_train), "mae_val": float(r.mae_val), "mae_test": float(r.mae_test),
            "r2_train": float(r.r2_train), "r2_val": float(r.r2_val), "r2_test": float(r.r2_test),
        }
        for r in reg_results
    ]
    best_reg = max(reg_results, key=lambda r: r.r2_test)
    findings["regression_best"] = best_reg.name
    findings["regression_test_r2"] = float(best_reg.r2_test)
    findings["regression_test_mae"] = float(best_reg.mae_test)

    # Bias-variance bar chart
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = [r.name for r in reg_results]
    x = np.arange(len(names))
    w = 0.27
    ax.bar(x - w, [r.mae_train for r in reg_results], w, label="Train", color="#3a7ca5")
    ax.bar(x, [r.mae_val for r in reg_results], w, label="Val", color="#c46e3a")
    ax.bar(x + w, [r.mae_test for r in reg_results], w, label="Test", color="#5a9a3a")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("MAE (deliveries/day)")
    ax.set_title("Productivity regression — MAE by split (lower is better)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_regression_bias_variance.png", dpi=140)
    plt.close()

    # --- 6. Underperformer classifier ---
    clf_results, base_rate = fit_underperformer_models(active)
    findings["underperformer"] = {
        "positive_rate": float(base_rate) if base_rate else None,
        "models": [
            {
                "model": r.name,
                "auc_train": float(r.auc_train), "auc_val": float(r.auc_val), "auc_test": float(r.auc_test),
                "f1_test": float(r.f1_test),
            }
            for r in clf_results
        ],
    }
    if clf_results:
        best_clf = max(clf_results, key=lambda r: r.auc_test)
        findings["classifier_best"] = best_clf.name
        findings["classifier_test_auc"] = float(best_clf.auc_test)
        findings["classifier_test_f1"] = float(best_clf.f1_test)
        findings["classifier_top_features"] = sorted(
            best_clf.feature_importances.items(),
            key=lambda kv: abs(kv[1]), reverse=True,
        )[:8]

        # Feature importance figure
        feats = sorted(best_clf.feature_importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        names = [k for k, _ in feats][::-1]
        vals = [v for _, v in feats][::-1]
        colors = ["#3a7ca5" if v > 0 else "#c46e3a" for v in vals]
        ax.barh(names, vals, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Coefficient / importance")
        ax.set_title(f"Underperformer classifier — top features ({best_clf.name})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "08_classifier_features.png", dpi=140)
        plt.close()

    # --- save findings ---
    with open("outputs/findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(json.dumps(findings, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_PATH_DEFAULT)
