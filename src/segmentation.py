"""
Courier-level behavioral segmentation.

Pipeline:
  1. Standardize courier-level features (z-score)
  2. K-Means with k chosen by silhouette score on k in {2..6}
  3. PCA (2D) for visualization
  4. Centroid descriptions to label archetypes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SEGMENTATION_FEATURES = [
    "active_days",
    "activity_rate",
    "mean_delivered_per_day",
    "mean_online_min",
    "mean_qualified_min",
    "mean_peak_min",
    "mean_ontime",
    "mean_avg_delivery_time",
    "mean_acceptance",
    "mean_lunch_peak",
    "mean_dinner_peak",
    "mean_fragmentation",
    "overdue_rate",
    "large_order_share",
]


@dataclass
class SegmentationResult:
    labels: np.ndarray
    k: int
    silhouette: float
    centroids_z: pd.DataFrame
    centroids_raw: pd.DataFrame
    pca_coords: np.ndarray
    pca_explained_variance: list


def run_segmentation(
    courier: pd.DataFrame,
    feature_cols: Sequence[str] = SEGMENTATION_FEATURES,
    k_range: range = range(2, 7),
    seed: int = 42,
    force_k: int | None = None,
) -> SegmentationResult:
    X = courier[list(feature_cols)].fillna(courier[list(feature_cols)].mean()).values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    silhouette_curve = {}
    best_k, best_score, best_labels = 2, -1.0, None
    for k in k_range:
        if k >= len(Xz):
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(Xz)
        score = silhouette_score(Xz, labels)
        silhouette_curve[k] = score
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    chosen_k = force_k if force_k is not None else best_k
    km_final = KMeans(n_clusters=chosen_k, random_state=seed, n_init=10)
    final_labels = km_final.fit_predict(Xz)
    final_score = silhouette_score(Xz, final_labels) if len(set(final_labels)) > 1 else 0.0

    centroids_z = pd.DataFrame(km_final.cluster_centers_, columns=feature_cols)
    centroids_z.index.name = "cluster"

    raw_means = []
    for c in range(chosen_k):
        m = courier.loc[final_labels == c, feature_cols].mean()
        m.name = c
        raw_means.append(m)
    centroids_raw = pd.DataFrame(raw_means)
    centroids_raw.index.name = "cluster"

    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xz)

    return SegmentationResult(
        labels=final_labels,
        k=chosen_k,
        silhouette=final_score,
        centroids_z=centroids_z,
        centroids_raw=centroids_raw,
        pca_coords=coords,
        pca_explained_variance=pca.explained_variance_ratio_.tolist(),
    )


def name_archetypes(result: SegmentationResult) -> dict:
    """
    Auto-name clusters based on standardized centroid signatures.
    Heuristic, based on which features dominate each centroid.
    """
    z = result.centroids_z
    names = {}
    for c in z.index:
        row = z.loc[c]
        name_bits = []
        if row["active_days"] > 0.5:
            name_bits.append("high-activity")
        elif row["active_days"] < -0.5:
            name_bits.append("low-activity")
        if row["mean_delivered_per_day"] > 0.5:
            name_bits.append("high-volume")
        elif row["mean_delivered_per_day"] < -0.5:
            name_bits.append("low-volume")
        if row["mean_avg_delivery_time"] > 0.5:
            name_bits.append("slow")
        elif row["mean_avg_delivery_time"] < -0.5:
            name_bits.append("fast")
        if row["mean_ontime"] < -0.5:
            name_bits.append("late-prone")
        if not name_bits:
            name_bits.append("baseline")
        names[c] = " · ".join(name_bits)
    return names
