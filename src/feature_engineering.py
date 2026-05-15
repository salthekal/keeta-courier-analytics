"""
Feature engineering on top of the cleaned + attendance-parsed scorecard.

Two granularities are supported:
  - day-level   (one row per courier-day, active days only)
  - courier-level (one row per courier, aggregated across the analysis window)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def derive_day_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add productivity / discipline / experience features at the courier-day level.

    Operates on rows where the courier was on shift; off-shift roster days
    are kept but most ratios will be NaN there.
    """
    df = df.copy()
    df["delivered_per_online_hr"] = df["delivered"] / (df["online_min"] / 60.0)
    df["delivered_per_qualified_hr"] = df["delivered"] / (df["total_qualified_min"] / 60.0)

    rejected = df["rejected_total"].fillna(0)
    accepted = df["accepted"].fillna(0)
    submitted = accepted + rejected
    df["acceptance_rate"] = np.where(submitted > 0, accepted / submitted, np.nan)

    df["arrival_rate"] = np.where(accepted > 0, df["restaurant_arrivals"] / accepted, np.nan)
    df["delivery_completion"] = np.where(accepted > 0, df["delivered"] / accepted, np.nan)

    df["overdue_rate"] = np.where(
        df["delivered"] > 0, df["overdue"] / df["delivered"], np.nan
    )
    df["severely_overdue_rate"] = np.where(
        df["delivered"] > 0, df["severely_overdue"] / df["delivered"], np.nan
    )
    df["large_share"] = np.where(
        df["delivered"] > 0, df["large_orders_completed"] / df["delivered"], np.nan
    )

    df["online_to_qualified_ratio"] = np.where(
        df["online_min"] > 0, df["total_qualified_min"] / df["online_min"], np.nan
    )

    inf_mask = np.isinf(df.select_dtypes(include="number"))
    df[df.select_dtypes(include="number").columns] = (
        df.select_dtypes(include="number").mask(inf_mask)
    )
    return df


def aggregate_to_courier_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate active-day rows to one row per courier.

    Activity rate: active_days / total_calendar_days_in_window.
    Means are computed over active days only; std measures consistency.
    """
    total_days = (df["date"].max() - df["date"].min()).days + 1
    active = df[df["on_shift_bool"]].copy()

    g = active.groupby("courier_id")

    agg = g.agg(
        active_days=("date", "nunique"),
        first_seen=("date", "min"),
        last_seen=("date", "max"),
        total_delivered=("delivered", "sum"),
        total_accepted=("accepted", "sum"),
        total_overdue=("overdue", "sum"),
        total_severely_overdue=("severely_overdue", "sum"),
        total_large_orders=("large_orders_completed", "sum"),
        mean_delivered_per_day=("delivered", "mean"),
        std_delivered_per_day=("delivered", "std"),
        mean_online_min=("online_min", "mean"),
        mean_qualified_min=("total_qualified_min", "mean"),
        mean_peak_min=("peak_online_min", "mean"),
        mean_ontime=("ontime_rate", "mean"),
        mean_large_ontime=("large_ontime_rate", "mean"),
        mean_avg_delivery_time=("avg_delivery_time_min", "mean"),
        mean_prop_over_55=("prop_over_55min", "mean"),
        mean_acceptance=("acceptance_rate", "mean"),
        mean_completion=("delivery_completion", "mean"),
        mean_lunch_peak=("attended_lunch_peak", "mean"),
        mean_dinner_peak=("attended_dinner_peak", "mean"),
        mean_fragmentation=("shift_fragmentation", "mean"),
        weekend_share=("is_weekend_ksa", "mean"),
    )

    agg["activity_rate"] = agg["active_days"] / total_days
    agg["overdue_rate"] = np.where(
        agg["total_delivered"] > 0, agg["total_overdue"] / agg["total_delivered"], 0
    )
    agg["severely_overdue_rate"] = np.where(
        agg["total_delivered"] > 0,
        agg["total_severely_overdue"] / agg["total_delivered"],
        0,
    )
    agg["large_order_share"] = np.where(
        agg["total_delivered"] > 0,
        agg["total_large_orders"] / agg["total_delivered"],
        0,
    )
    agg["delivery_consistency"] = agg["std_delivered_per_day"] / agg["mean_delivered_per_day"].replace(0, np.nan)

    return agg.reset_index()
