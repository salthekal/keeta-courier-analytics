"""
Data loading and cleaning for Keeta courier-day operational scorecards.

Handles the quirks of the platform export:
  - "-" placeholders for inactive days  -> NaN
  - Numeric columns shipped as object dtype
  - Time-of-day strings ("11 hr, 54 min")  -> minutes (float)
  - YYYYMMDD integer dates                -> datetime
  - Courier IDs hashed for privacy
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Original column names from the Keeta export (with leading-space "Date" preserved)
COL_RENAMES = {
    " Date": "date",
    "Date": "date",
    "Courier ID": "courier_id_raw",
    "Courier First Name": "first_name",
    "Courier Last Name": "last_name",
    "Supervisor": "supervisor",
    "Vehicle Type": "vehicle_type",
    "Shift_Attendance Summary": "attendance_summary",
    "Shift_On-Shift?": "on_shift",
    "Shift_Valid Day?": "valid_day",
    "Shift_Courier App Online Time": "online_time_str",
    "Shift_Valid Online Time": "valid_online_time_str",
    "Shift_Peak Online Hours": "peak_online_time_str",
    "Task Volumes_Accepted Tasks": "accepted",
    "Task Volumes_Tasks with restaurant arrivals": "restaurant_arrivals",
    "Task Volumes_Delivered Tasks": "delivered",
    "Task Volumes_Large Order Tasks Completed": "large_orders_completed",
    "Task Volumes_Rejected Tasks": "rejected_total",
    "Task Volumes_Rejected Tasks (Courier)": "rejected_courier",
    "Task Volumes_Rejected Tasks (Auto)": "rejected_auto",
    "Task Volumes_Cancellation Rate from Delivery Issues": "cancel_rate_delivery_issues",
    "Task Volumes_Order completion rate (non-delivery related)": "completion_rate_non_delivery",
    "Delivery Experience_On-time Rate (D)": "ontime_rate",
    "Delivery Experience_Large order on-time rate": "large_ontime_rate",
    "Delivery Experience_Avg Delivery Time of Delivered Orders": "avg_delivery_time_min",
    "Delivery Experience_Delivered Orders Prop. (Over 55min)": "prop_over_55min",
    "Delivery Experience_Overdue Order Tasks": "overdue",
    "Delivery Experience_Severely Overdue Order Tasks": "severely_overdue",
}

NUMERIC_COLS = [
    "accepted",
    "restaurant_arrivals",
    "delivered",
    "large_orders_completed",
    "rejected_total",
    "rejected_courier",
    "rejected_auto",
    "cancel_rate_delivery_issues",
    "completion_rate_non_delivery",
    "ontime_rate",
    "large_ontime_rate",
    "avg_delivery_time_min",
    "prop_over_55min",
    "overdue",
    "severely_overdue",
]


def parse_time_string_to_min(s) -> float:
    """Convert '11 hr, 54 min' / '0 sec' / '12 min, 36 sec' -> total minutes."""
    if not isinstance(s, str) or not s.strip():
        return np.nan
    s = s.strip().lower()
    if s in {"-", ""}:
        return np.nan
    total = 0.0
    for value, unit in re.findall(r"(\d+)\s*(hr|min|sec)", s):
        v = int(value)
        if unit == "hr":
            total += v * 60.0
        elif unit == "min":
            total += v
        elif unit == "sec":
            total += v / 60.0
    return total


def hash_courier_id(raw_id, salt: str = "keeta-portfolio") -> str:
    """Stable, irreversible 12-char hash of a courier ID."""
    payload = f"{salt}:{raw_id}".encode()
    return "C" + hashlib.sha256(payload).hexdigest()[:11].upper()


def load_keeta_scorecard(
    path: str | Path,
    salt: str = "keeta-portfolio",
    drop_pii: bool = True,
) -> pd.DataFrame:
    """
    Load + clean + anonymize a Keeta courier-day scorecard export.

    Parameters
    ----------
    path : str | Path
        Path to the .xlsx export.
    salt : str
        Salt used when hashing courier IDs.
    drop_pii : bool
        If True (default), drops first/last name columns after hashing.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe, one row per courier-day.
    """
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={k.strip(): v for k, v in COL_RENAMES.items()})

    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["courier_id"] = df["courier_id_raw"].apply(lambda x: hash_courier_id(x, salt))

    df["on_shift_bool"] = df["on_shift"].astype(str).str.strip().str.lower().eq("yes")
    df["valid_day_bool"] = df["valid_day"].astype(str).str.strip().str.lower().eq("yes")

    df["online_min"] = df["online_time_str"].apply(parse_time_string_to_min)
    df["valid_online_min"] = df["valid_online_time_str"].apply(parse_time_string_to_min)
    df["peak_online_min"] = df["peak_online_time_str"].apply(parse_time_string_to_min)

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col].replace({"-": np.nan}), errors="coerce")

    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend_ksa"] = df["dow"].isin([4, 5])

    ramadan_start = pd.Timestamp("2026-02-17")
    ramadan_end = pd.Timestamp("2026-03-19")
    df["is_ramadan"] = df["date"].between(ramadan_start, ramadan_end)

    if drop_pii:
        df = df.drop(columns=["first_name", "last_name", "courier_id_raw"], errors="ignore")

    return df


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "data/keeta_scorecard.xlsx"
    out = load_keeta_scorecard(p)
    print(f"loaded {len(out)} rows, {out['courier_id'].nunique()} couriers, "
          f"{out['date'].min().date()} to {out['date'].max().date()}")
