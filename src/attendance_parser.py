"""
Parser for the Keeta `Shift_Attendance Summary` field.

Format example:
    "00:00-03:00,Off-Shift,1 hr, 35 min|03:00-08:00,Off-Shift,0 sec|"
    "12:00-16:00,On-Shift,4 hr,qualified|20:00-24:00,On-Shift,2 hr, 6 min,qualified"

Each segment encodes:  TIME_WINDOW, STATE, DURATION [, "qualified"]
Note that DURATION itself can contain a comma ("1 hr, 35 min"), so the
parser splits carefully rather than naively on every comma.
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd

from .data_loader import parse_time_string_to_min


KSA_PEAK_LUNCH = "12:00-16:00"
KSA_PEAK_DINNER = "20:00-24:00"

ALL_BUCKETS = [
    "00:00-03:00",
    "03:00-08:00",
    "08:00-12:00",
    "12:00-16:00",
    "16:00-20:00",
    "20:00-24:00",
]


def parse_attendance_string(s: str) -> List[Dict]:
    """Parse one attendance summary string into a list of segment dicts."""
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for seg in s.split("|"):
        parts = [p.strip() for p in seg.split(",")]
        if len(parts) < 3:
            continue
        time_window, state = parts[0], parts[1]
        qualified = parts[-1].lower() == "qualified"
        duration_parts = parts[2:-1] if qualified else parts[2:]
        duration_str = ", ".join(duration_parts)
        out.append({
            "time_window": time_window,
            "state": state,
            "duration_min": parse_time_string_to_min(duration_str),
            "qualified": qualified,
        })
    return out


def attendance_features(s: str) -> Dict[str, float]:
    """
    Convert a single attendance string into a flat feature dict.

    Features:
      total_on_shift_min        - minutes flagged On-Shift across all buckets
      total_qualified_min       - minutes flagged qualified
      n_qualified_buckets       - count of qualified time buckets (0..6)
      n_on_shift_buckets        - count of buckets with any On-Shift presence
      attended_lunch_peak       - 1 if any On-Shift in 12:00-16:00 else 0
      attended_dinner_peak      - 1 if any On-Shift in 20:00-24:00 else 0
      peak_buckets_attended     - count of peak buckets attended (0..2)
      shift_fragmentation       - number of distinct on-shift "blocks"
                                  (state transitions Off->On)
    """
    feats = {
        "total_on_shift_min": 0.0,
        "total_qualified_min": 0.0,
        "n_qualified_buckets": 0,
        "n_on_shift_buckets": 0,
        "attended_lunch_peak": 0,
        "attended_dinner_peak": 0,
        "peak_buckets_attended": 0,
        "shift_fragmentation": 0,
    }
    segments = parse_attendance_string(s)
    if not segments:
        return feats

    prev_on = False
    transitions = 0
    for seg in segments:
        is_on = seg["state"] == "On-Shift" and (seg["duration_min"] or 0) > 0
        if is_on:
            feats["total_on_shift_min"] += seg["duration_min"]
            feats["n_on_shift_buckets"] += 1
            if seg["qualified"]:
                feats["total_qualified_min"] += seg["duration_min"]
                feats["n_qualified_buckets"] += 1
            if seg["time_window"] == KSA_PEAK_LUNCH:
                feats["attended_lunch_peak"] = 1
            if seg["time_window"] == KSA_PEAK_DINNER:
                feats["attended_dinner_peak"] = 1
            if not prev_on:
                transitions += 1
        prev_on = is_on

    feats["peak_buckets_attended"] = feats["attended_lunch_peak"] + feats["attended_dinner_peak"]
    feats["shift_fragmentation"] = transitions
    return feats


def add_attendance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply attendance_features() row-by-row and concatenate to the dataframe."""
    out = df["attendance_summary"].apply(attendance_features).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
