"""Flight data cleaning module.

Performs:
- Missing/unknown airline recovery via flight_id prefix mapping.
- Exact duplicate record removal.
- Distinct operational legs preservation for multi-leg flight IDs (e.g. 6F250).
- General timestamp inversion repair.
- Duration calculation and suspicious duration / outlier detection.
- Overnight flight detection.
"""

from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

from src.config import AIRLINE_PREFIX_MAP


def clean_flights_data(
    raw_flights: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Cleans raw flight data and generates anomaly logs for transformations.

    Args:
        raw_flights: Raw flights DataFrame.

    Returns:
        Tuple of (cleaned_flights_df, anomalies_list)
    """
    df = raw_flights.copy()
    anomalies: List[Dict[str, Any]] = []

    # 1. Clean string fields and whitespace
    for col in ["flight_id", "airline", "source", "destination"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Replace string 'nan' / 'None' with actual NaN
            df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan

    # 2. Record and remove exact duplicate rows
    exact_dups = df[df.duplicated(keep=False)]
    if not exact_dups.empty:
        dup_indices = df[df.duplicated(keep="first")].index
        for idx in dup_indices:
            anomalies.append({
                "entity": "Flight",
                "record_identifier": str(df.loc[idx, "flight_id"]),
                "field": "ALL",
                "issue_type": "EXACT_DUPLICATE",
                "severity": "MEDIUM",
                "description": f"Identical operational flight row duplicated; removing redundant row.",
                "whether_corrected": "YES",
                "safe_reference": f"Flight {df.loc[idx, 'flight_id']}",
            })
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

    # 3. Recover missing or 'UNKNOWN' airlines using prefix mapping
    for idx, row in df.iterrows():
        airline = row["airline"]
        fid = str(row["flight_id"])
        prefix = fid[:2]
        expected_airline = AIRLINE_PREFIX_MAP.get(prefix)

        if pd.isnull(airline) or airline.upper() == "UNKNOWN":
            if expected_airline:
                df.loc[idx, "airline"] = expected_airline
                anomalies.append({
                    "entity": "Flight",
                    "record_identifier": fid,
                    "field": "airline",
                    "issue_type": "MISSING_OR_UNKNOWN_AIRLINE",
                    "severity": "LOW",
                    "description": f"Airline was '{airline}'; objectively recovered as '{expected_airline}' from prefix '{prefix}'.",
                    "whether_corrected": "YES",
                    "safe_reference": fid,
                })
            else:
                anomalies.append({
                    "entity": "Flight",
                    "record_identifier": fid,
                    "field": "airline",
                    "issue_type": "UNRESOLVED_AIRLINE",
                    "severity": "MEDIUM",
                    "description": f"Airline is missing and prefix '{prefix}' is unrecognized.",
                    "whether_corrected": "NO",
                    "safe_reference": fid,
                })

    # 4. Parse timestamps
    df["departure_time"] = pd.to_datetime(df["departure_time"], errors="coerce")
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")

    # 5. General Timestamp Inversion Repair
    # If arrival_time < departure_time, repair arrival date objectively
    for idx, row in df.iterrows():
        dep = row["departure_time"]
        arr = row["arrival_time"]
        fid = row["flight_id"]

        if pd.notnull(dep) and pd.notnull(arr) and arr < dep:
            raw_arr_str = str(arr)
            raw_dep_str = str(dep)
            # General rule:
            # If arrival time-of-day >= departure time-of-day, the flight arrived on the same day as departure
            # If arrival time-of-day < departure time-of-day, the flight arrived the next day (overnight)
            if arr.time() >= dep.time():
                corrected_arr = datetime.combine(dep.date(), arr.time())
            else:
                corrected_arr = datetime.combine(dep.date() + timedelta(days=1), arr.time())

            df.loc[idx, "arrival_time"] = corrected_arr
            df.loc[idx, "had_inverted_timestamps"] = True
            anomalies.append({
                "entity": "Flight",
                "record_identifier": fid,
                "field": "arrival_time",
                "issue_type": "INVERTED_TIMESTAMPS",
                "severity": "HIGH",
                "description": f"Arrival timestamp ({raw_arr_str}) occurred before departure ({raw_dep_str}). Objectively corrected arrival date to {corrected_arr.date()} based on operational time component.",
                "whether_corrected": "YES",
                "safe_reference": fid,
            })

    # 6. Calculate Duration in Minutes objectively from timestamps
    df["duration_minutes"] = (
        (df["arrival_time"] - df["departure_time"]).dt.total_seconds() / 60.0
    ).round(2)

    # 7. Overnight flight flag
    df["is_overnight"] = df["arrival_time"].dt.date > df["departure_time"].dt.date

    # 8. Detect Multi-Leg Flight collisions (e.g. 6F250)
    flight_counts = df.groupby("flight_id")["flight_id"].transform("count")
    df["is_multi_leg_collision"] = flight_counts > 1

    multi_leg_ids = df[df["is_multi_leg_collision"]]["flight_id"].unique()
    for m_id in multi_leg_ids:
        anomalies.append({
            "entity": "Flight",
            "record_identifier": m_id,
            "field": "flight_id",
            "issue_type": "MULTI_LEG_COLLISION",
            "severity": "HIGH",
            "description": f"Flight ID '{m_id}' corresponds to multiple distinct operational legs. Assigned distinct surrogate keys to avoid data loss.",
            "whether_corrected": "FLAGGED",
            "safe_reference": m_id,
        })

    if "had_inverted_timestamps" in df.columns:
        has_inverted = df["had_inverted_timestamps"].eq(True)
        df.drop(columns=["had_inverted_timestamps"], inplace=True)
    else:
        has_inverted = pd.Series(False, index=df.index, dtype=bool)

    df["is_duration_outlier"] = has_inverted | (df["duration_minutes"] <= 0) | (df["duration_minutes"] > 360.0)

    outlier_flights = df[df["is_duration_outlier"]]
    for idx, row in outlier_flights.iterrows():
        # Avoid duplicate anomaly logging if already logged as INVERTED_TIMESTAMPS
        if not any(a["record_identifier"] == str(row["flight_id"]) and a["issue_type"] == "INVERTED_TIMESTAMPS" for a in anomalies):
            anomalies.append({
                "entity": "Flight",
                "record_identifier": str(row["flight_id"]),
                "field": "duration_minutes",
                "issue_type": "DURATION_OUTLIER",
                "severity": "LOW",
                "description": f"Flight duration of {row['duration_minutes']} minutes flagged as statistical or domain outlier.",
                "whether_corrected": "FLAGGED",
                "safe_reference": str(row["flight_id"]),
            })

    # 10. Assign unique operational flight surrogate key
    df["flight_key"] = range(1, len(df) + 1)

    return df, anomalies
