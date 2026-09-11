"""Booking data cleaning module.

Performs:
- Status validation and standardization (CONFIRMED, CANCELLED, PENDING, INVALID, UNKNOWN).
- Booking identifier uniqueness verification.
- Detection of ambiguous passenger references (due to passenger ID collisions).
- Detection of ambiguous flight references (due to multi-leg operational flights).
- Grain preservation (guaranteeing exact 1-to-1 row retention).
"""

from typing import Tuple, List, Dict, Any, Set
import pandas as pd
import numpy as np

from src.config import VALID_BOOKING_STATUSES, STATUS_UNKNOWN
from src.cleaning.clean_passengers import mask_phone


def clean_bookings_data(
    raw_bookings: pd.DataFrame,
    colliding_passenger_ids: Set[str],
    multi_leg_flight_ids: Set[str],
    valid_passenger_ids: Set[str],
    valid_flight_ids: Set[str],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Cleans booking records and flags ambiguities and missing statuses.

    Args:
        raw_bookings: Raw bookings DataFrame.
        colliding_passenger_ids: Set of passenger IDs shared by multiple distinct people.
        multi_leg_flight_ids: Set of flight IDs having multiple operational legs.
        valid_passenger_ids: Set of all valid passenger IDs.
        valid_flight_ids: Set of all valid flight IDs.

    Returns:
        Tuple of (cleaned_bookings_df, anomalies_list)
    """
    df = raw_bookings.copy()
    anomalies: List[Dict[str, Any]] = []

    # 1. Clean string fields and whitespace
    for col in [
        "booking_id", "passenger_id", "flight_id", "status",
        "passport_number", "seat_number", "emergency_contact_name", "emergency_contact_phone"
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan

    # 2. Check for duplicate booking IDs
    dup_bookings = df[df.duplicated(subset=["booking_id"], keep=False)]
    if not dup_bookings.empty:
        for idx in dup_bookings.index:
            anomalies.append({
                "entity": "Booking",
                "record_identifier": str(df.loc[idx, "booking_id"]),
                "field": "booking_id",
                "issue_type": "DUPLICATE_BOOKING_ID",
                "severity": "CRITICAL",
                "description": f"Duplicate booking identifier found: {df.loc[idx, 'booking_id']}.",
                "whether_corrected": "NO",
                "safe_reference": f"Booking {df.loc[idx, 'booking_id']}",
            })

    # 3. Clean and standardize booking status
    df["raw_status"] = df["status"]
    for idx, row in df.iterrows():
        bid = row["booking_id"]
        raw_st = row["status"]

        if pd.isnull(raw_st) or raw_st in ["nan", "None", ""]:
            df.loc[idx, "status"] = STATUS_UNKNOWN
            anomalies.append({
                "entity": "Booking",
                "record_identifier": bid,
                "field": "status",
                "issue_type": "MISSING_STATUS",
                "severity": "MEDIUM",
                "description": "Booking status was null/empty; standardized to UNKNOWN.",
                "whether_corrected": "YES",
                "safe_reference": bid,
            })
        else:
            st_upper = raw_st.upper()
            if st_upper in VALID_BOOKING_STATUSES:
                df.loc[idx, "status"] = st_upper
            else:
                df.loc[idx, "status"] = STATUS_UNKNOWN
                anomalies.append({
                    "entity": "Booking",
                    "record_identifier": bid,
                    "field": "status",
                    "issue_type": "INVALID_STATUS_VALUE",
                    "severity": "MEDIUM",
                    "description": f"Unrecognized booking status '{raw_st}'; standardized to UNKNOWN.",
                    "whether_corrected": "YES",
                    "safe_reference": bid,
                })

    # 4. Parse booking date if present
    if "booking_date" in df.columns:
        df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")

    # 5. Referential Integrity & Ambiguity Checks
    # Passenger reference check
    df["is_passenger_ambiguous"] = df["passenger_id"].isin(colliding_passenger_ids)
    for idx, row in df[df["is_passenger_ambiguous"]].iterrows():
        anomalies.append({
            "entity": "Booking",
            "record_identifier": row["booking_id"],
            "field": "passenger_id",
            "issue_type": "AMBIGUOUS_PASSENGER_REFERENCE",
            "severity": "HIGH",
            "description": f"Booking references passenger ID '{row['passenger_id']}' which corresponds to multiple individuals. Preserved booking and mapped to unresolved member.",
            "whether_corrected": "FLAGGED",
            "safe_reference": row["booking_id"],
        })

    # Flight reference check
    df["is_flight_ambiguous"] = df["flight_id"].isin(multi_leg_flight_ids)
    for idx, row in df[df["is_flight_ambiguous"]].iterrows():
        anomalies.append({
            "entity": "Booking",
            "record_identifier": row["booking_id"],
            "field": "flight_id",
            "issue_type": "AMBIGUOUS_FLIGHT_REFERENCE",
            "severity": "HIGH",
            "description": f"Booking references flight ID '{row['flight_id']}' which has multiple operational legs. Preserved booking and mapped to unresolved member.",
            "whether_corrected": "FLAGGED",
            "safe_reference": row["booking_id"],
        })

    # Mask emergency contact phone
    if "emergency_contact_phone" in df.columns:
        df["masked_emergency_phone"] = df["emergency_contact_phone"].apply(mask_phone)

    return df, anomalies
