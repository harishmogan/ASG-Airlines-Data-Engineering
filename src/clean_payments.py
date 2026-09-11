"""Payment data cleaning module.

Performs:
- Monetary amount cleaning, validation, and invalid value detection.
- Payment method standardization.
- Referential integrity verification against booking records.
- Strict separation of Gross Processed Payments and Realized Confirmed Revenue.
"""

from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

from src.config import VALID_PAYMENT_METHODS


def clean_payments_data(
    raw_payments: pd.DataFrame,
    bookings_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Cleans payment records, flags null/invalid amounts, and maps booking status.

    Args:
        raw_payments: Raw payments DataFrame.
        bookings_df: Cleaned bookings DataFrame containing booking_id and status.

    Returns:
        Tuple of (cleaned_payments_df, anomalies_list)
    """
    df = raw_payments.copy()
    anomalies: List[Dict[str, Any]] = []

    # 1. Clean string fields and whitespace
    for col in ["payment_id", "booking_id", "payment_method"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan

    # 2. Check for duplicate payment IDs
    dup_payments = df[df.duplicated(subset=["payment_id"], keep=False)]
    if not dup_payments.empty:
        for idx in dup_payments.index:
            anomalies.append({
                "entity": "Payment",
                "record_identifier": str(df.loc[idx, "payment_id"]),
                "field": "payment_id",
                "issue_type": "DUPLICATE_PAYMENT_ID",
                "severity": "CRITICAL",
                "description": f"Duplicate payment ID found: {df.loc[idx, 'payment_id']}.",
                "whether_corrected": "NO",
                "safe_reference": df.loc[idx, "payment_id"],
            })

    # 3. Clean and validate payment amount
    df["raw_amount"] = df["amount"]
    df["amount_cleaned"] = pd.to_numeric(df["amount"], errors="coerce")
    df["is_valid_amount"] = df["amount_cleaned"].notnull() & (df["amount_cleaned"] > 0)

    for idx, row in df.iterrows():
        pid = row["payment_id"]
        raw_amt = str(row["raw_amount"]).strip()

        if pd.isnull(row["raw_amount"]) or raw_amt in ["nan", "None", ""]:
            anomalies.append({
                "entity": "Payment",
                "record_identifier": pid,
                "field": "amount",
                "issue_type": "MISSING_PAYMENT_AMOUNT",
                "severity": "HIGH",
                "description": "Payment amount was null/missing.",
                "whether_corrected": "FLAGGED",
                "safe_reference": pid,
            })
        elif raw_amt.upper() == "INVALID":
            anomalies.append({
                "entity": "Payment",
                "record_identifier": pid,
                "field": "amount",
                "issue_type": "INVALID_PAYMENT_AMOUNT",
                "severity": "HIGH",
                "description": "Payment amount was corrupted with literal text 'INVALID'.",
                "whether_corrected": "FLAGGED",
                "safe_reference": pid,
            })
        elif row["amount_cleaned"] <= 0:
            anomalies.append({
                "entity": "Payment",
                "record_identifier": pid,
                "field": "amount",
                "issue_type": "NON_POSITIVE_PAYMENT_AMOUNT",
                "severity": "HIGH",
                "description": f"Payment amount was non-positive ({row['amount_cleaned']}).",
                "whether_corrected": "FLAGGED",
                "safe_reference": pid,
            })

    # 4. Standardize Payment Method
    df["payment_method"] = df["payment_method"].str.upper()
    invalid_methods = df[~df["payment_method"].isin(VALID_PAYMENT_METHODS)]
    for idx in invalid_methods.index:
        anomalies.append({
            "entity": "Payment",
            "record_identifier": df.loc[idx, "payment_id"],
            "field": "payment_method",
            "issue_type": "INVALID_PAYMENT_METHOD",
            "severity": "MEDIUM",
            "description": f"Unrecognized payment method '{df.loc[idx, 'payment_method']}'.",
            "whether_corrected": "NO",
            "safe_reference": df.loc[idx, "payment_id"],
        })

    # 5. Referential Integrity & Revenue Status Classification
    # Map booking status from bookings_df
    booking_status_map = dict(zip(bookings_df["booking_id"], bookings_df["status"]))
    df["booking_status"] = df["booking_id"].map(booking_status_map).fillna("UNKNOWN")

    unmatched_bookings = df[~df["booking_id"].isin(booking_status_map)]
    for idx in unmatched_bookings.index:
        anomalies.append({
            "entity": "Payment",
            "record_identifier": df.loc[idx, "payment_id"],
            "field": "booking_id",
            "issue_type": "ORPHAN_PAYMENT_BOOKING_REF",
            "severity": "HIGH",
            "description": f"Payment references non-existent booking ID '{df.loc[idx, 'booking_id']}'.",
            "whether_corrected": "NO",
            "safe_reference": df.loc[idx, "payment_id"],
        })

    # Revenue Classification Flags
    df["is_realized_revenue"] = df["is_valid_amount"] & (df["booking_status"] == "CONFIRMED")
    df["is_cancelled_booking_payment"] = df["is_valid_amount"] & (df["booking_status"] == "CANCELLED")
    df["is_pending_booking_payment"] = df["is_valid_amount"] & (df["booking_status"] == "PENDING")

    return df, anomalies
