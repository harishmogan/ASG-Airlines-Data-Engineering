"""Passenger data cleaning module.

Performs:
- Identifier type protection and leading zero padding (e.g. 12-digit Aadhaar).
- Collision detection for non-unique business passenger IDs.
- Surrogate passenger key assignment to prevent data loss.
- Objectively reparable name separation for concatenated name values.
- Sensitive PII masking (Aadhaar, Phone, Email) for privacy preservation.
"""

import re
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np


def mask_aadhaar(aadhaar_str: Any) -> str:
    """Masks a 12-digit Aadhaar number as XXXX-XXXX-1234."""
    if pd.isnull(aadhaar_str):
        return "XXXX-XXXX-XXXX"
    val = str(aadhaar_str).strip()
    val = val.zfill(12)
    return f"XXXX-XXXX-{val[-4:]}"


def mask_phone(phone_str: Any) -> str:
    """Masks a phone number as +91-XXXXXX1234."""
    if pd.isnull(phone_str):
        return "+91-XXXXXXXXXX"
    val = str(phone_str).strip()
    digits = re.sub(r"[^\d]", "", val)
    last4 = digits[-4:] if len(digits) >= 4 else "0000"
    return f"+91-XXXXXX{last4}"


def mask_email(email_str: Any) -> str:
    """Masks an email as u***r@domain.com."""
    if pd.isnull(email_str):
        return "masked@airline.internal"
    val = str(email_str).strip()
    if "@" not in val:
        return "masked@airline.internal"
    user_part, domain_part = val.split("@", 1)
    if len(user_part) <= 2:
        masked_user = user_part[0] + "***" if len(user_part) > 0 else "***"
    else:
        masked_user = f"{user_part[0]}***{user_part[-1]}"
    return f"{masked_user}@{domain_part}"


def split_concatenated_name(first_name_str: str) -> Tuple[str, str]:
    """Splits concatenated first and last names (e.g. IshaanKulkarni -> Ishaan, Kulkarni)."""
    if pd.isnull(first_name_str):
        return "", ""
    name = str(first_name_str).strip()

    # Match CamelCase: e.g. IshaanKulkarni, MyraSharma, ArjunSingh
    match = re.match(r"^([A-Z][a-z]+)([A-Z][a-z]+)$", name)
    if match:
        return match.group(1), match.group(2)

    # Common surnames in dataset
    common_surnames = [
        "Kulkarni", "Naidu", "Sharma", "Singh", "Patel",
        "Gupta", "Nair", "Pillai", "Dewan", "Chahal"
    ]
    for s in common_surnames:
        if name.lower().endswith(s.lower()):
            idx = len(name) - len(s)
            fn = name[:idx].rstrip("pP")
            return fn, s

    return name, ""


def clean_passengers_data(
    raw_passengers: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Cleans passenger records, handles ID collisions, pads Aadhaar zeros, and masks PII.

    Args:
        raw_passengers: Raw passengers DataFrame.

    Returns:
        Tuple of (cleaned_passengers_df, anomalies_list)
    """
    df = raw_passengers.copy()
    anomalies: List[Dict[str, Any]] = []

    # 1. Clean string fields and whitespace
    string_cols = ["passenger_id", "first_name", "last_name", "gender", "email", "phone", "aadhaar_id"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan

    # 2. Preserve and format Aadhaar strings (strictly 12 digits with leading zeros)
    for idx, row in df.iterrows():
        raw_aadhaar = row["aadhaar_id"]
        pid = row["passenger_id"]
        if pd.notnull(raw_aadhaar):
            clean_aadhaar = str(raw_aadhaar).strip().zfill(12)
            if len(raw_aadhaar) < 12:
                anomalies.append({
                    "entity": "Passenger",
                    "record_identifier": pid,
                    "field": "aadhaar_id",
                    "issue_type": "LEADING_ZERO_TRUNCATED",
                    "severity": "MEDIUM",
                    "description": f"Aadhaar length was {len(raw_aadhaar)} digits; restored leading zeros to 12 digits.",
                    "whether_corrected": "YES",
                    "safe_reference": mask_aadhaar(clean_aadhaar),
                })
            df.loc[idx, "aadhaar_id"] = clean_aadhaar

    # 3. Objectively resolve missing last names where first_name has concatenated surname
    for idx, row in df.iterrows():
        ln = row["last_name"]
        fn = row["first_name"]
        pid = row["passenger_id"]

        if pd.isnull(ln) or ln in ["nan", "None", ""]:
            sep_fn, sep_ln = split_concatenated_name(fn)
            if sep_ln:
                df.loc[idx, "first_name"] = sep_fn
                df.loc[idx, "last_name"] = sep_ln
                anomalies.append({
                    "entity": "Passenger",
                    "record_identifier": pid,
                    "field": "last_name",
                    "issue_type": "CONCATENATED_NAME",
                    "severity": "LOW",
                    "description": f"Last name was missing; extracted '{sep_ln}' from combined first name '{fn}'.",
                    "whether_corrected": "YES",
                    "safe_reference": f"{sep_fn} {sep_ln[0]}.",
                })

    # 4. Standardize Gender, Age, and Date of Birth
    df["gender"] = df["gender"].str.upper()
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")

    # 5. Collision Detection: Identical passenger_id assigned to different individuals
    pid_counts = df.groupby("passenger_id")["passenger_id"].transform("count")
    df["is_collision"] = pid_counts > 1

    colliding_pids = df[df["is_collision"]]["passenger_id"].unique()
    for c_id in colliding_pids:
        rows_matching = df[df["passenger_id"] == c_id]
        anomalies.append({
            "entity": "Passenger",
            "record_identifier": c_id,
            "field": "passenger_id",
            "issue_type": "PASSENGER_ID_COLLISION",
            "severity": "HIGH",
            "description": f"Passenger ID '{c_id}' is shared by {len(rows_matching)} distinct individuals. Assigned unique surrogate keys to preserve all distinct passengers.",
            "whether_corrected": "FLAGGED",
            "safe_reference": f"Collision ID: {c_id}",
        })

    # 6. Assign unique surrogate passenger key
    df["passenger_key"] = range(1, len(df) + 1)

    # 7. Generate privacy-safe masked fields for Gold/Silver
    df["masked_aadhaar"] = df["aadhaar_id"].apply(mask_aadhaar)
    df["masked_phone"] = df["phone"].apply(mask_phone)
    df["masked_email"] = df["email"].apply(mask_email)

    return df, anomalies
