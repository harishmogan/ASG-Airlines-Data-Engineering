"""Reusable data validation functions for the Airlines Data Pipeline.

Provides vectorized checks for schema requirements, value formats, timestamp logic,
referential integrity, and business identifier collision detection.
"""

from typing import List, Set, Tuple
import pandas as pd


def validate_required_fields(
    df: pd.DataFrame, required_cols: List[str]
) -> pd.DataFrame:
    """Identifies rows where required fields are missing, empty, or whitespace-only.

    Args:
        df: Input DataFrame.
        required_cols: List of column names that must not be null/empty.

    Returns:
        DataFrame containing rows with missing required fields along with the missing column name.
    """
    failures = []
    for col in required_cols:
        if col not in df.columns:
            continue
        # Check null, string 'nan', or blank whitespace
        is_missing = (
            df[col].isnull()
            | (df[col].astype(str).str.strip() == "")
            | (df[col].astype(str).str.lower() == "nan")
            | (df[col].astype(str).str.lower() == "none")
        )
        for idx in df[is_missing].index:
            failures.append({
                "row_index": idx,
                "column": col,
                "value": df.loc[idx, col],
            })
    return pd.DataFrame(failures)


def validate_categorical_values(
    series: pd.Series, allowed_values: Set[str], allow_null: bool = False
) -> pd.Series:
    """Validates that values in a series belong to an allowed set of categories.

    Args:
        series: Pandas Series to validate.
        allowed_values: Set of valid category strings (case-insensitive check).
        allow_null: Whether nulls are permitted.

    Returns:
        Boolean Series where True indicates valid values.
    """
    clean_series = series.astype(str).str.strip().str.upper()
    allowed_upper = {v.upper() for v in allowed_values}

    is_valid = clean_series.isin(allowed_upper)
    if allow_null:
        is_valid = is_valid | series.isnull()
    return is_valid


def validate_positive_numeric(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Validates that values can be converted to positive numbers (> 0).

    Args:
        series: Pandas Series of amounts or measurements.

    Returns:
        Tuple of (numeric_series, is_valid_mask)
    """
    numeric = pd.to_numeric(series, errors="coerce")
    is_valid = numeric.notnull() & (numeric > 0)
    return numeric, is_valid


def validate_timestamps(
    dep_series: pd.Series, arr_series: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Validates departure and arrival timestamps and flags inverted pairs.

    Args:
        dep_series: Departure timestamps.
        arr_series: Arrival timestamps.

    Returns:
        Tuple of (parsed_dep, parsed_arr, is_inverted_mask)
    """
    dep = pd.to_datetime(dep_series, errors="coerce")
    arr = pd.to_datetime(arr_series, errors="coerce")
    is_inverted = (dep.notnull() & arr.notnull()) & (arr < dep)
    return dep, arr, is_inverted


def detect_duplicates(
    df: pd.DataFrame, subset: List[str] = None, keep: str = False
) -> pd.DataFrame:
    """Identifies duplicated rows according to a subset of columns.

    Args:
        df: Input DataFrame.
        subset: Columns to check for duplicates. If None, checks all columns.
        keep: 'first', 'last', or False (to mark all duplicates).

    Returns:
        DataFrame containing the duplicate rows.
    """
    mask = df.duplicated(subset=subset, keep=keep)
    return df[mask]


def detect_referential_integrity(
    child_series: pd.Series, parent_series: pd.Series
) -> pd.Series:
    """Checks referential integrity between a foreign key and primary key.

    Args:
        child_series: Foreign key series in child table.
        parent_series: Primary key series in parent table.

    Returns:
        Boolean Series where True indicates the foreign key exists in the parent.
    """
    parent_keys = set(parent_series.dropna().unique())
    return child_series.isin(parent_keys)


def detect_identifier_collisions(
    df: pd.DataFrame, id_col: str, differentiating_cols: List[str]
) -> pd.DataFrame:
    """Detects collisions where a business ID is reused for distinct real-world entities.

    Args:
        df: Input DataFrame.
        id_col: The business identifier column (e.g., 'passenger_id').
        differentiating_cols: Columns that distinguish entities (e.g., 'aadhaar_id', 'phone').

    Returns:
        DataFrame of rows that share an ID but differ across differentiating columns.
    """
    # Group by id_col and count distinct combinations of differentiating_cols
    grouped = df.groupby(id_col)[differentiating_cols].nunique()
    colliding_ids = grouped[grouped.max(axis=1) > 1].index
    return df[df[id_col].isin(colliding_ids)]
