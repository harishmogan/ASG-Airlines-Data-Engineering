"""Excel ingestion module for the Airlines Data Pipeline.

Efficiently loads all sheets from the authoritative raw workbook using a single
pd.ExcelFile instance, preserving string identifiers, leading zeros, and timestamps.
"""

from pathlib import Path
from typing import Dict
import pandas as pd

from src.config import (
    RAW_EXCEL_PATH,
    RAW_DTYPES,
    SHEET_FLIGHTS,
    SHEET_PAYMENTS,
    SHEET_BOOKINGS,
    SHEET_PASSENGERS,
)


def load_raw_workbook(file_path: Path = RAW_EXCEL_PATH) -> Dict[str, pd.DataFrame]:
    """Loads all four sheets from the raw Excel workbook in a single open operation.

    Args:
        file_path: Path to the Excel workbook.

    Returns:
        Dictionary mapping sheet name to its raw DataFrame.

    Raises:
        FileNotFoundError: If the Excel file does not exist.
        ValueError: If required sheets are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw Excel workbook not found at: {path}")

    required_sheets = {SHEET_FLIGHTS, SHEET_PAYMENTS, SHEET_BOOKINGS, SHEET_PASSENGERS}

    # Open the workbook once using pd.ExcelFile for optimal efficiency
    with pd.ExcelFile(path) as xl:
        available_sheets = set(xl.sheet_names)
        missing = required_sheets - available_sheets
        if missing:
            raise ValueError(f"Workbook is missing required sheets: {missing}")

        raw_data = {}
        for sheet_name in required_sheets:
            dtypes = RAW_DTYPES.get(sheet_name, {})
            # Parse sheet preserving identifier string types
            df = xl.parse(sheet_name, dtype=dtypes)
            raw_data[sheet_name] = df

    return raw_data
