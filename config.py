"""Configuration module for the Airlines Data Pipeline (Portable Distribution).

Defines paths, constants, categorical mappings, and data schema rules.
"""

from pathlib import Path

# Base Paths - relative to this file's parent.parent (AIRLINE_FINAL)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Check data/raw first, fallback to root
_raw_candidate = PROJECT_ROOT / "data" / "raw" / "UseCase - Airlines.xlsx"
RAW_EXCEL_PATH = (
    _raw_candidate
    if _raw_candidate.exists()
    else PROJECT_ROOT / "UseCase - Airlines.xlsx"
)

DATA_DIR = PROJECT_ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Excel Sheet Names
SHEET_FLIGHTS = "flights"
SHEET_PAYMENTS = "payments"
SHEET_BOOKINGS = "bookings"
SHEET_PASSENGERS = "passengers"

# Airline Prefix Mapping (objective recovery of missing airline names)
AIRLINE_PREFIX_MAP = {
    "6F": "IndiGo",
    "AI": "Air India",
    "SJ": "SpiceJet",
    "UK": "Vistara",
}

# Standardized Booking Statuses
VALID_BOOKING_STATUSES = {"CONFIRMED", "CANCELLED", "PENDING", "INVALID"}
STATUS_UNKNOWN = "UNKNOWN"

# Standardized Payment Methods
VALID_PAYMENT_METHODS = {"UPI", "CARD", "NETBANKING"}

# Strict Ingestion Dtypes to preserve leading zeros and prevent identifier truncation
RAW_DTYPES = {
    SHEET_FLIGHTS: {
        "flight_id": str,
        "airline": str,
        "source": str,
        "destination": str,
    },
    SHEET_PASSENGERS: {
        "passenger_id": str,
        "first_name": str,
        "last_name": str,
        "gender": str,
        "email": str,
        "phone": str,
        "aadhaar_id": str,
    },
    SHEET_BOOKINGS: {
        "booking_id": str,
        "passenger_id": str,
        "flight_id": str,
        "status": str,
        "passport_number": str,
        "seat_number": str,
        "emergency_contact_name": str,
        "emergency_contact_phone": str,
    },
    SHEET_PAYMENTS: {
        "payment_id": str,
        "booking_id": str,
        "payment_method": str,
    },
}

# Special surrogate key identifiers for ambiguous/unresolved dimensional members
UNRESOLVED_KEY = -1
UNRESOLVED_CODE = "UNRESOLVED"
