"""End-to-end Airline Data Engineering and Analytics Pipeline.

Entry Point:
    python src/pipeline.py

Executes:
1. Ingestion: Efficiently loads all raw sheets from UseCase - Airlines.xlsx
2. Validation: Validates input schemas and types
3. Cleaning: Standardizes records, handles collisions/ambiguities, restores PII leading zeros
4. Silver Layer: Exports clean standardized CSVs
5. Gold Layer: Builds Star Schema dimensions and fact tables (guaranteeing zero join fan-out)
6. KPI Computation: Calculates flight, booking, payment, and data quality metrics
7. Anomaly Reporting: Consolidates and exports PII-masked data quality anomaly report
8. Execution Summary: Prints concise operational and financial metrics
"""

import sys
from pathlib import Path

# Add project root to sys.path so pipeline runs cleanly from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    RAW_EXCEL_PATH,
    SILVER_DIR,
    GOLD_DIR,
    SHEET_FLIGHTS,
    SHEET_PAYMENTS,
    SHEET_BOOKINGS,
    SHEET_PASSENGERS,
)
from src.ingestion.excel_loader import load_raw_workbook
from src.validation.validators import (
    validate_required_fields,
    detect_identifier_collisions,
    detect_referential_integrity,
)
from src.cleaning.clean_flights import clean_flights_data
from src.cleaning.clean_passengers import clean_passengers_data
from src.cleaning.clean_bookings import clean_bookings_data
from src.cleaning.clean_payments import clean_payments_data
from src.modeling.star_schema import build_star_schema
from src.analytics.kpis import compute_all_kpis
from src.analytics.anomaly_reporter import generate_anomaly_report


def run_pipeline() -> dict:
    """Executes the complete data engineering pipeline end-to-end.

    Returns:
        Dictionary containing summary execution statistics and output paths.
    """
    print("=" * 70)
    print("          AIRLINE DATA ENGINEERING & ANALYTICS PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. INGESTION
    # ---------------------------------------------------------
    print("\n[Step 1/6] Ingesting raw Excel workbook...")
    raw_data = load_raw_workbook(RAW_EXCEL_PATH)
    raw_flights = raw_data[SHEET_FLIGHTS]
    raw_passengers = raw_data[SHEET_PASSENGERS]
    raw_bookings = raw_data[SHEET_BOOKINGS]
    raw_payments = raw_data[SHEET_PAYMENTS]
    print(f"  -> Ingested {len(raw_flights)} flights, {len(raw_passengers)} passengers, "
          f"{len(raw_bookings)} bookings, {len(raw_payments)} payments.")

    # ---------------------------------------------------------
    # 2. VALIDATION
    # ---------------------------------------------------------
    print("\n[Step 2/6] Executing initial data validations...")
    # Validate required fields
    val_fl = validate_required_fields(raw_flights, ["flight_id", "source", "destination"])
    val_pa = validate_required_fields(raw_passengers, ["passenger_id", "aadhaar_id"])
    val_bo = validate_required_fields(raw_bookings, ["booking_id", "flight_id", "passenger_id"])
    val_py = validate_required_fields(raw_payments, ["payment_id", "booking_id"])
    total_val_issues = len(val_fl) + len(val_pa) + len(val_bo) + len(val_py)
    print(f"  -> Validated schemas. Found {total_val_issues} missing required field issues.")

    # ---------------------------------------------------------
    # 3. CLEANING & SILVER LAYER GENERATION
    # ---------------------------------------------------------
    print("\n[Step 3/6] Cleaning and standardizing source data (Silver Layer)...")
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    # A. Clean Flights
    flights_clean, flight_anomalies = clean_flights_data(raw_flights)

    # B. Clean Passengers
    passengers_clean, passenger_anomalies = clean_passengers_data(raw_passengers)

    # Identify collision sets for bookings mapping
    colliding_passenger_ids = set(passengers_clean[passengers_clean["is_collision"]]["passenger_id"].unique())
    multi_leg_flight_ids = set(flights_clean[flights_clean["is_multi_leg_collision"]]["flight_id"].unique())
    valid_passenger_ids = set(passengers_clean["passenger_id"].unique())
    valid_flight_ids = set(flights_clean["flight_id"].unique())

    # C. Clean Bookings
    bookings_clean, booking_anomalies = clean_bookings_data(
        raw_bookings,
        colliding_passenger_ids,
        multi_leg_flight_ids,
        valid_passenger_ids,
        valid_flight_ids,
    )

    # D. Clean Payments
    payments_clean, payment_anomalies = clean_payments_data(raw_payments, bookings_clean)

    # Export Silver CSVs
    flights_clean.to_csv(SILVER_DIR / "flights_cleaned.csv", index=False)
    passengers_clean.to_csv(SILVER_DIR / "passengers_cleaned.csv", index=False)
    bookings_clean.to_csv(SILVER_DIR / "bookings_cleaned.csv", index=False)
    payments_clean.to_csv(SILVER_DIR / "payments_cleaned.csv", index=False)
    print(f"  -> Silver datasets generated in: {SILVER_DIR}")

    # ---------------------------------------------------------
    # 4. GOLD LAYER / STAR SCHEMA MODELING
    # ---------------------------------------------------------
    print("\n[Step 4/6] Constructing Star Schema dimensional models (Gold Layer)...")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    star_tables = build_star_schema(flights_clean, passengers_clean, bookings_clean, payments_clean)

    # Export Gold Star Schema CSVs
    for table_name, df_table in star_tables.items():
        csv_file = GOLD_DIR / f"{table_name}.csv"
        df_table.to_csv(csv_file, index=False)
        print(f"  -> Exported {table_name}.csv ({len(df_table)} rows)")

    # ---------------------------------------------------------
    # 5. ANOMALY REPORTING
    # ---------------------------------------------------------
    print("\n[Step 5/6] Generating Data Quality Anomalies Report...")
    all_anomalies = flight_anomalies + passenger_anomalies + booking_anomalies + payment_anomalies
    anomalies_df = generate_anomaly_report(all_anomalies, GOLD_DIR / "data_quality_anomalies_report.csv")
    print(f"  -> Generated data_quality_anomalies_report.csv ({len(anomalies_df)} anomalies logged)")

    # ---------------------------------------------------------
    # 6. KPI COMPUTATION
    # ---------------------------------------------------------
    print("\n[Step 6/6] Computing business and data quality KPIs...")
    kpis_dict = compute_all_kpis(
        star_tables["fact_flights"],
        star_tables["fact_bookings"],
        payments_clean,
        star_tables["dim_airline"],
        star_tables["dim_route"],
        anomalies_df,
    )

    # Export KPI tables
    kpis_dict["flight_kpis"].to_csv(GOLD_DIR / "flight_kpis.csv", index=False)
    kpis_dict["booking_kpis"].to_csv(GOLD_DIR / "booking_kpis.csv", index=False)
    kpis_dict["payment_kpis"].to_csv(GOLD_DIR / "payment_kpis.csv", index=False)
    kpis_dict["data_quality_kpis"].to_csv(GOLD_DIR / "data_quality_kpis.csv", index=False)
    kpis_dict["airline_distribution"].to_csv(GOLD_DIR / "airline_flight_distribution.csv", index=False)
    kpis_dict["route_traffic"].to_csv(GOLD_DIR / "route_flight_traffic.csv", index=False)
    kpis_dict["payment_method_breakdown"].to_csv(GOLD_DIR / "payment_method_breakdown.csv", index=False)
    print(f"  -> Exported all KPI CSVs to: {GOLD_DIR}")

    # ---------------------------------------------------------
    # EXECUTION SUMMARY
    # ---------------------------------------------------------
    fb = star_tables["fact_bookings"]
    realized_rev = fb["realized_revenue"].sum()
    gross_pay = fb["gross_payment_amount"].sum()
    conv_rate = (fb["is_confirmed"].sum() / len(fb)) * 100.0

    print("\n" + "=" * 70)
    print("                    PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Status                     : SUCCESS")
    print(f"  Operational Flights (Gold) : {len(star_tables['fact_flights']):,} (Deduplicated from {len(raw_flights)})")
    print(f"  Passengers Preserved (Gold): {len(passengers_clean):,} (Surrogate keys assigned)")
    print(f"  Bookings Grain (Gold)      : {len(fb):,} (Zero join fan-out guaranteed)")
    print(f"  Gross Processed Payments   : INR {gross_pay:,.2f}")
    print(f"  Realized Confirmed Revenue : INR {realized_rev:,.2f}")
    print(f"  Booking Conversion Rate    : {conv_rate:.2f}%")
    print(f"  Data Quality Anomalies     : {len(anomalies_df)} issues tracked & privacy-masked")
    print("=" * 70)

    return {
        "status": "SUCCESS",
        "fact_flights_count": len(star_tables["fact_flights"]),
        "fact_bookings_count": len(fb),
        "gross_processed_payments": gross_pay,
        "realized_revenue": realized_rev,
        "conversion_rate": conv_rate,
        "anomalies_count": len(anomalies_df),
    }


if __name__ == "__main__":
    run_pipeline()
