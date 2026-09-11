"""Dimensional modeling module for the Airlines Data Pipeline.

Constructs conformed Star Schema tables:
- Dim_Airline
- Dim_Route
- Dim_Passenger
- Dim_Flight
- Fact_Flights
- Fact_Bookings

Strictly preserves fact grain (zero join fan-out) by using surrogate primary keys
and explicit unresolved members for ambiguous relationships.
"""

from typing import Dict, Any, Tuple
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np 

from src.config import AIRLINE_PREFIX_MAP, UNRESOLVED_KEY, UNRESOLVED_CODE


def build_star_schema(
    flights_clean: pd.DataFrame,
    passengers_clean: pd.DataFrame,
    bookings_clean: pd.DataFrame,
    payments_clean: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Constructs dimensional star schema tables from cleaned datasets.

    Args:
        flights_clean: Cleaned flights DataFrame.
        passengers_clean: Cleaned passengers DataFrame.
        bookings_clean: Cleaned bookings DataFrame.
        payments_clean: Cleaned payments DataFrame.

    Returns:
        Dictionary mapping table name to DataFrame:
        - 'dim_airline'
        - 'dim_route'
        - 'dim_passenger'
        - 'dim_flight'
        - 'fact_flights'
        - 'fact_bookings'
    """
    # ---------------------------------------------------------
    # 1. Dim_Airline
    # ---------------------------------------------------------
    airline_records = [
        {"airline_key": UNRESOLVED_KEY, "airline_code": UNRESOLVED_CODE, "airline_name": "Unresolved Airline"}
    ]
    airline_code_to_key = {UNRESOLVED_CODE: UNRESOLVED_KEY}
    airline_name_to_key = {"Unresolved Airline": UNRESOLVED_KEY}

    key_counter = 1
    for prefix, name in AIRLINE_PREFIX_MAP.items():
        airline_records.append({
            "airline_key": key_counter,
            "airline_code": prefix,
            "airline_name": name,
        })
        airline_code_to_key[prefix] = key_counter
        airline_name_to_key[name] = key_counter
        key_counter += 1

    dim_airline = pd.DataFrame(airline_records)

    # ---------------------------------------------------------
    # 2. Dim_Route
    # ---------------------------------------------------------
    # Unique routes from cleaned flights
    route_pairs = (
        flights_clean[["source", "destination"]]
        .drop_duplicates()
        .sort_values(["source", "destination"])
        .reset_index(drop=True)
    )

    route_records = [
        {
            "route_key": UNRESOLVED_KEY,
            "source_airport": UNRESOLVED_CODE,
            "destination_airport": UNRESOLVED_CODE,
            "route_code": UNRESOLVED_CODE,
        }
    ]
    route_pair_to_key = {}

    route_key_counter = 1
    for _, r in route_pairs.iterrows():
        src = r["source"]
        dst = r["destination"]
        code = f"{src}-{dst}"
        route_records.append({
            "route_key": route_key_counter,
            "source_airport": src,
            "destination_airport": dst,
            "route_code": code,
        })
        route_pair_to_key[(src, dst)] = route_key_counter
        route_key_counter += 1

    dim_route = pd.DataFrame(route_records)

    # ---------------------------------------------------------
    # 3. Dim_Flight
    # ---------------------------------------------------------
    # Link flights to airline_key and route_key
    flights_df = flights_clean.copy()
    flights_df["airline_key"] = flights_df["airline"].map(airline_name_to_key).fillna(UNRESOLVED_KEY).astype(int)
    flights_df["route_key"] = [
        route_pair_to_key.get((s, d), UNRESOLVED_KEY)
        for s, d in zip(flights_df["source"], flights_df["destination"])
    ]

    flight_cols = [
        "flight_key",
        "flight_id",
        "airline_key",
        "route_key",
        "departure_time",
        "arrival_time",
        "duration_minutes",
        "is_overnight",
        "is_duration_outlier",
        "is_multi_leg_collision",
    ]
    dim_flight_rows = flights_df[flight_cols].copy()

    # Add Unresolved member record
    unresolved_flight = pd.DataFrame([{
        "flight_key": UNRESOLVED_KEY,
        "flight_id": UNRESOLVED_CODE,
        "airline_key": UNRESOLVED_KEY,
        "route_key": UNRESOLVED_KEY,
        "departure_time": pd.NaT,
        "arrival_time": pd.NaT,
        "duration_minutes": 0.0,
        "is_overnight": False,
        "is_duration_outlier": False,
        "is_multi_leg_collision": True,
    }])
    dim_flight = pd.concat([unresolved_flight, dim_flight_rows], ignore_index=True)

    # ---------------------------------------------------------
    # 4. Dim_Passenger
    # ---------------------------------------------------------
    pass_cols = [
        "passenger_key",
        "passenger_id",
        "first_name",
        "last_name",
        "age",
        "gender",
        "masked_email",
        "masked_phone",
        "masked_aadhaar",
        "is_collision",
    ]
    dim_pass_rows = passengers_clean[pass_cols].copy()

    unresolved_pass = pd.DataFrame([{
        "passenger_key": UNRESOLVED_KEY,
        "passenger_id": UNRESOLVED_CODE,
        "first_name": "Unresolved",
        "last_name": "Passenger",
        "age": 0,
        "gender": "UNKNOWN",
        "masked_email": "unresolved@airline.internal",
        "masked_phone": "+91-XXXXXXXXXX",
        "masked_aadhaar": "XXXX-XXXX-XXXX",
        "is_collision": True,
    }])
    dim_passenger = pd.concat([unresolved_pass, dim_pass_rows], ignore_index=True)

    # ---------------------------------------------------------
    # 5. Fact_Flights
    # ---------------------------------------------------------
    fact_flights = flights_df[[
        "flight_key",
        "flight_id",
        "airline_key",
        "route_key",
        "departure_time",
        "arrival_time",
        "duration_minutes",
        "is_overnight",
        "is_duration_outlier",
    ]].copy()
    fact_flights["departure_date"] = fact_flights["departure_time"].dt.date
    fact_flights["is_overnight"] = fact_flights["is_overnight"].astype(int)
    fact_flights["is_duration_outlier"] = fact_flights["is_duration_outlier"].astype(int)

    # ---------------------------------------------------------
    # 6. Fact_Bookings
    # ---------------------------------------------------------
    # Mapping passenger_key:
    # If passenger_id is non-colliding (1 unique record), map to its passenger_key.
    # If passenger_id is colliding, map to UNRESOLVED_KEY (-1) to avoid arbitrary choice and fan-out.
    non_colliding_passengers = passengers_clean[~passengers_clean["is_collision"]]
    pass_id_to_key = dict(zip(non_colliding_passengers["passenger_id"], non_colliding_passengers["passenger_key"]))

    # Mapping flight_key:
    # If flight_id is non-colliding (1 unique operational record), map to its flight_key.
    # If flight_id has multiple operational legs (e.g. 6F250), map to UNRESOLVED_KEY (-1).
    non_colliding_flights = flights_df[~flights_df["is_multi_leg_collision"]]
    flight_id_to_key = dict(zip(non_colliding_flights["flight_id"], non_colliding_flights["flight_key"]))

    # Map flight_key to route_key for resolved flights
    flight_key_to_route_key = dict(zip(flights_df["flight_key"], flights_df["route_key"]))

    # Aggregate payments to booking grain
    # Distinguish Gross Processed Payments from Realized Confirmed Revenue
    valid_payments = payments_clean[payments_clean["is_valid_amount"]].copy()

    pay_total_agg = payments_clean.groupby("booking_id").agg(
        payment_record_count=("payment_id", "count")
    ).reset_index()

    pay_valid_agg = valid_payments.groupby("booking_id").agg(
        valid_payment_count=("payment_id", "count"),
        gross_payment_amount=("amount_cleaned", "sum"),
    ).reset_index()

    confirmed_payments = valid_payments[valid_payments["is_realized_revenue"]]
    pay_realized_agg = confirmed_payments.groupby("booking_id").agg(
        realized_revenue=("amount_cleaned", "sum")
    ).reset_index()

    b_df = bookings_clean.copy()
    b_df = b_df.merge(pay_total_agg, on="booking_id", how="left")
    b_df = b_df.merge(pay_valid_agg, on="booking_id", how="left")
    b_df = b_df.merge(pay_realized_agg, on="booking_id", how="left")

    b_df["payment_record_count"] = b_df["payment_record_count"].fillna(0).astype(int)
    b_df["valid_payment_count"] = b_df["valid_payment_count"].fillna(0).astype(int)
    b_df["gross_payment_amount"] = b_df["gross_payment_amount"].fillna(0.0).round(2)
    b_df["realized_revenue"] = b_df["realized_revenue"].fillna(0.0).round(2)
    b_df["has_valid_payment"] = (b_df["valid_payment_count"] > 0).astype(int)

    # Dimensional Keys Assignment
    b_df["passenger_key"] = b_df["passenger_id"].map(pass_id_to_key).fillna(UNRESOLVED_KEY).astype(int)
    b_df["flight_key"] = b_df["flight_id"].map(flight_id_to_key).fillna(UNRESOLVED_KEY).astype(int)

    # Route key: if flight_key resolved, use route_key of flight; otherwise UNRESOLVED_KEY
    b_df["route_key"] = b_df["flight_key"].map(flight_key_to_route_key).fillna(UNRESOLVED_KEY).astype(int)

    # Airline key: derived from flight_id prefix (6F, AI, SJ, UK) which is known even if leg is multi-leg
    b_df["flight_prefix"] = b_df["flight_id"].astype(str).str[:2]
    b_df["airline_key"] = b_df["flight_prefix"].map(airline_code_to_key).fillna(UNRESOLVED_KEY).astype(int)

    # Status indicator flags
    b_df["is_confirmed"] = (b_df["status"] == "CONFIRMED").astype(int)
    b_df["is_cancelled"] = (b_df["status"] == "CANCELLED").astype(int)
    b_df["is_pending"] = (b_df["status"] == "PENDING").astype(int)
    b_df["is_invalid"] = (b_df["status"] == "INVALID").astype(int)
    b_df["is_unknown"] = (b_df["status"] == "UNKNOWN").astype(int)

    b_df["is_unresolved_passenger"] = (b_df["passenger_key"] == UNRESOLVED_KEY).astype(int)
    b_df["is_unresolved_flight"] = (b_df["flight_key"] == UNRESOLVED_KEY).astype(int)

    fact_bookings_cols = [
        "booking_id",
        "booking_date",
        "flight_key",
        "passenger_key",
        "airline_key",
        "route_key",
        "passenger_id",
        "flight_id",
        "status",
        "is_confirmed",
        "is_cancelled",
        "is_pending",
        "is_invalid",
        "is_unknown",
        "is_unresolved_flight",
        "is_unresolved_passenger",
        "gross_payment_amount",
        "realized_revenue",
        "payment_record_count",
        "valid_payment_count",
        "has_valid_payment",
    ]
    fact_bookings = b_df[fact_bookings_cols].copy()
    fact_bookings.rename(columns={
        "status": "booking_status",
        "passenger_id": "raw_passenger_id",
        "flight_id": "raw_flight_id",
    }, inplace=True)

    return {
        "dim_airline": dim_airline,
        "dim_route": dim_route,
        "dim_flight": dim_flight,
        "dim_passenger": dim_passenger,
        "fact_flights": fact_flights,
        "fact_bookings": fact_bookings,
    }
