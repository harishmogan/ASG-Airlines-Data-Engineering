# ASG Airlines - Data Engineering and Analytics
## Project Overview

This project is about processing airline data and presenting useful information through a Power BI dashboard.
The data starts from an Excel file. It is checked, cleaned and converted into analysis-ready data before creating the final dashboard.

## Tools Used
- Python
- Pandas
- Power BI
- Excel
- CSV

## Project Flow

Raw Data  
↓  
Data Validation  
↓  
Data Cleaning  
↓  
Data Modelling  
↓  
KPI Calculation  
↓  
Power BI Dashboard

## Data Cleaning

The project handles:

- Missing values
- Duplicate records
- Invalid booking statuses
- Invalid payment amounts
- Incorrect timestamps
- Flight duration issues
- Ambiguous passenger and flight references

Passenger information is also protected by masking personal identifiers such as Aadhaar numbers, phone numbers and email addresses.

## Flight Analysis

Flight duration is calculated using the departure and arrival timestamps.
Flights that arrive on the next day are identified as overnight flights. Unusual flight durations are flagged as outliers for further checking.

## Main KPIs

The dashboard includes:

- Total Flights
- Average Flight Duration
- Overnight Flights
- Total Bookings
- Confirmed Bookings
- Booking Conversion Rate
- Processed Payments
- Realized Revenue
- Data Quality Anomalies

## Power BI Dashboard

The report contains four pages:

1. Executive Overview
2. Flight Analysis
3. Bookings & Revenue
4. Data Quality

The dashboard includes filters and charts for airlines, routes, booking status, payments and data quality.

## Project Structure

```text
ASG-Airlines-Data-Engineering
│
├── ASG_Airlines_Data_Engineering.ipynb
├── Airlines_Analytics_Dashboard.pbix
├── ASG_Airlines_Simple_Documentation.pdf
│
├── src
│   ├── ingestion
│   ├── validation
│   ├── cleaning
│   ├── modeling
│   ├── analytics
│   └── pipeline.py
│
└── data
    └── cleaned_data
