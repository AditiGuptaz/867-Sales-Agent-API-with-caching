"""
Step 1: Excel to SQLite Ingestion
867 Sales Data - Jan 2022 to July 2024 (Flat format)
Run this ONCE to load your Excel data into a SQLite database.
"""

import pandas as pd
import sqlite3
import os

EXCEL_PATH = "867_Data_Jan12022ToJuly272024.xlsx"
DB_PATH    = "sales867.db"


def short_product(dose: str) -> str:
    """Map item dose description to a clean short product name."""
    if pd.isna(dose):       return "Unknown"
    if "300" in str(dose):  return "AKYNZEO Oral"
    if "235" in str(dose):  return "AKYNZEO IV"
    return str(dose)


def ingest(excel_path: str = EXCEL_PATH, db_path: str = DB_PATH):
    print(f"Reading: {excel_path}")
    df = pd.read_excel(excel_path)

    # ── Clean column names ────────────────────────────────────────────────────
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("#", "_no")
        for c in df.columns
    ]

    # ── Parse dates + derive time columns ────────────────────────────────────
    df["day_date"]   = pd.to_datetime(df["day_date"])
    df["year"]       = df["day_date"].dt.year
    df["month"]      = df["day_date"].dt.month
    df["month_name"] = df["day_date"].dt.strftime("%B")
    df["quarter"]    = df["day_date"].dt.quarter
    df["week"]       = df["day_date"].dt.isocalendar().week.astype(int)

    # ── Friendly product name ─────────────────────────────────────────────────
    df["product"] = df["item_dose"].apply(short_product)

    # ── Write to SQLite ───────────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    df.to_sql("sales", conn, if_exists="replace", index=False)

    index_cols = [
        "year", "month", "quarter",
        "receiver_state", "distributor_state",
        "product", "transaction_type",
        "receiver_class_of_trade", "distributor_class_of_trade",
        "receiver_city", "receiver_zip",
    ]
    for col in index_cols:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON sales({col})")

    conn.commit()
    conn.close()

    print(f"\nRows loaded    : {len(df):,}")
    print(f"Date range     : {df['day_date'].min().date()} → {df['day_date'].max().date()}")
    print(f"Database saved : {db_path}")
    print(f"\nKey column summary:")
    print(f"  Products           : {df['product'].value_counts().to_dict()}")
    print(f"  Transaction types  : {df['transaction_type'].value_counts().to_dict()}")
    print(f"  Years covered      : {sorted(df['year'].unique())}")
    print(f"  States (receiver)  : {df['receiver_state'].nunique()} unique")
    print(f"  Class of trade     : {df['receiver_class_of_trade'].nunique()} unique types")
    print(f"\nAll columns:")
    for c in df.columns:
        print(f"  {c}")


if __name__ == "__main__":
    ingest()
