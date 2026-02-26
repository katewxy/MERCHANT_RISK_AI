"""
schema.py — Data Enrichment Layer

Adds synthetic merchant_id (20 merchants) and customer_id (200 customers)
to the raw Kaggle dataset, and converts the raw 'Time' column (seconds since
first transaction) into a real datetime for time-series analysis.

This module has no knowledge of ML models or risk logic — it purely
normalises and enriches the raw data contract.
"""

import pandas as pd
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

NUM_MERCHANTS = 20
NUM_CUSTOMERS = 200
RANDOM_SEED = 42

# Fixed reference point for converting relative seconds → wall-clock time
BASE_DATETIME = pd.Timestamp("2024-01-01 00:00:00")


# ── Internal helpers ─────────────────────────────────────────────────────────

def _assign_merchant_ids(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Randomly assign one of NUM_MERCHANTS synthetic merchant identifiers."""
    merchants = [f"MRC_{str(i).zfill(3)}" for i in range(1, NUM_MERCHANTS + 1)]
    df["merchant_id"] = rng.choice(merchants, size=len(df))
    return df


def _assign_customer_ids(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Randomly assign one of NUM_CUSTOMERS synthetic customer identifiers."""
    customers = [f"CUST_{str(i).zfill(4)}" for i in range(1, NUM_CUSTOMERS + 1)]
    df["customer_id"] = rng.choice(customers, size=len(df))
    return df


def _add_transaction_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw 'Time' (seconds offset) to a Pandas Timestamp column."""
    df["transaction_time"] = BASE_DATETIME + pd.to_timedelta(df["Time"], unit="s")
    return df


# ── Public API ───────────────────────────────────────────────────────────────

def enrich_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main enrichment entry point.

    Accepts a clean, validated DataFrame and returns a copy enriched with:
      - merchant_id  : one of 20 synthetic merchant identifiers
      - customer_id  : one of 200 synthetic customer identifiers
      - transaction_time : wall-clock Timestamp derived from 'Time'

    Assignments are seeded for reproducibility.
    """
    df = df.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    df = _assign_merchant_ids(df, rng)
    df = _assign_customer_ids(df, rng)
    df = _add_transaction_time(df)
    return df
