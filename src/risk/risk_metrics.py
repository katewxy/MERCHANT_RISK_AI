"""
risk_metrics.py — Metrics Computation Layer

All analytical aggregations live here. The analytics service and the AI agent
delegate every computation to this module — nothing upstream touches groupby,
agg, or filter logic directly.

Exported functions:
  avg_risk()                  — portfolio-wide mean risk score
  fraud_rate()                — proportion of Class == 1
  high_risk_count()           — count of transactions above threshold
  merchant_ranking()          — per-merchant aggregated risk table
  daily_risk_trend()          — day-level time series of avg risk + fraud count
  high_risk_transactions()    — filtered table of flagged transactions
  compute_all()               — convenience wrapper used by the AI agent
"""

import pandas as pd
import numpy as np
from typing import Any


# ── Scalar metrics ───────────────────────────────────────────────────────────

def avg_risk(df: pd.DataFrame) -> float:
    """Mean final_risk_score across all transactions in df."""
    return float(df["final_risk_score"].mean())


def fraud_rate(df: pd.DataFrame) -> float:
    """Proportion of transactions with Class == 1 (actual fraud label)."""
    return float(df["Class"].mean())


def high_risk_count(df: pd.DataFrame, threshold: float = 0.70) -> int:
    """Number of transactions where final_risk_score >= threshold."""
    return int((df["final_risk_score"] >= threshold).sum())


# ── Aggregated tables ────────────────────────────────────────────────────────

def merchant_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-merchant risk metrics.

    Columns returned:
      merchant_id, total_transactions, avg_risk, fraud_count,
      avg_amount, fraud_rate

    Sorted by avg_risk descending so the riskiest merchant is always row 0.
    """
    ranking = (
        df.groupby("merchant_id", as_index=False)
        .agg(
            total_transactions=("final_risk_score", "count"),
            avg_risk=("final_risk_score", "mean"),
            fraud_count=("Class", "sum"),
            avg_amount=("Amount", "mean"),
        )
        .sort_values("avg_risk", ascending=False)
        .reset_index(drop=True)
    )
    ranking["fraud_rate"] = (
        ranking["fraud_count"] / ranking["total_transactions"]
    ).round(4)
    return ranking.round({"avg_risk": 4, "avg_amount": 2})


def daily_risk_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day-level time series of average risk score and fraud count.

    Requires 'transaction_time' column (added by schema.py).
    Returns an empty DataFrame with correct columns if the column is absent.

    Columns returned:
      date, avg_risk, fraud_count, transaction_count
    """
    _empty = pd.DataFrame(
        columns=["date", "avg_risk", "fraud_count", "transaction_count"]
    )
    if "transaction_time" not in df.columns:
        return _empty

    temp = df.assign(date=df["transaction_time"].dt.date)
    trend = (
        temp.groupby("date", as_index=False)
        .agg(
            avg_risk=("final_risk_score", "mean"),
            fraud_count=("Class", "sum"),
            transaction_count=("final_risk_score", "count"),
        )
        .sort_values("date")
    )
    return trend.round({"avg_risk": 4})


def high_risk_transactions(
    df: pd.DataFrame,
    threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Return transactions where final_risk_score >= threshold.

    Sorted by final_risk_score descending. Column order is fixed for
    consistent display regardless of how the DataFrame was assembled.
    """
    display_cols = [
        "transaction_time",
        "merchant_id",
        "customer_id",
        "Amount",
        "rule_risk",
        "ml_probability",
        "final_risk_score",
        "risk_label",
        "Class",
    ]
    available = [c for c in display_cols if c in df.columns]
    return (
        df.loc[df["final_risk_score"] >= threshold, available]
        .sort_values("final_risk_score", ascending=False)
        .reset_index(drop=True)
    )


# ── Convenience wrapper ──────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame, threshold: float = 0.70) -> dict[str, Any]:
    """
    Compute the full metrics payload in one call.

    Used by the AI agent which needs all metrics simultaneously.
    Direct dashboard consumers should prefer the individual functions
    via analytics_service to avoid redundant computation.

    Returns:
        {
          avg_risk            : float,
          fraud_rate          : float,
          high_risk_count     : int,
          total_transactions  : int,
          merchant_ranking    : pd.DataFrame,
          daily_risk_trend    : pd.DataFrame,
          high_risk_transactions : pd.DataFrame,
        }
    """
    return {
        "avg_risk": avg_risk(df),
        "fraud_rate": fraud_rate(df),
        "high_risk_count": high_risk_count(df, threshold),
        "total_transactions": len(df),
        "merchant_ranking": merchant_ranking(df),
        "daily_risk_trend": daily_risk_trend(df),
        "high_risk_transactions": high_risk_transactions(df, threshold),
    }
