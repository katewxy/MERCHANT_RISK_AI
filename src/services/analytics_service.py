"""
analytics_service.py — Analytics Service Layer

The clean, typed interface between the scored transaction dataset and the
dashboard. Every function here is a thin, named delegate to risk_metrics.py —
the service layer exists to give the dashboard a stable, semantically clear API
that is decoupled from the internal aggregation implementation.

Rules:
  - The dashboard only imports from this module (never from risk_metrics directly).
  - This module never renders anything — it only returns data structures.
  - All heavy computation stays in risk_metrics.py.
"""

import pandas as pd
from src.risk import risk_metrics


# ── KPI snapshot ──────────────────────────────────────────────────────────────

def get_dashboard_snapshot(df: pd.DataFrame, threshold: float = 0.70) -> dict:
    """
    Scalar KPIs for the dashboard header row.

    Returns:
        {
          total_transactions : int,
          avg_risk           : float,
          fraud_rate         : float,
          high_risk_count    : int,
        }
    """
    return {
        "total_transactions": len(df),
        "avg_risk": risk_metrics.avg_risk(df),
        "fraud_rate": risk_metrics.fraud_rate(df),
        "high_risk_count": risk_metrics.high_risk_count(df, threshold),
    }


# ── Table feeds ───────────────────────────────────────────────────────────────

def get_high_risk_transactions(
    df: pd.DataFrame,
    threshold: float = 0.70,
) -> pd.DataFrame:
    """Flagged high-risk transactions for tabular display, sorted by score."""
    return risk_metrics.high_risk_transactions(df, threshold)


def get_merchant_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Per-merchant risk ranking table, sorted by avg_risk descending."""
    return risk_metrics.merchant_ranking(df)


# ── Chart feeds ───────────────────────────────────────────────────────────────

def get_daily_risk_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Day-level time series of avg_risk and fraud_count for trend chart."""
    return risk_metrics.daily_risk_trend(df)


def get_risk_score_series(df: pd.DataFrame) -> pd.DataFrame:
    """Single-column DataFrame of final_risk_score values for histogram."""
    return df[["final_risk_score"]].copy()


# ── AI insights ───────────────────────────────────────────────────────────────

def get_ai_insights(df: pd.DataFrame, threshold: float = 0.70) -> list[str]:
    """
    Compute the full metrics payload and pass it to the AI agent for
    natural language insight generation.

    Returns a list of insight strings ready for dashboard rendering.
    The import is local to avoid a circular dependency at module load time.
    """
    from src.ai.agent import generate_insights

    all_metrics = risk_metrics.compute_all(df, threshold)
    return generate_insights(all_metrics)
