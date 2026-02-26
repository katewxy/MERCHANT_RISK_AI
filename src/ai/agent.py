"""
agent.py — AI Insight Generation Agent

Analyses computed risk metrics and produces human-readable, actionable
observations for display in the Risk Control Center dashboard.

This agent uses rule-based natural language generation — no external
LLM API is required, so it runs fully offline. The interface is designed
to be drop-in replaceable with an LLM backend (e.g. Claude API) without
changing the analytics_service or dashboard contracts.

Public interface:
  generate_insights(metrics: dict) -> list[str]

Each returned string begins with a severity prefix that the dashboard
uses to choose the appropriate alert component:
  "ALERT:"   → st.error  (red)
  "WARNING:" → st.warning (orange)
  (none)     → st.info   (blue)
"""

import pandas as pd
from typing import Any


# ── Formatters ────────────────────────────────────────────────────────────────

def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _score(value: float) -> str:
    return f"{value:.4f}"


# ── Individual insight generators ─────────────────────────────────────────────

def _insight_fraud_rate(fraud_rate: float) -> str:
    if fraud_rate > 0.05:
        return (
            f"ALERT: Fraud rate is critically elevated at {_pct(fraud_rate)}. "
            "Immediate escalation and manual case review are recommended."
        )
    if fraud_rate > 0.01:
        return (
            f"WARNING: Fraud rate is above normal at {_pct(fraud_rate)}. "
            "Increase monitoring frequency and review flagged merchants."
        )
    return (
        f"Fraud rate is within acceptable bounds at {_pct(fraud_rate)}. "
        "No systemic fraud pattern detected."
    )


def _insight_avg_risk(avg_risk: float) -> str:
    if avg_risk > 0.70:
        return (
            f"ALERT: Portfolio-wide average risk score is HIGH ({_score(avg_risk)}). "
            "Review top-ranked merchants and consider tightening transaction limits."
        )
    if avg_risk > 0.40:
        return (
            f"WARNING: Portfolio average risk score is MODERATE ({_score(avg_risk)}). "
            "Monitor borderline merchants proactively."
        )
    return (
        f"Portfolio average risk score is LOW ({_score(avg_risk)}). "
        "System operating within normal risk parameters."
    )


def _insight_top_merchant(merchant_df: pd.DataFrame) -> str:
    if merchant_df.empty:
        return "No merchant data available for analysis."

    top = merchant_df.iloc[0]
    return (
        f"Highest-risk merchant: {top['merchant_id']} — "
        f"avg risk score {_score(top['avg_risk'])}, "
        f"{int(top['total_transactions']):,} transactions, "
        f"fraud rate {_pct(float(top['fraud_rate']))}."
    )


def _insight_risk_trend(trend_df: pd.DataFrame) -> str:
    if trend_df.empty or len(trend_df) < 2:
        return "Insufficient time-series data for trend analysis."

    # Compare the most recent period against the earliest to detect drift
    recent_avg = trend_df.tail(3)["avg_risk"].mean()
    baseline_avg = trend_df.head(3)["avg_risk"].mean()

    if recent_avg > baseline_avg * 1.10:
        return (
            f"WARNING: Risk trend is INCREASING. "
            f"Recent avg: {_score(recent_avg)} vs baseline: {_score(baseline_avg)}. "
            "Investigate recent activity spikes and new merchant patterns."
        )
    if recent_avg < baseline_avg * 0.90:
        return (
            f"Risk trend is DECREASING. "
            f"Recent avg: {_score(recent_avg)} vs baseline: {_score(baseline_avg)}. "
            "Risk controls appear to be working effectively."
        )
    return (
        f"Risk trend is STABLE. "
        f"Recent avg: {_score(recent_avg)} is consistent with baseline: {_score(baseline_avg)}."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_insights(metrics: dict[str, Any]) -> list[str]:
    """
    Main agent entry point.

    Accepts the full metrics dictionary produced by risk_metrics.compute_all()
    and returns a list of insight strings ordered by analytical priority.

    Args:
        metrics: Output of risk_metrics.compute_all()

    Returns:
        List of 4 insight strings with optional severity prefixes.
    """
    return [
        _insight_fraud_rate(metrics.get("fraud_rate", 0.0)),
        _insight_avg_risk(metrics.get("avg_risk", 0.0)),
        _insight_top_merchant(metrics.get("merchant_ranking", pd.DataFrame())),
        _insight_risk_trend(metrics.get("daily_risk_trend", pd.DataFrame())),
    ]
