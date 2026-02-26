"""
rule_engine.py — Rule-Based Risk Scoring

Applies domain-driven, interpretable rules to produce a rule_risk score
(range [0.0, 1.0]) per transaction. This component is fully independent
of the ML model — it provides an auditable, explainable risk signal.

Rules implemented:
  1. Transaction amount tiers    — high-value transactions are inherently riskier
  2. Customer velocity           — abnormally high per-customer transaction counts
  3. Off-hours activity          — transactions between midnight and 05:00

All three signals are additive; the result is clamped to [0.0, 1.0].

Thresholds are defined as module-level constants so they can be adjusted
without touching business logic.
"""

import pandas as pd
import numpy as np

# ── Thresholds ───────────────────────────────────────────────────────────────

AMOUNT_MODERATE: float = 200.0       # Low-level signal
AMOUNT_HIGH: float = 1_000.0         # Medium-level signal
AMOUNT_VERY_HIGH: float = 3_000.0    # High-level signal

# Customers with >= this many transactions are flagged for velocity risk
HIGH_VELOCITY_CUTOFF: int = 10

# Hour range (inclusive) considered off-hours: midnight to 5 AM
OFF_HOURS_START: int = 0
OFF_HOURS_END: int = 5


# ── Rule functions ───────────────────────────────────────────────────────────

def _score_amount(df: pd.DataFrame) -> pd.Series:
    """
    Tiered amount risk score.
      ≥ $3,000  → 0.6
      ≥ $1,000  → 0.4
      ≥ $200    → 0.2
      < $200    → 0.0
    """
    scores = pd.Series(0.0, index=df.index)
    scores[df["Amount"] >= AMOUNT_MODERATE] = 0.2
    scores[df["Amount"] >= AMOUNT_HIGH] = 0.4
    scores[df["Amount"] >= AMOUNT_VERY_HIGH] = 0.6
    return scores


def _score_customer_velocity(df: pd.DataFrame) -> pd.Series:
    """
    Flag customers with abnormally high transaction frequency.
    High velocity can indicate account takeover or carding attacks.
    """
    freq = df["customer_id"].map(df["customer_id"].value_counts())
    scores = pd.Series(0.0, index=df.index)
    scores[freq >= HIGH_VELOCITY_CUTOFF] = 0.2
    return scores


def _score_off_hours(df: pd.DataFrame) -> pd.Series:
    """
    Penalise transactions occurring between midnight and 05:00 local time.
    Requires 'transaction_time' column; silently returns zero if absent.
    """
    scores = pd.Series(0.0, index=df.index)
    if "transaction_time" in df.columns:
        hour = df["transaction_time"].dt.hour
        scores[(hour >= OFF_HOURS_START) & (hour <= OFF_HOURS_END)] = 0.2
    return scores


# ── Public API ───────────────────────────────────────────────────────────────

def apply_rules(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate all rule scores into a single rule_risk per transaction.

    Each rule contributes independently; scores are summed and clamped
    to [0.0, 1.0] so the result remains a valid probability-like input
    for the hybrid risk engine.

    Args:
        df: Enriched transaction DataFrame (must include 'Amount',
            'customer_id', and optionally 'transaction_time').

    Returns:
        pd.Series of float rule_risk values aligned with df's index.
    """
    combined = (
        _score_amount(df)
        + _score_customer_velocity(df)
        + _score_off_hours(df)
    )
    return combined.clip(0.0, 1.0)
