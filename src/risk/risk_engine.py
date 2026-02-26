"""
risk_engine.py — Hybrid Risk Scoring Engine

Combines the rule-based risk score and the ML fraud probability into a
single, interpretable final_risk_score per transaction.

Weighting rationale:
  ML probability  (65%) — carries the bulk of the signal. Logistic regression
                          on PCA features captures latent fraud patterns that
                          rules cannot express.
  Rule risk       (35%) — provides domain-driven amplification and a meaningful
                          floor for high-value or off-hours transactions even
                          when the model is uncertain.

Both inputs are in [0, 1]. The weighted sum is therefore naturally in [0, 1]
without requiring clipping.

Risk labels:
  HIGH    ≥ 0.70   — immediate review recommended
  MEDIUM  ≥ 0.40   — elevated; monitor proactively
  LOW     < 0.40   — within normal bounds
"""

import pandas as pd
import numpy as np

# ── Weights ──────────────────────────────────────────────────────────────────

RULE_WEIGHT: float = 0.35
ML_WEIGHT: float = 0.65

# ── Risk tier thresholds ─────────────────────────────────────────────────────

THRESHOLD_HIGH: float = 0.70
THRESHOLD_MEDIUM: float = 0.40


# ── Scoring logic ────────────────────────────────────────────────────────────

def _compute_hybrid_score(
    rule_risk: pd.Series,
    ml_probability: pd.Series,
) -> pd.Series:
    """
    Weighted linear combination of rule risk and ML probability.
    Result is in [0.0, 1.0] by construction (clip is a safety guard only).
    """
    return (RULE_WEIGHT * rule_risk + ML_WEIGHT * ml_probability).clip(0.0, 1.0)


def _assign_risk_label(score: pd.Series) -> pd.Series:
    """
    Map a continuous risk score to a categorical risk tier string.
    Uses np.select for a clean, vectorised conditional.
    """
    conditions = [
        score >= THRESHOLD_HIGH,
        score >= THRESHOLD_MEDIUM,
    ]
    choices = ["HIGH", "MEDIUM"]
    return pd.Series(
        np.select(conditions, choices, default="LOW"),
        index=score.index,
        dtype="object",
    )


# ── Public API ───────────────────────────────────────────────────────────────

def score_transactions(
    df: pd.DataFrame,
    rule_risk: pd.Series,
    ml_probability: pd.Series,
) -> pd.DataFrame:
    """
    Attach all risk columns to the transaction DataFrame and return a copy.

    Added columns:
      rule_risk         — output of the rule engine  [0, 1]
      ml_probability    — P(fraud) from the ML model [0, 1]
      final_risk_score  — hybrid weighted combination [0, 1]
      risk_label        — categorical tier: HIGH / MEDIUM / LOW

    Args:
        df             : Enriched transaction DataFrame.
        rule_risk      : Rule engine scores aligned with df's index.
        ml_probability : Model fraud probabilities aligned with df's index.

    Returns:
        A new DataFrame with all original columns plus the four risk columns.
    """
    df = df.copy()
    df["rule_risk"] = rule_risk.values
    df["ml_probability"] = ml_probability.values
    df["final_risk_score"] = _compute_hybrid_score(rule_risk, ml_probability).values
    df["risk_label"] = _assign_risk_label(df["final_risk_score"]).values
    return df
