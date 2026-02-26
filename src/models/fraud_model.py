"""
fraud_model.py — ML Model Layer

Trains a Logistic Regression classifier for transaction fraud detection
and returns per-transaction fraud probability scores.

Design decisions:
  - class_weight="balanced" compensates for the severe class imbalance in card
    fraud data (~0.17% fraud rate). Without this, the model collapses to
    predicting all-legitimate and achieves misleadingly high accuracy.
  - solver="lbfgs" is efficient for medium-sized datasets with dense features.
  - We train on the full dataset here because this is a monitoring/scoring
    system, not a hold-out evaluation. In production, training would be a
    separate offline job with versioned model artefacts.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# ── Model factory ────────────────────────────────────────────────────────────

def build_model() -> LogisticRegression:
    """
    Return an untrained Logistic Regression estimator with production settings.

    class_weight="balanced" is the single most important parameter for this
    problem — it prevents the model from ignoring the minority fraud class.
    """
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )


# ── Training ─────────────────────────────────────────────────────────────────

def train(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """Fit the model on the provided feature matrix and return it."""
    model = build_model()
    model.fit(X, y)
    print(
        f"[FraudModel] Trained on {len(X):,} samples. "
        f"Fraud prevalence: {y.mean():.4%}"
    )
    return model


# ── Inference ────────────────────────────────────────────────────────────────

def predict_fraud_probability(
    model: LogisticRegression,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Return P(fraud) for each row in X.
    Output shape: (n_samples,), values in [0.0, 1.0].
    """
    return model.predict_proba(X)[:, 1]


# ── Convenience ──────────────────────────────────────────────────────────────

def train_and_score(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[LogisticRegression, np.ndarray]:
    """
    Train and immediately score the same dataset.

    Returns:
        model         — fitted LogisticRegression
        probabilities — P(fraud) array aligned with X's index
    """
    model = train(X, y)
    probabilities = predict_fraud_probability(model, X)
    return model, probabilities
