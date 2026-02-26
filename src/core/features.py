"""
features.py — Feature Engineering Layer

Constructs the ML feature matrix from the cleaned, enriched DataFrame.

The Kaggle dataset provides PCA-transformed features V1–V28 which are already
centred and normalised. The raw 'Amount' column spans multiple orders of
magnitude and must be standardised independently to avoid dominating the
logistic regression coefficients.

'Time' is excluded: the timestamp information is captured in 'transaction_time'
and used by the rule engine, not the ML model.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Constants ────────────────────────────────────────────────────────────────

V_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
TARGET_COL: str = "Class"


# ── Public API ───────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the supervised learning inputs from a clean enriched DataFrame.

    Returns:
        X  — feature DataFrame with columns V1–V28 + Amount_scaled
        y  — binary label Series (0 = legitimate, 1 = fraud)

    Amount is fit-transformed on the full dataset passed in. In a production
    system the scaler would be serialised and applied separately to held-out
    data; here we score the same population we train on, so this is acceptable.
    """
    df = df.copy()

    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

    feature_cols = V_FEATURES + ["Amount_scaled"]
    X = df[feature_cols]
    y = df[TARGET_COL]

    return X, y


def get_feature_names() -> list[str]:
    """Return the ordered list of feature column names used by the model."""
    return V_FEATURES + ["Amount_scaled"]
