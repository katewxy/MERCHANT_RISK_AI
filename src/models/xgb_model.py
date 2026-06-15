"""
xgb_model.py — XGBoost Fraud Detection Model

Uses scale_pos_weight to handle the severe class imbalance (~0.17% fraud rate)
without resampling. scale_pos_weight = neg_count / pos_count tells XGBoost to
penalise false negatives on the minority class proportionally.
"""

import numpy as np
import pandas as pd
import xgboost as xgb


def build_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )


def train(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """Fit XGBoost on training data. Derives scale_pos_weight from y."""
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = neg / pos
    model = build_model(scale_pos_weight=spw)
    model.fit(X, y)
    print(
        f"[XGBModel] Trained on {len(X):,} samples. "
        f"Fraud prevalence: {y.mean():.4%}  |  scale_pos_weight: {spw:.1f}"
    )
    return model


def predict_fraud_probability(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return P(fraud) for each row in X."""
    return model.predict_proba(X)[:, 1]
