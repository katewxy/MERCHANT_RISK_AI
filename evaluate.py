import sys
import pathlib

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from src.core.features import build_feature_matrix
from src.models.fraud_model import train as lr_train
from src.models.xgb_model import train as xgb_train, predict_fraud_probability


# ── Load & split ──────────────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "raw" / "creditcard.csv")
print(f"Loaded {len(df):,} rows  |  Fraud rate: {df['Class'].mean():.4%}\n")

X, y = build_feature_matrix(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")
print("=" * 60)


# ── Train both models ─────────────────────────────────────────────────────────
lr_model  = lr_train(X_train, y_train)
xgb_model = xgb_train(X_train, y_train)

lr_prob  = lr_model.predict_proba(X_test)[:, 1]
xgb_prob = predict_fraud_probability(xgb_model, X_test)


# ── Threshold tuning: max Precision s.t. Recall >= 0.90 ──────────────────────
def best_threshold(probs, y_true, min_recall=0.90):
    """Return threshold that maximises Precision while keeping Recall >= min_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    # precision_recall_curve appends a trailing point with no threshold
    mask = recall[:-1] >= min_recall
    if not mask.any():
        return 0.5
    best_idx = np.argmax(precision[:-1][mask])
    return float(thresholds[mask][best_idx])

xgb_threshold = best_threshold(xgb_prob, y_test, min_recall=0.90)
lr_threshold  = best_threshold(lr_prob,  y_test, min_recall=0.90)

lr_pred  = (lr_prob  >= lr_threshold).astype(int)
xgb_pred = (xgb_prob >= xgb_threshold).astype(int)


# ── Comparison table ──────────────────────────────────────────────────────────
def metrics(y_true, y_pred, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    report = classification_report(y_true, y_pred, output_dict=True)
    fraud = report["1"]
    return dict(
        AUC=auc,
        Precision=fraud["precision"],
        Recall=fraud["recall"],
        F1=fraud["f1-score"],
    )

lr_m  = metrics(y_test, lr_pred,  lr_prob)
xgb_m = metrics(y_test, xgb_pred, xgb_prob)

print("\n" + "=" * 60)
print("MODEL COMPARISON  (threshold tuned: Recall >= 90%)")
print("=" * 60)
header = f"{'Metric':<14} {'LogisticReg':>14} {'XGBoost':>14}"
print(header)
print("-" * 44)
for key in ["AUC", "Precision", "Recall", "F1"]:
    print(f"{key:<14} {lr_m[key]:>14.4f} {xgb_m[key]:>14.4f}")
print(f"\n  LR  decision threshold : {lr_threshold:.4f}")
print(f"  XGB decision threshold : {xgb_threshold:.4f}")

print("\n" + "─" * 44)
print("Logistic Regression — full classification report:")
print(classification_report(y_test, lr_pred,
      target_names=["Legitimate", "Fraud"], digits=4))

print("─" * 44)
print("XGBoost — full classification report:")
print(classification_report(y_test, xgb_pred,
      target_names=["Legitimate", "Fraud"], digits=4))


# ── Confusion matrices ────────────────────────────────────────────────────────
def print_cm(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix — {name}")
    print(f"              Pred Legit  Pred Fraud")
    print(f"Actual Legit  {cm[0,0]:>10,}  {cm[0,1]:>10,}")
    print(f"Actual Fraud  {cm[1,0]:>10,}  {cm[1,1]:>10,}\n")

print_cm("Logistic Regression", y_test, lr_pred)
print_cm("XGBoost",             y_test, xgb_pred)


# ── Feature importance plot ───────────────────────────────────────────────────
importances = xgb_model.feature_importances_
feature_names = X.columns.tolist()
top_n = 10

idx  = np.argsort(importances)[::-1][:top_n]
top_names  = [feature_names[i] for i in idx]
top_scores = importances[idx]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(top_names[::-1], top_scores[::-1], color="#2196F3")
ax.set_xlabel("Feature Importance (gain)")
ax.set_title("XGBoost — Top 10 Feature Importances")
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
plt.tight_layout()
plot_path = ROOT / "xgb_feature_importance.png"
plt.savefig(plot_path, dpi=150)
print(f"Feature importance plot saved → {plot_path}")

print("\nTop 10 features (XGBoost):")
for rank, (name, score) in enumerate(zip(top_names, top_scores), 1):
    print(f"  {rank:>2}. {name:<18} {score:.4f}")
