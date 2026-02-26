"""
pipeline.py — Orchestration Layer

The single entry point that assembles all pipeline stages into one
end-to-end execution path. Downstream consumers (analytics service,
dashboard, scripts) call only run() and receive a fully scored DataFrame.

Pipeline stages:
  1. Load      — read raw creditcard.csv from disk
  2. Govern    — validate schema, clean, deduplicate
  3. Enrich    — add merchant_id, customer_id, transaction_time
  4. Features  — build ML feature matrix
  5. Model     — train Logistic Regression, generate P(fraud)
  6. Rules     — apply rule-based risk scores
  7. Score     — combine into final_risk_score + risk_label

The data path is resolved relative to this file so the pipeline runs
correctly regardless of the working directory the process is started from.
"""

import pandas as pd
from pathlib import Path

from src.core import schema, governance, features
from src.models.fraud_model import train_and_score
from src.risk.rule_engine import apply_rules
from src.risk.risk_engine import score_transactions

# ── Path resolution ───────────────────────────────────────────────────────────
# __file__ is  MERCHANT_RISK_AI/src/services/pipeline.py
# .parent.parent.parent resolves to MERCHANT_RISK_AI/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = _PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


# ── Loader ────────────────────────────────────────────────────────────────────

def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Read the raw Kaggle credit card CSV from disk.

    Raises FileNotFoundError with a helpful message if the dataset has not
    been placed in the expected location.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n\n"
            "Download 'creditcard.csv' from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it at:  data/raw/creditcard.csv"
        )
    df = pd.read_csv(path)
    print(f"[Pipeline] Loaded {len(df):,} raw records from '{path.name}'.")
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def run(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Execute the full pipeline and return a scored DataFrame.

    This function is deterministic and stateless — safe to cache with
    @st.cache_data or any memoisation layer.

    Returns a DataFrame with all original Kaggle columns plus:
      merchant_id, customer_id, transaction_time  (from schema)
      rule_risk, ml_probability                   (intermediate scores)
      final_risk_score, risk_label                (hybrid output)
    """
    # Stage 1 — Load
    raw = load_raw(path)

    # Stage 2 — Govern
    cleaned = governance.clean(raw)

    # Stage 3 — Enrich
    enriched = schema.enrich_schema(cleaned)

    # Stage 4 — Features
    X, y = features.build_feature_matrix(enriched)

    # Stage 5 — Model
    _, ml_probs = train_and_score(X, y)

    # Stage 6 — Rules
    rule_risk = apply_rules(enriched)

    # Stage 7 — Hybrid score
    ml_probability = pd.Series(ml_probs, index=enriched.index)
    scored = score_transactions(enriched, rule_risk, ml_probability)

    print(f"[Pipeline] Complete — {len(scored):,} transactions scored and ready.")
    return scored
