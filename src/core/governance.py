"""
governance.py — Data Quality & Validation Layer

Enforces the data contract before any modelling or enrichment occurs.
This layer is intentionally strict: it raises on structural violations
and quietly drops malformed rows that would poison downstream metrics.

Responsibilities:
  - Schema validation (required columns present)
  - Null removal
  - Deduplication
  - Amount range filtering (business rule: 0 ≤ amount ≤ 50,000)
  - Label validation (Class ∈ {0, 1})
"""

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = (
    ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
)

# Upper ceiling for plausible card transaction amounts (USD)
AMOUNT_CEILING: float = 50_000.0


# ── Validation steps ─────────────────────────────────────────────────────────

def _validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is absent."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"[Governance] Schema violation — missing columns: {missing}"
        )


def _drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()
    removed = before - len(df)
    if removed:
        print(f"[Governance] Dropped {removed:,} rows containing null values.")
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        print(f"[Governance] Removed {removed:,} exact duplicate rows.")
    return df


def _filter_invalid_amounts(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[(df["Amount"] >= 0.0) & (df["Amount"] <= AMOUNT_CEILING)]
    removed = before - len(df)
    if removed:
        print(f"[Governance] Filtered {removed:,} rows with out-of-range Amount values.")
    return df


def _filter_invalid_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where Class is a known binary label."""
    return df[df["Class"].isin([0, 1])].copy()


# ── Public API ───────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full governance pipeline. Call this before any enrichment or modelling.

    Steps (in order):
      1. Schema validation  — raises on violation
      2. Null removal       — drops rows silently
      3. Deduplication      — drops rows silently
      4. Amount filter      — drops rows silently
      5. Label filter       — drops rows silently

    Returns a clean, validated DataFrame.
    """
    _validate_schema(df)
    df = _drop_nulls(df)
    df = _remove_duplicates(df)
    df = _filter_invalid_amounts(df)
    df = _filter_invalid_labels(df)
    print(f"[Governance] Validation complete — {len(df):,} clean records.")
    return df
