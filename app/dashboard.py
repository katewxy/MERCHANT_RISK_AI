"""
dashboard.py — Streamlit Risk Control Center

Presentation layer only. This file contains zero business logic,
zero pandas aggregations, and zero direct model calls.

All data flows through analytics_service, which in turn delegates
to risk_metrics. The dashboard's job is layout, formatting, and
user interaction — nothing else.

Run from MERCHANT_RISK_AI/:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of launch directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.services.pipeline import run as run_pipeline
from src.services import analytics_service

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Risk Control Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS ────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    div[data-testid="metric-container"] {
        background-color: #1a1a2e;
        border: 1px solid #2d2d44;
        border-radius: 10px;
        padding: 18px 22px;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.78rem;
        color: #a0a0b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running pipeline — training model and scoring transactions...")
def load_scored_data() -> pd.DataFrame:
    """Load and cache the fully scored transaction dataset."""
    return run_pipeline()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> tuple[list[str], float]:
    """
    Render the sidebar controls and return the active filter state.

    Returns:
        selected_merchants : list of merchant IDs to include
        threshold          : high-risk score threshold (float)
    """
    with st.sidebar:
        st.title("🛡️ Risk Control Center")
        st.markdown("**Hybrid AI Risk Monitoring**")
        st.divider()

        all_merchants = sorted(df["merchant_id"].unique().tolist())
        selected = st.multiselect(
            "Filter by Merchant",
            options=all_merchants,
            default=all_merchants,
            help="Scope the dashboard to one or more specific merchants.",
        )
        # Fall back to all merchants if user clears the selection
        if not selected:
            selected = all_merchants

        threshold = st.slider(
            "High-Risk Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            help=(
                "Transactions with final_risk_score ≥ this value are "
                "classified as HIGH RISK and appear in the flagged table."
            ),
        )

        st.divider()
        st.caption("Model:    Logistic Regression (class_weight=balanced)")
        st.caption("Scoring:  Hybrid — Rule 35% · ML 65%")
        st.caption("Dataset:  Kaggle Credit Card Fraud")

    return selected, threshold


# ── KPI row ───────────────────────────────────────────────────────────────────

def render_kpi_row(snapshot: dict) -> None:
    """Render the four top-level KPI metric cards."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Transactions",
            value=f"{snapshot['total_transactions']:,}",
        )
    with c2:
        st.metric(
            label="Avg Risk Score",
            value=f"{snapshot['avg_risk']:.4f}",
        )
    with c3:
        st.metric(
            label="Fraud Rate (ground truth)",
            value=f"{snapshot['fraud_rate'] * 100:.3f}%",
        )
    with c4:
        st.metric(
            label="High-Risk Transactions",
            value=f"{snapshot['high_risk_count']:,}",
        )


# ── Risk score histogram ──────────────────────────────────────────────────────

def render_risk_histogram(df: pd.DataFrame) -> None:
    """Plotly histogram of final_risk_score distribution."""
    score_df = analytics_service.get_risk_score_series(df)
    fig = px.histogram(
        score_df,
        x="final_risk_score",
        nbins=60,
        labels={
            "final_risk_score": "Final Risk Score",
            "count": "Transaction Count",
        },
        color_discrete_sequence=["#E63946"],
    )
    fig.update_layout(
        bargap=0.04,
        xaxis_title="Final Risk Score",
        yaxis_title="Transaction Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Daily trend chart ─────────────────────────────────────────────────────────

def render_daily_trend(df: pd.DataFrame) -> None:
    """
    Dual-axis Plotly chart: avg_risk (line) + fraud_count (bar).
    The Kaggle dataset spans ~48 hours so this shows 2–3 data points,
    which is still meaningful for trend direction.
    """
    trend = analytics_service.get_daily_risk_trend(df)

    if trend.empty:
        st.info("No time-series data available for the current selection.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend["date"].astype(str),
        y=trend["avg_risk"],
        mode="lines+markers",
        name="Avg Risk Score",
        line=dict(color="#E63946", width=2.5),
        marker=dict(size=7),
        yaxis="y1",
    ))

    fig.add_trace(go.Bar(
        x=trend["date"].astype(str),
        y=trend["fraud_count"],
        name="Fraud Count",
        opacity=0.35,
        marker_color="#457B9D",
        yaxis="y2",
    ))

    fig.update_layout(
        yaxis=dict(title="Avg Risk Score", rangemode="tozero"),
        yaxis2=dict(
            title="Fraud Count",
            overlaying="y",
            side="right",
            rangemode="tozero",
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Merchant ranking table ────────────────────────────────────────────────────

def render_merchant_table(df: pd.DataFrame) -> None:
    """Colour-coded merchant risk ranking table."""
    ranking = analytics_service.get_merchant_ranking(df)
    st.dataframe(
        ranking.style.background_gradient(subset=["avg_risk"], cmap="OrRd"),
        use_container_width=True,
        hide_index=True,
    )


# ── High-risk transactions table ──────────────────────────────────────────────

def render_high_risk_table(df: pd.DataFrame, threshold: float) -> None:
    """Flagged transaction table, capped at 200 rows for performance."""
    high_risk = analytics_service.get_high_risk_transactions(df, threshold)

    if high_risk.empty:
        st.success(
            f"No transactions exceed the risk threshold of {threshold:.2f}. "
            "The portfolio looks clean at this sensitivity level."
        )
        return

    display_limit = 200
    total = len(high_risk)
    shown = min(total, display_limit)
    st.caption(f"Showing top {shown:,} of {total:,} high-risk transactions")

    st.dataframe(
        high_risk.head(display_limit).style.background_gradient(
            subset=["final_risk_score"], cmap="Reds"
        ),
        use_container_width=True,
        hide_index=True,
    )


# ── AI insights panel ─────────────────────────────────────────────────────────

def render_ai_panel(df: pd.DataFrame, threshold: float) -> None:
    """
    Render AI-generated insights using appropriate Streamlit alert widgets
    based on the severity prefix returned by the agent.
    """
    insights = analytics_service.get_ai_insights(df, threshold)
    for insight in insights:
        if insight.startswith("ALERT"):
            st.error(f"🚨 {insight}")
        elif insight.startswith("WARNING"):
            st.warning(f"⚠️  {insight}")
        else:
            st.info(f"ℹ️  {insight}")


# ── Main layout ───────────────────────────────────────────────────────────────

def main() -> None:
    # Load data (cached after first run)
    df_full = load_scored_data()

    # Sidebar controls
    selected_merchants, threshold = render_sidebar(df_full)

    # Apply merchant filter — all downstream components use this scoped view
    df = df_full[df_full["merchant_id"].isin(selected_merchants)]

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("Hybrid AI Risk Control Center")
    st.markdown(
        f"Real-time transaction risk monitoring &nbsp;·&nbsp; "
        f"Logistic Regression + Rule Engine &nbsp;·&nbsp; "
        f"**{len(selected_merchants)} merchant(s)** in scope &nbsp;·&nbsp; "
        f"Threshold: **{threshold:.2f}**",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── KPI row ───────────────────────────────────────────────────────────────
    snapshot = analytics_service.get_dashboard_snapshot(df, threshold)
    render_kpi_row(snapshot)

    st.divider()

    # ── AI Insights ───────────────────────────────────────────────────────────
    with st.expander("🤖  AI Risk Insights", expanded=True):
        render_ai_panel(df, threshold)

    st.divider()

    # ── Charts row ────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Risk Score Distribution")
        render_risk_histogram(df)
    with col_right:
        st.subheader("Daily Risk Trend")
        render_daily_trend(df)

    st.divider()

    # ── Merchant ranking ──────────────────────────────────────────────────────
    st.subheader("Merchant Risk Ranking")
    render_merchant_table(df)

    st.divider()

    # ── High-risk transactions ────────────────────────────────────────────────
    st.subheader(f"High-Risk Transactions  (score ≥ {threshold:.2f})")
    render_high_risk_table(df, threshold)


if __name__ == "__main__":
    main()
