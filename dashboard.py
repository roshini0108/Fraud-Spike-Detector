"""
Fraud-Spike Detector — Risk Dashboard
Razorpay AI Buildathon — Track 02: AI Risk Manager

Run with:  streamlit run dashboard.py

This app is purely a display layer. It reads the 4 CSV files your
notebook already produces in Step 14 ("Save dashboard-ready outputs")
and does not run any modeling itself. Place this file in the same
folder as:
    merchant_daily_risk.csv
    risk_alert_queue.csv
    merchant_risk_summary.csv
    final_model_metrics.csv
"""

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Fraud-Spike Detector", layout="wide")


# ----------------------------------------------------------------
# Load data (cached so the app doesn't reload CSVs on every click)
# ----------------------------------------------------------------
@st.cache_data
def load_data():
    daily = pd.read_csv("merchant_daily_risk.csv")
    alerts = pd.read_csv("risk_alert_queue.csv")
    summary = pd.read_csv("merchant_risk_summary.csv")
    metrics = pd.read_csv("final_model_metrics.csv").iloc[0]
    daily["date"] = pd.to_datetime(daily["date"])
    alerts["date"] = pd.to_datetime(alerts["date"])
    return daily, alerts, summary, metrics


try:
    daily, alerts, summary, metrics = load_data()
except FileNotFoundError as e:
    st.error(
        f"Missing file: {e.filename}. Run the notebook through Step 14 "
        "first, then place the generated CSVs next to this script."
    )
    st.stop()


st.title("🛡️ Fraud-Spike Detector")
st.caption(
    "Merchant behavioral risk scoring — flags merchant-days whose "
    "transaction pattern deviates from that merchant's own historical baseline."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🚨 Alert Queue", "🏪 Merchant Summary", "🔍 Merchant Drill-down"]
)

# ----------------------------------------------------------------
# TAB 1 — Overview: the headline numbers from Step 7's evaluation
# ----------------------------------------------------------------
with tab1:
    st.subheader("Model performance (held-out test set)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{metrics['precision']:.1%}")
    c2.metric("Recall", f"{metrics['recall']:.1%}")
    c3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c4.metric("Alert rate", f"{metrics['alert_rate_percent']:.2f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test merchant-days", int(metrics["test_merchant_days"]))
    c2.metric("Fraud-days in test", int(metrics["test_fraud_days"]))
    c3.metric("False positives", int(metrics["false_positives"]))
    c4.metric("False negatives (missed)", int(metrics["false_negatives"]))

    st.divider()
    st.subheader("Risk level distribution across all monitored merchant-days")
    level_counts = daily["risk_level"].value_counts().reindex(
        ["Low", "Medium", "High", "Critical"]
    ).fillna(0)
    fig = px.bar(
        x=level_counts.index, y=level_counts.values,
        labels={"x": "Risk level", "y": "Number of merchant-days"},
        color=level_counts.index,
        color_discrete_map={
            "Low": "#4CAF50", "Medium": "#FFC107",
            "High": "#FF9800", "Critical": "#F44336",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Note: this is simulated (Sparkov) transaction data, not real payment data — "
        "absolute numbers won't transfer directly to production, but the merchant-relative "
        "walk-forward methodology does."
    )

# ----------------------------------------------------------------
# TAB 2 — Alert queue: what a risk-ops analyst would actually work from
# ----------------------------------------------------------------
with tab2:
    st.subheader("Ranked alert queue")

    col1, col2 = st.columns(2)
    level_filter = col1.multiselect(
        "Filter by risk level", ["Low", "Medium", "High", "Critical"],
        default=["High", "Critical"],
    )
    min_score = col2.slider("Minimum risk score", 0, 100, 75)

    filtered = alerts[
        (alerts["risk_level"].isin(level_filter)) & (alerts["risk_score"] >= min_score)
    ]

    st.write(f"Showing {len(filtered)} of {len(alerts)} total alerts")
    st.dataframe(
        filtered[
            ["merchant", "date", "risk_score", "risk_level", "risk_reason_text", "had_fraud"]
        ].sort_values("risk_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# ----------------------------------------------------------------
# TAB 3 — Merchant summary: aggregate view per merchant
# ----------------------------------------------------------------
with tab3:
    st.subheader("Per-merchant risk summary")
    st.dataframe(
        summary.sort_values("maximum_risk", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# ----------------------------------------------------------------
# TAB 4 — Drill-down: pick one merchant, see their full risk timeline
# ----------------------------------------------------------------
with tab4:
    st.subheader("Merchant drill-down")

    merchant_list = sorted(daily["merchant"].unique())
    chosen = st.selectbox("Select a merchant", merchant_list)

    m_data = daily[daily["merchant"] == chosen].sort_values("date")

    fig = px.line(
        m_data, x="date", y="risk_score",
        title=f"Daily risk score — {chosen}",
        labels={"risk_score": "Risk score (0-100)", "date": "Date"},
    )
    fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Alert threshold")
    fraud_days = m_data[m_data["had_fraud"] == 1]
    if len(fraud_days) > 0:
        fig.add_scatter(
            x=fraud_days["date"], y=fraud_days["risk_score"],
            mode="markers", marker=dict(color="black", size=9, symbol="x"),
            name="Actual fraud-day",
        )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top flagged days for this merchant**")
    top_days = m_data[m_data["is_alert"] == True].sort_values("risk_score", ascending=False)
    if len(top_days) == 0:
        st.info("No alerts were raised for this merchant in the monitored period.")
    else:
        st.dataframe(
            top_days[["date", "risk_score", "risk_level", "risk_reason_text", "had_fraud"]],
            use_container_width=True,
            hide_index=True,
        )
