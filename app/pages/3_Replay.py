"""Replay mode page: compares forecast issued on Day X vs actual demand that followed."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Ensure repository root and app directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from app.data_loader import load_serving_table
except ImportError:
    from data_loader import load_serving_table

st.set_page_config(page_title="Forecasting Replay & Verification", page_icon="⏪", layout="wide")

replay_df = load_serving_table("replay")
hospitals_df = load_serving_table("hospitals")
medicines_df = load_serving_table("medicines")

st.title("Replay Mode — Forecast Verification")
st.caption("Inspect exact ex-ante forecasts issued on Day X against actual ground-truth patient demand that subsequently realized.")

if replay_df.empty:
    st.error("Replay dataset not found in published serving tables. Run the scenario engine first.")
    st.stop()

# Selectors
scenarios = sorted(replay_df["scenario"].unique().tolist())
selected_scenario = st.sidebar.selectbox("Scenario", scenarios, index=0)

h_dict = dict(zip(hospitals_df["hospital_id"], hospitals_df["name"])) if not hospitals_df.empty else {}
m_dict = dict(zip(medicines_df["medicine_id"], medicines_df["name"])) if not medicines_df.empty else {}

h_options = sorted(replay_df["hospital_id"].unique().tolist())
selected_h = st.sidebar.selectbox("Hospital", h_options, format_func=lambda hid: h_dict.get(hid, hid))

m_options = sorted(replay_df["medicine_id"].unique().tolist())
selected_m = st.sidebar.selectbox("Medicine", m_options, format_func=lambda mid: m_dict.get(mid, mid))

sub_replay = replay_df[
    (replay_df["scenario"] == selected_scenario) &
    (replay_df["hospital_id"] == selected_h) &
    (replay_df["medicine_id"] == selected_m)
].sort_values("day")

if sub_replay.empty:
    st.info("No replay history matching selected filters.")
    st.stop()

# Time series chart: Forecast vs Actual Demand over time
st.subheader(f"📊 Forecast Horizon Trajectory for {m_dict.get(selected_m, selected_m)} at {h_dict.get(selected_h, selected_h)}")

fig = go.Figure()

# Actual realized demand
fig.add_trace(go.Scatter(
    x=sub_replay["day"],
    y=sub_replay["actual_demand"],
    mode="lines+markers",
    name="Actual Patient Demand",
    line=dict(color="#10b981", width=3),
    marker=dict(size=7),
))

# Ex-ante forecast
fig.add_trace(go.Scatter(
    x=sub_replay["day"],
    y=sub_replay["forecast"],
    mode="lines+markers",
    name="Forecast Issued",
    line=dict(color="#38bdf8", width=2, dash="dash"),
    marker=dict(size=6),
))

# 80% Confidence Interval band
fig.add_trace(go.Scatter(
    x=pd.concat([sub_replay["day"], sub_replay["day"][::-1]]),
    y=pd.concat([sub_replay["upper_bound"], sub_replay["lower_bound"][::-1]]),
    fill="toself",
    fillcolor="rgba(56, 189, 248, 0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip",
    showlegend=True,
    name="80% Forecast Confidence Band",
))

fig.update_layout(
    height=450,
    xaxis_title="Simulated Day",
    yaxis_title="Units",
    margin=dict(l=10, r=10, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# Performance metrics on this time series
abs_errors = np.abs(sub_replay["forecast"] - sub_replay["actual_demand"])
mae = abs_errors.mean()
wape = (abs_errors.sum() / max(1.0, sub_replay["actual_demand"].sum())) * 100.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Series MAE", f"{mae:.2f} units")
m2.metric("Series WAPE", f"{wape:.1f}%")
m3.metric("Peak Demand", f"{sub_replay['actual_demand'].max():.0f} units")
m4.metric("Days Sampled", f"{len(sub_replay)} days")

st.markdown("---")
st.subheader("📋 Step-by-Step Historical Replay Audit Log")
audit_table = sub_replay[[
    "day", "forecast", "lower_bound", "upper_bound", "actual_demand", "unmet_demand", "stock_remaining"
]].rename(columns={
    "day": "Day",
    "forecast": "Ex-Ante Forecast",
    "lower_bound": "Lower 80% CI",
    "upper_bound": "Upper 80% CI",
    "actual_demand": "Actual Demand",
    "unmet_demand": "Stockout Deficit",
    "stock_remaining": "Stock Remaining",
})
st.dataframe(audit_table, hide_index=True, use_container_width=True)
