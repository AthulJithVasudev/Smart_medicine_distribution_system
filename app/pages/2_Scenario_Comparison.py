"""Scenario comparison page evaluating policies across normal, monsoon, dengue, and supply delays."""
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

st.set_page_config(page_title="Scenario Comparison & Stress Testing", page_icon="🔬", layout="wide")

catalog_df = load_serving_table("scenario_catalog")
eval_opt_df = load_serving_table("evaluation_optimizer")
daily_kpis_df = load_serving_table("kpis_daily")

st.title("Scenario Comparison & Distribution Shift Stress Testing")
st.caption("Evaluating OR-Tools Optimizer, Greedy Heuristics, and No-Transfer Policies Across Seen vs Unseen Scenarios")

if catalog_df.empty or eval_opt_df.empty:
    st.error("Scenario evaluation data not found. Please run the scenario matrix first.")
    st.stop()

# Scenario Categories Tab/Separation
seen_scenarios = catalog_df[catalog_df["type"] == "seen"]["name"].tolist()
unseen_scenarios = catalog_df[catalog_df["type"] == "unseen"]["name"].tolist()

c_seen, c_unseen = st.columns(2)
with c_seen:
    st.success(f"**SEEN CONDITIONS (In-Distribution)**\n\n{', '.join(seen_scenarios)}")
with c_unseen:
    st.warning(f"**UNSEEN CONDITIONS (Distribution Shift / Stress Tests)**\n\n{', '.join(unseen_scenarios)}")

st.markdown("---")

# Controls
col_s1, col_s2 = st.columns(2)
with col_s1:
    selected_scenario = st.selectbox("Select Scenario to Inspect", catalog_df["name"].tolist(), index=0)
with col_s2:
    selected_policy = st.selectbox("Select Benchmark Policy", ["optimizer", "greedy", "no_transfer"], index=0)

sc_info = catalog_df[catalog_df["name"] == selected_scenario].iloc[0]
st.markdown(f"**Description**: {sc_info['description']}")
st.markdown(f"**Severity Multiplier**: `{sc_info['outbreak_multiplier']}x` | **Monsoon Flag**: `{sc_info['is_monsoon']}` | **Lead Delay**: `{sc_info['supplier_delay_days']} days`")

st.markdown("---")

# Cross-Policy Benchmark Bar Charts
st.subheader("📊 Cross-Policy Performance Comparison")

# Filter evaluation summary
sc_eval = eval_opt_df[eval_opt_df["scenario"] == selected_scenario]
if not sc_eval.empty:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        fig_w = px.bar(
            sc_eval,
            x="policy",
            y="wastage_units_mean",
            error_y="wastage_units_std",
            title="Wastage Units (Lower is Better)",
            color="policy",
            color_discrete_map={"optimizer": "#10b981", "greedy": "#f59e0b", "no_transfer": "#ef4444"},
        )
        fig_w.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_w, use_container_width=True)

    with m2:
        fig_s = px.bar(
            sc_eval,
            x="policy",
            y="stockout_days_mean",
            error_y="stockout_days_std",
            title="Stockout Incidents (Lower is Better)",
            color="policy",
            color_discrete_map={"optimizer": "#10b981", "greedy": "#f59e0b", "no_transfer": "#ef4444"},
        )
        fig_s.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_s, use_container_width=True)

    with m3:
        fig_f = px.bar(
            sc_eval,
            x="policy",
            y="fill_rate_mean",
            error_y="fill_rate_std",
            title="Fill Rate % (Higher is Better)",
            color="policy",
            color_discrete_map={"optimizer": "#10b981", "greedy": "#f59e0b", "no_transfer": "#ef4444"},
        )
        fig_f.update_layout(showlegend=False, height=350, yaxis_range=[0.7, 1.0])
        st.plotly_chart(fig_f, use_container_width=True)

    with m4:
        fig_t = px.bar(
            sc_eval,
            x="policy",
            y="transport_cost_mean",
            error_y="transport_cost_std",
            title="Transport Cost ₹",
            color="policy",
            color_discrete_map={"optimizer": "#10b981", "greedy": "#f59e0b", "no_transfer": "#ef4444"},
        )
        fig_t.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_t, use_container_width=True)
else:
    st.info("Evaluation metrics for this scenario are being processed.")

# Comparison Against Normal Baseline
st.markdown("---")
st.subheader("⚖️ Scenario vs Baseline Normal Comparison")
norm_eval = eval_opt_df[eval_opt_df["scenario"] == "normal"]
if not sc_eval.empty and not norm_eval.empty:
    opt_sc = sc_eval[sc_eval["policy"] == "optimizer"].iloc[0] if not sc_eval[sc_eval["policy"] == "optimizer"].empty else None
    opt_norm = norm_eval[norm_eval["policy"] == "optimizer"].iloc[0] if not norm_eval[norm_eval["policy"] == "optimizer"].empty else None

    if opt_sc is not None and opt_norm is not None:
        delta_stockout = opt_sc["stockout_days_mean"] - opt_norm["stockout_days_mean"]
        delta_fill = (opt_sc["fill_rate_mean"] - opt_norm["fill_rate_mean"]) * 100.0
        delta_cost = opt_sc["transport_cost_mean"] - opt_norm["transport_cost_mean"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Stockout Shift vs Normal", f"{opt_sc['stockout_days_mean']:.1f} days", delta=f"{delta_stockout:+.1f} days", delta_color="inverse")
        c2.metric("Fill Rate Delta vs Normal", f"{opt_sc['fill_rate_mean']*100:.1f}%", delta=f"{delta_fill:+.1f}%")
        c3.metric("Transport Cost Delta vs Normal", f"₹{opt_sc['transport_cost_mean']:,.0f}", delta=f"₹{delta_cost:+,.0f}", delta_color="inverse")

# Full Matrix Table
st.markdown("---")
st.subheader("📑 Full Scenario Matrix Benchmark Table")
st.dataframe(
    eval_opt_df[[
        "scenario", "policy", "seeds_count", "wastage_units_mean", "stockout_days_mean", "fill_rate_mean", "transport_cost_mean"
    ]].rename(columns={
        "scenario": "Scenario",
        "policy": "Policy",
        "seeds_count": "Seeds",
        "wastage_units_mean": "Mean Wastage (Units)",
        "stockout_days_mean": "Mean Stockout Days",
        "fill_rate_mean": "Mean Fill Rate",
        "transport_cost_mean": "Mean Transport Cost (₹)",
    }),
    hide_index=True,
    use_container_width=True,
)
