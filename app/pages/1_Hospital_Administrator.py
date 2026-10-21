"""Hospital Administrator Portal: facility inventory, forecast trends, and transfer manifests."""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.data_loader import load_serving_table, render_sample_banner
from app.labels import STATUS_CONFIG, GLOSSARY, get_scenario_display, get_policy_display
from app.state import render_sidebar_filters

st.set_page_config(page_title="Hospital Administrator - Inventory & Orders", page_icon="👨‍⚕️", layout="wide")

st.title("👨‍⚕️ Hospital Administrator Portal")
st.markdown("**Facility Inventory Management, Consumption Forecasting & Local Transfer Orders**")

render_sample_banner()

# Load Serving Data
hospitals_df = load_serving_table("hospitals")
medicines_df = load_serving_table("medicines")
state_df = load_serving_table("state_daily")
transfers_df = load_serving_table("transfers")
replay_df = load_serving_table("replay")

active_filters = render_sidebar_filters(hospitals_df, state_df)
selected_scenario = active_filters["scenario"]
selected_policy = active_filters["policy"]
selected_day = active_filters["day"]

if hospitals_df.empty:
    st.warning("No hospital records available. Please run the pipeline demo.")
    st.stop()

# Hospital Selector
h_dict = hospitals_df.set_index("hospital_id")["name"].to_dict()
h_options = list(h_dict.keys())
h_names = [f"{h_dict[h_id]} ({h_id})" for h_id in h_options]

c_sel, c_meta = st.columns([2, 3])
with c_sel:
    selected_h_idx = st.selectbox("Select Hospital Facility:", range(len(h_options)), format_func=lambda i: h_names[i])
    selected_h_id = h_options[selected_h_idx]

hosp_info = hospitals_df[hospitals_df["hospital_id"] == selected_h_id].iloc[0]
with c_meta:
    st.markdown(f"""
    **Facility:** {hosp_info['name']} • **Type:** {hosp_info['hospital_type']} • **District:** {hosp_info['district']} • **Beds:** {hosp_info['bed_count']} • **Rural:** {'Yes' if hosp_info['is_rural'] else 'No'}
    """)

st.markdown("---")

# Facility Inventory State on Selected Day
hosp_state = state_df[
    (state_df["scenario"] == selected_scenario) &
    (state_df["policy"] == selected_policy) &
    (state_df["day"] == selected_day) &
    (state_df["hospital_id"] == selected_h_id)
].copy()

m_name_map = medicines_df.set_index("medicine_id")["name"].to_dict() if not medicines_df.empty else {}
m_crit_map = medicines_df.set_index("medicine_id")["criticality"].to_dict() if not medicines_df.empty else {}

if not hosp_state.empty:
    hosp_state["Medicine"] = hosp_state["medicine_id"].map(m_name_map)
    hosp_state["Criticality"] = hosp_state["medicine_id"].map(m_crit_map)
    # Estimate daily demand from actual or historical demand
    hosp_state["Daily Demand"] = hosp_state["actual_demand"].replace(0, 1)
    hosp_state["Days of Stock"] = (hosp_state["stock_remaining"] / hosp_state["Daily Demand"]).round(1)

    # Summary Metrics for Hospital
    h_tot_stock = int(hosp_state["stock_remaining"].sum())
    h_tot_short = int(hosp_state["unmet_demand"].sum())
    h_tot_exp = int(hosp_state["expired_units"].sum())
    critical_items = int((hosp_state["Days of Stock"] <= 2).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Facility Total Stock", f"{h_tot_stock:,} units")
    with c2:
        st.metric("Impending Shortages", f"{h_tot_short:,} units", delta=f"{h_tot_short} deficit", delta_color="inverse")
    with c3:
        st.metric("Stock at Expiry Risk", f"{h_tot_exp:,} units", delta="FEFO Monitored", delta_color="inverse")
    with c4:
        st.metric("Critical Medicine Items", f"{critical_items} items", delta="≤ 2 Days of Stock", delta_color="inverse")

    # Inventory Cards & Table
    st.markdown("### 📦 Stock & Days of Supply by Medicine")
    hosp_state["Status"] = np.where(
        hosp_state["unmet_demand"] > 0, STATUS_CONFIG["critical_shortage"]["badge"],
        np.where(hosp_state["Days of Stock"] <= 3, STATUS_CONFIG["low_stock"]["badge"], STATUS_CONFIG["normal"]["badge"])
    )
    st.dataframe(
        hosp_state[["Medicine", "Criticality", "stock_remaining", "Days of Stock", "unmet_demand", "Status"]].rename(columns={
            "stock_remaining": "Current Stock",
            "unmet_demand": "Unmet Deficit",
        }),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("No daily inventory state records available for this facility on Day " + str(selected_day))

st.markdown("---")

# Consumption Forecast with Uncertainty Bands vs Actual
st.markdown("### 📈 Consumption Forecast vs Realized Demand")
st.caption("Forecast issued at time t with 95% confidence bounds compared against true patient consumption.")

if not replay_df.empty:
    hosp_replay = replay_df[
        (replay_df["scenario"] == selected_scenario) &
        (replay_df["policy"] == selected_policy) &
        (replay_df["hospital_id"] == selected_h_id)
    ].copy()
    if not hosp_replay.empty:
        hosp_replay["Medicine"] = hosp_replay["medicine_id"].map(m_name_map)
        available_meds = sorted(hosp_replay["Medicine"].dropna().unique())
        selected_med = st.selectbox("Choose Medicine to Inspect Forecast:", available_meds, index=0)
        med_replay = hosp_replay[hosp_replay["Medicine"] == selected_med].sort_values("day")

        fig_fc = go.Figure()
        # Uncertainty band
        fig_fc.add_trace(go.Scatter(
            x=list(med_replay["day"]) + list(med_replay["day"])[::-1],
            y=list(med_replay["upper_bound"]) + list(med_replay["lower_bound"])[::-1],
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Forecast Band",
        ))
        # Point Forecast
        fig_fc.add_trace(go.Scatter(
            x=med_replay["day"],
            y=med_replay["forecast"],
            mode="lines+markers",
            name="ML Point Forecast",
            line=dict(color="#38bdf8", dash="dash"),
        ))
        # Realized Actual
        fig_fc.add_trace(go.Scatter(
            x=med_replay["day"],
            y=med_replay["actual_demand"],
            mode="lines+markers",
            name="Realized Demand",
            line=dict(color="#10b981", width=2),
        ))
        fig_fc.update_layout(
            title=f"Forecast Accuracy Trajectory: {selected_med} at {hosp_info['name']}",
            xaxis_title="Operational Day",
            yaxis_title="Medicine Units",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_fc, use_container_width=True)
    else:
        st.info("No replay forecast records available for this facility.")
else:
    st.info("Replay table not found.")

st.markdown("---")

# Incoming & Outgoing Transfer Directives
st.markdown("### 🚚 Facility Transfer Manifests (Incoming & Outgoing)")

if not transfers_df.empty:
    h_trans = transfers_df[
        (transfers_df["day"] == selected_day) &
        ((transfers_df["from_hospital"] == selected_h_id) | (transfers_df["to_hospital"] == selected_h_id))
    ].copy()

    if not h_trans.empty:
        h_trans["From Facility"] = h_trans["from_hospital"].map(h_dict)
        h_trans["To Facility"] = h_trans["to_hospital"].map(h_dict)
        h_trans["Medicine"] = h_trans["medicine_id"].map(m_name_map)
        h_trans["Direction"] = np.where(h_trans["from_hospital"] == selected_h_id, "OUTGOING (Surplus Dispatch)", "INCOMING (Relief Inflow)")
        h_trans["Reason"] = np.where(
            h_trans["from_hospital"] == selected_h_id,
            "Dispatched from local surplus buffer to relieve destination stockout without risking local safety stock.",
            "Replenishment scheduled to arrive before local deficit causes patient stockout."
        )

        st.dataframe(
            h_trans[["Direction", "Medicine", "From Facility", "To Facility", "quantity", "transit_days", "distance_km", "Reason"]].rename(columns={
                "quantity": "Units",
                "transit_days": "Transit Days",
                "distance_km": "Distance (km)",
            }),
            hide_index=True,
            use_container_width=True,
        )

        csv_orders = h_trans.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {hosp_info['name']} Transfer Orders (CSV)",
            data=csv_orders,
            file_name=f"transfer_manifest_{selected_h_id}_day_{selected_day}.csv",
            mime="text/csv",
        )
    else:
        st.success(f"No active transfer directives assigned to {hosp_info['name']} on Day {selected_day}. Local stock is self-sufficient.")
else:
    st.info("No transfer log records available.")
