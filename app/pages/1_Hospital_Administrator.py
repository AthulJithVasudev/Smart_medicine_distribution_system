"""Hospital Administrator local view: inventory, days of stock, forecast uncertainty, and action orders."""
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

st.set_page_config(page_title="Hospital Administrator", page_icon="🏥", layout="wide")

hospitals_df = load_serving_table("hospitals")
medicines_df = load_serving_table("medicines")
state_df = load_serving_table("state_daily")
transfers_df = load_serving_table("transfers")
replay_df = load_serving_table("replay")

if hospitals_df.empty or state_df.empty:
    st.error("Serving data not found. Please run the pipeline export first.")
    st.stop()

st.title("Hospital Administrator Portal")
st.caption("Local Facility Operations, Inventory Health, and Actionable Transfer Orders")

# Sidebar Hospital Selector
h_dict = dict(zip(hospitals_df["hospital_id"], hospitals_df["name"]))
selected_hid = st.sidebar.selectbox("Select Your Facility", list(h_dict.keys()), format_func=lambda hid: h_dict[hid])
selected_day = st.sidebar.slider("Current Day", 0, int(state_df["day"].max()), int(min(7, state_df["day"].max())))

h_info = hospitals_df[hospitals_df["hospital_id"] == selected_hid].iloc[0]
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Facility**: {h_info['name']}")
st.sidebar.markdown(f"**Type**: {h_info['hospital_type']}")
st.sidebar.markdown(f"**District**: {h_info['district']}")
st.sidebar.markdown(f"**Bed Capacity**: {h_info['bed_count']} beds")
st.sidebar.markdown(f"**Location**: {'Rural' if h_info['is_rural'] else 'Urban'}")

# Current hospital day data
h_day = state_df[(state_df["hospital_id"] == selected_hid) & (state_df["day"] == selected_day)]
m_dict = dict(zip(medicines_df["medicine_id"], medicines_df["name"]))

# Facility Status Metrics
total_units = int(h_day["stock_remaining"].sum()) if not h_day.empty else 0
shortages_count = int((h_day["unmet_demand"] > 0).sum()) if not h_day.empty else 0
expiring_units = int(h_day["expired_units"].sum()) if not h_day.empty else 0

# Check incoming & outgoing transfers for this hospital on this day
inc_trans = transfers_df[transfers_df["to_hospital"] == selected_hid] if not transfers_df.empty else pd.DataFrame()
out_trans = transfers_df[transfers_df["from_hospital"] == selected_hid] if not transfers_df.empty else pd.DataFrame()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total On-Hand Inventory", f"{total_units:,} units")
with col2:
    st.metric("Critical Shortage Items", f"{shortages_count} medicines", delta=f"-{shortages_count}" if shortages_count > 0 else "Normal", delta_color="inverse")
with col3:
    st.metric("Incoming Shipments", f"{len(inc_trans)} orders")
with col4:
    st.metric("Dispatched Shipments", f"{len(out_trans)} orders")

st.markdown("---")

# Actionable Recommendation Messages Banner
st.subheader("📋 Central Command Actionable Directives")
rec_messages = []
if not inc_trans.empty:
    for _, tr in inc_trans.iterrows():
        from_name = h_dict.get(tr["from_hospital"], tr["from_hospital"])
        med_name = m_dict.get(tr["medicine_id"], tr["medicine_id"])
        rec_messages.append(
            f"**INCOMING SHIPMENT**: Expect delivery of **{int(tr['quantity'])} units** of **{med_name}** from **{from_name}** (Transit: {tr['transit_days']} days, Dist: {tr['distance_km']} km)."
        )
if not out_trans.empty:
    for _, tr in out_trans.iterrows():
        to_name = h_dict.get(tr["to_hospital"], tr["to_hospital"])
        med_name = m_dict.get(tr["medicine_id"], tr["medicine_id"])
        rec_messages.append(
            f"**ACTION REQUIRED**: Dispatch **{int(tr['quantity'])} units** of **{med_name}** to **{to_name}** to mitigate regional stockout."
        )

if rec_messages:
    for msg in rec_messages:
        st.info(msg)
else:
    st.success("No transfer directives required today. Local inventory is balanced.")

st.markdown("---")

# Forecast with Uncertainty & Current Stock per Medicine
st.subheader("📈 Medicine Inventory & Demand Forecast with Uncertainty")

if not replay_df.empty:
    h_replay = replay_df[(replay_df["hospital_id"] == selected_hid) & (replay_df["day"] == selected_day)].copy()
    if not h_replay.empty:
        h_replay["Medicine"] = h_replay["medicine_id"].map(m_dict)
        h_replay["Days of Stock"] = (h_replay["stock_remaining"] / h_replay["forecast"].clip(lower=1.0)).round(1)

        # Plotly horizontal bar chart comparing stock vs forecasted demand
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Bar(
            y=h_replay["Medicine"],
            x=h_replay["stock_remaining"],
            name="Available Stock",
            orientation="h",
            marker_color="#38bdf8",
        ))
        fig_inv.add_trace(go.Bar(
            y=h_replay["Medicine"],
            x=h_replay["forecast"],
            name="Forecasted Demand",
            orientation="h",
            marker_color="#f59e0b",
            error_x=dict(
                type="data",
                symmetric=False,
                array=h_replay["upper_bound"] - h_replay["forecast"],
                arrayminus=h_replay["forecast"] - h_replay["lower_bound"],
                color="#cbd5e1",
            ),
        ))
        fig_inv.update_layout(
            barmode="group",
            height=500,
            xaxis_title="Units",
            margin=dict(l=10, r=10, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_inv, use_container_width=True)

        st.markdown("### 📋 Detailed Formulary Table")
        disp_table = h_replay[[
            "Medicine", "stock_remaining", "forecast", "lower_bound", "upper_bound", "actual_demand", "unmet_demand", "Days of Stock"
        ]].rename(columns={
            "stock_remaining": "Stock On Hand",
            "forecast": "Forecast Demand",
            "lower_bound": "Lower 80% CI",
            "upper_bound": "Upper 80% CI",
            "actual_demand": "Realized Demand",
            "unmet_demand": "Stockout Deficit",
        })
        st.dataframe(disp_table, hide_index=True, use_container_width=True)
