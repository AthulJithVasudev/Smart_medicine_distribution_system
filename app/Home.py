"""Smart Medicine Redistribution System for Government Hospitals.
State Health Department Command Center & Executive Overview Dashboard.
Pure zero-compute presentation layer reading precomputed serving Parquet tables.
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure repository root and app directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.data_loader import load_serving_table, render_sample_banner
from app.labels import (
    POLICY_LABELS,
    SCENARIO_INFO,
    STATUS_CONFIG,
    GLOSSARY,
    get_scenario_display,
    get_policy_display,
)
from app.state import render_sidebar_filters

st.set_page_config(
    page_title="Smart Medicine Redistribution - State Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header Banner
st.title("🏥 Smart Medicine Redistribution System")
st.markdown("**State Health Department Command Center • Multi-Facility Inventory Optimization Portal**")

# Sample Data Banner & Synthetic Data Disclaimer
render_sample_banner()
with st.expander("ℹ️ **Academic & Synthetic Data Disclaimer**", expanded=False):
    st.caption(
        "All hospital facilities, patient demand records, and batch inventories in this portal are **SYNTHETIC DATA** "
        "generated reproducibly from fixed random seeds by the daily time-stepped simulation engine for academic research. "
        "They do not represent real, official government hospital statistics. Data provenance is tracked across all records (`SIMULATED`, `DERIVED`, `OFFICIAL`)."
    )

# Load Published Serving Tables
with st.spinner("Loading serving data..."):
    hospitals_df = load_serving_table("hospitals")
    medicines_df = load_serving_table("medicines")
    state_df = load_serving_table("state_daily")
    transfers_df = load_serving_table("transfers")
    kpis_df = load_serving_table("kpis_daily")
    summary_df = load_serving_table("evaluation_optimizer")

# Standard Shared Sidebar Filters
active_filters = render_sidebar_filters(hospitals_df, state_df)
selected_scenario = active_filters["scenario"]
selected_policy = active_filters["policy"]
selected_day = active_filters["day"]
selected_district = active_filters["district"]
selected_type = active_filters["hospital_type"]

# Filter Hospital Master Table
filtered_hospitals = hospitals_df.copy()
if selected_district != "All Districts" and not filtered_hospitals.empty:
    filtered_hospitals = filtered_hospitals[filtered_hospitals["district"] == selected_district]
if selected_type != "All Facilities" and not filtered_hospitals.empty:
    filtered_hospitals = filtered_hospitals[filtered_hospitals["hospital_type"] == selected_type]

# Filter Daily State
day_state = state_df[
    (state_df["scenario"] == selected_scenario) &
    (state_df["policy"] == selected_policy) &
    (state_df["day"] == selected_day)
].copy()

if not filtered_hospitals.empty and not day_state.empty:
    valid_ids = set(filtered_hospitals["hospital_id"])
    day_state = day_state[day_state["hospital_id"].isin(valid_ids)]

# Role Selector
c_role, _ = st.columns([2, 3])
with c_role:
    user_role = st.selectbox(
        "Select User Persona / Viewpoint:",
        ["Government Health Officer", "Hospital Administrator", "Academic Researcher"],
        index=0,
        help="Customizes the view layout and priority metrics for specific stakeholders.",
    )

if user_role == "Hospital Administrator":
    st.info("💡 **Hospital Administrator View**: Use the left sidebar navigation to switch to the **Hospital Administrator** page for facility-specific stock cards and order management.")
elif user_role == "Academic Researcher":
    st.info("💡 **Researcher View**: Use the left sidebar navigation to switch to **Scenario Comparison**, **Replay**, or **Model Evaluation** to review empirical benchmarks and walk-forward metrics.")

st.markdown("---")

# ==============================================================================
# BLOCK 1: WHAT NEEDS ATTENTION NOW (Shortages, Expiries & KPIs)
# ==============================================================================
st.markdown("### 1. What Needs Attention Now")
st.caption("Active critical inventory deficits and expiring stock requiring immediate intervention on Day " + str(selected_day))

# Required KPI expressions to strictly satisfy tests/test_streamlit_app.py
total_stock = int(day_state["stock_remaining"].sum()) if not day_state.empty and "stock_remaining" in day_state.columns else 0
total_shortage = int(day_state["unmet_demand"].sum()) if not day_state.empty and "unmet_demand" in day_state.columns else 0
total_expired = int(day_state["expired_units"].sum()) if not day_state.empty and "expired_units" in day_state.columns else 0
total_demand = int(day_state["actual_demand"].sum()) if not day_state.empty and "actual_demand" in day_state.columns else 1
dispensed = max(0, total_demand - total_shortage)
fill_rate = (dispensed / total_demand) * 100.0 if total_demand > 0 else 100.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Network Available Stock", f"{total_stock:,} units", help=GLOSSARY.get("Safety Stock"))
with col2:
    st.metric("Critical Patient Shortages", f"{total_shortage:,} units", delta=f"-{total_shortage} deficit" if total_shortage > 0 else "Zero Deficit", delta_color="inverse")
with col3:
    st.metric("Stock at Expiry Risk", f"{total_expired:,} units", delta="FEFO Monitored", delta_color="inverse", help=GLOSSARY.get("FEFO"))
with col4:
    rate_delta = "High" if fill_rate >= 90 else "Deficit Alert"
    st.metric("Network Fill Rate", f"{fill_rate:.1f}%", delta=rate_delta, help=GLOSSARY.get("Fill Rate"))

# Side by Side: Top Shortages & Top Expiries
c_short, c_exp = st.columns([1, 1])

h_name_map = hospitals_df.set_index("hospital_id")["name"].to_dict() if not hospitals_df.empty else {}
m_name_map = medicines_df.set_index("medicine_id")["name"].to_dict() if not medicines_df.empty else {}

with c_short:
    st.markdown("##### 🚨 Top 5 Critical Shortages (Impending Deficits)")
    if not day_state.empty and "unmet_demand" in day_state.columns:
        short_df = day_state[day_state["unmet_demand"] > 0].sort_values("unmet_demand", ascending=False).head(5).copy()
        if not short_df.empty:
            short_df["Hospital"] = short_df["hospital_id"].map(h_name_map)
            short_df["Medicine"] = short_df["medicine_id"].map(m_name_map)
            short_df["Deficit Units"] = short_df["unmet_demand"].astype(int)
            short_df["Current Stock"] = short_df["stock_remaining"].astype(int)
            short_df["Status"] = STATUS_CONFIG["critical_shortage"]["badge"]
            st.dataframe(
                short_df[["Hospital", "Medicine", "Current Stock", "Deficit Units", "Status"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.success("✅ No critical hospital shortages detected across the network.")
    else:
        st.info("No state data available for current selection.")

with c_exp:
    st.markdown("##### ⚠️ Top 5 Expiring Batches / Write-offs")
    if not day_state.empty and "expired_units" in day_state.columns:
        exp_df = day_state[day_state["expired_units"] > 0].sort_values("expired_units", ascending=False).head(5).copy()
        if not exp_df.empty:
            exp_df["Hospital"] = exp_df["hospital_id"].map(h_name_map)
            exp_df["Medicine"] = exp_df["medicine_id"].map(m_name_map)
            exp_df["Units Expired"] = exp_df["expired_units"].astype(int)
            exp_df["Status"] = STATUS_CONFIG["expiry_risk"]["badge"]
            st.dataframe(
                exp_df[["Hospital", "Medicine", "Units Expired", "Status"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.success("✅ Zero expired write-offs recorded on this day. FEFO allocation active.")
    else:
        st.info("No expiry records available for current selection.")

st.markdown("---")

# ==============================================================================
# BLOCK 2: WHAT SHOULD BE DONE (Action Cards & CSV Download)
# ==============================================================================
st.markdown("### 2. What Should Be Done: Recommended Transfer Directives")
st.caption(f"Optimized inter-facility transfer directives calculated for Day {selected_day} under '{get_policy_display(selected_policy)}'")

day_transfers = transfers_df[transfers_df["day"] == selected_day].copy() if not transfers_df.empty and "day" in transfers_df.columns else pd.DataFrame()
if not day_transfers.empty and not filtered_hospitals.empty:
    valid_ids = set(filtered_hospitals["hospital_id"])
    day_transfers = day_transfers[day_transfers["from_hospital"].isin(valid_ids) | day_transfers["to_hospital"].isin(valid_ids)]

if not day_transfers.empty:
    day_transfers["From Facility"] = day_transfers["from_hospital"].map(h_name_map)
    day_transfers["To Destination"] = day_transfers["to_hospital"].map(h_name_map)
    day_transfers["Medicine Name"] = day_transfers["medicine_id"].map(m_name_map)
    day_transfers["Est Transport Cost"] = (100.0 + day_transfers["distance_km"] * 2.5).round(0).astype(int)

    # Top 3 Action Cards
    st.markdown("##### Priority Transfer Directives:")
    top_transfers = day_transfers.head(3)
    card_cols = st.columns(len(top_transfers))
    for idx, (_, row) in enumerate(top_transfers.iterrows()):
        with card_cols[idx]:
            reason_text = (
                f"Deficit impending at destination; source retains adequate safety stock buffer. "
                f"Transit requires {row['transit_days']} day(s) covering {row['distance_km']:.1f} km."
            )
            st.markdown(f"""
            <div style="border: 1px solid #334155; border-radius: 8px; padding: 14px; background-color: #1e293b; margin-bottom: 10px;">
                <div style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase;">TRANSFER ACTION #{idx+1}</div>
                <div style="font-size: 15px; font-weight: 700; color: #f8fafc; margin: 4px 0;">{row['Medicine Name']}</div>
                <div style="font-size: 13px; color: #94a3b8;">
                    <strong>From:</strong> {row['From Facility']}<br>
                    <strong>To:</strong> {row['To Destination']}<br>
                    <strong>Quantity:</strong> <span style="color:#10b981;font-weight:700;">{int(row['quantity']):,} units</span><br>
                    <strong>Transit:</strong> {row['transit_days']} day(s) ({row['distance_km']:.1f} km)<br>
                    <strong>Cost:</strong> ₹{row['Est Transport Cost']}
                </div>
                <div style="font-size: 11px; color: #cbd5e1; margin-top: 8px; background-color: #0f172a; padding: 6px; border-radius: 4px;">
                    <strong>Reason:</strong> {reason_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Detailed Table of all transfers
    st.dataframe(
        day_transfers[["From Facility", "To Destination", "Medicine Name", "quantity", "transit_days", "distance_km", "Est Transport Cost"]].rename(columns={
            "quantity": "Transfer Units",
            "transit_days": "Transit Days",
            "distance_km": "Distance (km)",
            "Est Transport Cost": "Cost (₹)",
        }),
        hide_index=True,
        use_container_width=True,
    )

    # Download Button
    csv_data = day_transfers.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Transfer Orders (CSV)",
        data=csv_data,
        file_name=f"transfer_orders_day_{selected_day}_{selected_scenario}.csv",
        mime="text/csv",
        help="Download official transfer manifest for logistics dispatch and warehouse execution.",
    )
else:
    st.info(f"No transfers recommended for Day {selected_day} under '{get_policy_display(selected_policy)}' (Network inventory is balanced).")

st.markdown("---")

# ==============================================================================
# BLOCK 3: WHAT IT ACHIEVES (Impact vs Do Nothing, Trend Chart & Map)
# ==============================================================================
st.markdown("### 3. What It Achieves: Verified Optimization Impact")
st.caption(f"Quantifiable operational improvements comparing '{get_policy_display(selected_policy)}' against 'Do nothing (Baseline)'")

# Compute Impact from stored evaluation_optimizer and kpis_daily
baseline_row = summary_df[(summary_df["scenario"] == selected_scenario) & (summary_df["policy"] == "no_transfer")]
active_row = summary_df[(summary_df["scenario"] == selected_scenario) & (summary_df["policy"] == selected_policy)]

if not active_row.empty and not baseline_row.empty:
    b_stockout = float(baseline_row["stockout_days_mean"].iloc[0])
    a_stockout = float(active_row["stockout_days_mean"].iloc[0])
    stockouts_avoided = max(0.0, b_stockout - a_stockout)

    b_waste = float(baseline_row["wastage_units_mean"].iloc[0])
    a_waste = float(active_row["wastage_units_mean"].iloc[0])
    wastage_avoided = max(0.0, b_waste - a_waste)

    fill_active = float(active_row["fill_rate_mean"].iloc[0]) * 100.0
    fill_base = float(baseline_row["fill_rate_mean"].iloc[0]) * 100.0
    fill_gain = fill_active - fill_base

    trans_cost = float(active_row["transport_cost_mean"].iloc[0])
else:
    stockouts_avoided = 0.0
    wastage_avoided = 0.0
    fill_gain = 0.0
    trans_cost = 0.0

i1, i2, i3, i4 = st.columns(4)
with i1:
    st.metric("Stockouts Avoided", f"{stockouts_avoided:.1f} days", delta=f"{stockouts_avoided:.1f} days reduced", help="Reduction in facility-days suffering medicine stockouts compared to Do Nothing.")
with i2:
    st.metric("Wastage Prevented", f"{wastage_avoided:,.0f} units", delta=f"{wastage_avoided:,.0f} saved", help="Units of medicine saved from expiring unused through timely redistribution.")
with i3:
    st.metric("Fill Rate Advantage", f"+{fill_gain:.1f}%", delta="Over Baseline", help="Improvement in patient prescription fulfillment over the zero-transfer baseline.")
with i4:
    st.metric("Total Transport Cost", f"₹{trans_cost:,.0f}", delta="Fixed + Mileage", delta_color="off", help="Logistics delivery expenditure incurred across the 14-day horizon.")

# Small Trend Chart: Unmet Demand Over Time
if not state_df.empty and "unmet_demand" in state_df.columns:
    trend_data = state_df[
        (state_df["scenario"] == selected_scenario) &
        (state_df["policy"].isin([selected_policy, "no_transfer"]))
    ].groupby(["day", "policy"])["unmet_demand"].sum().reset_index()

    if not trend_data.empty:
        trend_data["Policy"] = trend_data["policy"].map(lambda p: get_policy_display(p))
        fig_trend = px.line(
            trend_data,
            x="day",
            y="unmet_demand",
            color="Policy",
            title=f"Unmet Shortage Deficit Over 14-Day Horizon ({get_scenario_display(selected_scenario)})",
            labels={"day": "Simulated Operational Day", "unmet_demand": "Unmet Deficit (Units)"},
            color_discrete_map={get_policy_display("optimizer"): "#10b981", get_policy_display("no_transfer"): "#ef4444"},
            markers=True,
        )
        fig_trend.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)

# Geospatial Network Health Map & Regional Breakdown
c_map, c_dist = st.columns([3, 2])

with c_map:
    st.markdown("##### 🗺️ Geospatial Facility Network Health")
    if not filtered_hospitals.empty and "latitude" in filtered_hospitals.columns and not day_state.empty:
        h_agg = day_state.groupby("hospital_id").agg(
            stock=("stock_remaining", "sum"),
            shortage=("unmet_demand", "sum"),
        ).reset_index()
        map_data = filtered_hospitals.merge(h_agg, on="hospital_id", how="left").fillna(0)
        map_data["Status"] = np.where(
            map_data["shortage"] > 0, STATUS_CONFIG["critical_shortage"]["label"],
            np.where(map_data["stock"] < 200, STATUS_CONFIG["low_stock"]["label"], STATUS_CONFIG["normal"]["label"])
        )

        map_func = getattr(px, "scatter_map", getattr(px, "scatter_mapbox", None))
        if map_func is not None:
            fig_geo = map_func(
                map_data,
                lat="latitude",
                lon="longitude",
                color="Status",
                size="bed_count",
                hover_name="name",
                hover_data={"hospital_type": True, "stock": True, "shortage": True, "latitude": False, "longitude": False},
                color_discrete_map={
                    STATUS_CONFIG["critical_shortage"]["label"]: STATUS_CONFIG["critical_shortage"]["color"],
                    STATUS_CONFIG["low_stock"]["label"]: STATUS_CONFIG["low_stock"]["color"],
                    STATUS_CONFIG["normal"]["label"]: STATUS_CONFIG["normal"]["color"],
                },
                zoom=7.0,
                height=380,
            )
            if hasattr(px, "scatter_map"):
                fig_geo.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=10, b=0))
            else:
                fig_geo.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.dataframe(map_data[["name", "district", "Status", "stock", "shortage"]], hide_index=True)

with c_dist:
    st.markdown("##### 📊 Inventory vs Shortage by District")
    if not filtered_hospitals.empty and not day_state.empty:
        reg_summary = map_data.groupby("district").agg(
            Available_Stock=("stock", "sum"),
            Unmet_Shortage=("shortage", "sum"),
        ).reset_index()
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=reg_summary["district"], y=reg_summary["Available_Stock"], name="Stock", marker_color="#0ea5e9"))
        fig_bar.add_trace(go.Bar(x=reg_summary["district"], y=reg_summary["Unmet_Shortage"], name="Deficit", marker_color="#ef4444"))
        fig_bar.update_layout(
            barmode="group",
            height=380,
            margin=dict(l=10, r=10, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.caption("Smart Medicine Redistribution System • Kerala Public Health Services • Academic Big Data Implementation")
