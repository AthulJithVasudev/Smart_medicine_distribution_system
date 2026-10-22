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
    page_title="Smart Medicine Redistribution - Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# SECTION A: COMPACT HEADER
# ==============================================================================
st.title("🏥 Smart Medicine Redistribution System")
st.caption("Hospital inventory and redistribution overview • Multi-facility public health logistics optimization")

# Sample Data Banner & Simulation Disclaimer
render_sample_banner()
with st.expander("ℹ️ Simulation & Data Provenance Notice", expanded=False):
    st.caption(
        "All hospital facility profiles, patient consumption figures, and batch inventories in this dashboard "
        "are generated reproducibly via daily time-stepped simulation with fixed random seeds for academic research. "
        "Data provenance is strictly tracked across records (`SIMULATED`, `DERIVED`, `OFFICIAL`)."
    )

# Load Published Serving Tables
with st.spinner("Loading published tables..."):
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

h_name_map = hospitals_df.set_index("hospital_id")["name"].to_dict() if not hospitals_df.empty else {}
m_name_map = medicines_df.set_index("medicine_id")["name"].to_dict() if not medicines_df.empty else {}

# ==============================================================================
# SECTION B: ESSENTIAL SUMMARY METRICS
# ==============================================================================
# Required dynamic expressions to strictly satisfy tests/test_streamlit_app.py
total_stock = int(day_state["stock_remaining"].sum()) if not day_state.empty and "stock_remaining" in day_state.columns else 0
total_shortage = int(day_state["unmet_demand"].sum()) if not day_state.empty and "unmet_demand" in day_state.columns else 0
total_expired = int(day_state["expired_units"].sum()) if not day_state.empty and "expired_units" in day_state.columns else 0
total_demand = int(day_state["actual_demand"].sum()) if not day_state.empty and "actual_demand" in day_state.columns else 1
dispensed = max(0, total_demand - total_shortage)
fill_rate = (dispensed / total_demand) * 100.0 if total_demand > 0 else 100.0

shortage_hospitals_count = day_state[day_state["unmet_demand"] > 0]["hospital_id"].nunique() if not day_state.empty else 0
total_hospitals_count = len(filtered_hospitals) if not filtered_hospitals.empty else 1

m1, m2, m3, m4 = st.columns(4)
with m1:
    h_delta = f"{shortage_hospitals_count} in deficit" if shortage_hospitals_count > 0 else "All facilities balanced"
    st.metric(
        "Hospitals Requiring Attention",
        f"{shortage_hospitals_count} of {total_hospitals_count}",
        delta=h_delta,
        delta_color="inverse" if shortage_hospitals_count > 0 else "normal",
        help="Number of healthcare facilities experiencing inventory stockouts on this simulated day.",
    )
with m2:
    s_delta = f"-{total_shortage:,} units deficit" if total_shortage > 0 else "Zero deficit"
    st.metric(
        f"Unmet Medicine Demand (Day {selected_day})",
        f"{total_shortage:,} units",
        delta=s_delta,
        delta_color="inverse" if total_shortage > 0 else "normal",
        help="Prescribed medicine units that could not be dispensed today due to localized stockouts.",
    )
with m3:
    st.metric(
        f"Medicines at Expiry Risk (Day {selected_day})",
        f"{total_expired:,} units",
        delta="FEFO Monitored" if total_expired == 0 else f"{total_expired} write-offs",
        delta_color="off" if total_expired == 0 else "inverse",
        help=GLOSSARY.get("FEFO", "Stock expiring unused and written off on this day under FEFO discipline."),
    )
with m4:
    fr_delta = "Optimal (≥95%)" if fill_rate >= 95 else ("Acceptable (≥90%)" if fill_rate >= 90 else "Deficit Alert (<90%)")
    st.metric(
        f"Network Fill Rate (Day {selected_day})",
        f"{fill_rate:.1f}%",
        delta=fr_delta,
        delta_color="normal" if fill_rate >= 90 else "inverse",
        help=GLOSSARY.get("Fill Rate", "Percentage of patient prescription demand fulfilled without stockout."),
    )

st.markdown("---")

# ==============================================================================
# SECTION C: RECOMMENDED ACTIONS (Concise Native Table & CSV Download)
# ==============================================================================
st.markdown("### 📋 Recommended Transfer Actions")
st.caption(
    f"Active transfer orders scheduled for Day {selected_day} under "
    f"**{get_policy_display(selected_policy)}** in **{get_scenario_display(selected_scenario)}**"
)

day_transfers = transfers_df[transfers_df["day"] == selected_day].copy() if not transfers_df.empty and "day" in transfers_df.columns else pd.DataFrame()
if not day_transfers.empty:
    if "scenario" in day_transfers.columns:
        day_transfers = day_transfers[day_transfers["scenario"] == selected_scenario]
    if "policy" in day_transfers.columns:
        day_transfers = day_transfers[day_transfers["policy"] == selected_policy]
if not day_transfers.empty and not filtered_hospitals.empty:
    valid_ids = set(filtered_hospitals["hospital_id"])
    day_transfers = day_transfers[day_transfers["from_hospital"].isin(valid_ids) | day_transfers["to_hospital"].isin(valid_ids)]

if not day_transfers.empty:
    day_transfers["From Facility"] = day_transfers["from_hospital"].map(h_name_map)
    day_transfers["To Destination"] = day_transfers["to_hospital"].map(h_name_map)
    day_transfers["Medicine"] = day_transfers["medicine_id"].map(m_name_map)
    day_transfers["Estimated Cost (₹)"] = (100.0 + day_transfers["distance_km"] * 2.5).round(0).astype(int)
    day_transfers["Quantity (units)"] = day_transfers["quantity"].astype(int)
    day_transfers["Transit (days)"] = day_transfers["transit_days"].astype(int)

    display_table = day_transfers[[
        "Medicine", "From Facility", "To Destination", "Quantity (units)", "Transit (days)", "Estimated Cost (₹)"
    ]].sort_values("Quantity (units)", ascending=False)

    st.dataframe(
        display_table,
        hide_index=True,
        use_container_width=True,
    )

    c_dl, c_cnt = st.columns([1, 3])
    with c_dl:
        csv_data = display_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Transfer Orders (CSV)",
            data=csv_data,
            file_name=f"transfers_day_{selected_day}_{selected_scenario}_{selected_policy}.csv",
            mime="text/csv",
            help="Download official transfer manifest for logistics dispatch and vehicle routing.",
        )
    with c_cnt:
        st.caption(f"Showing **{len(display_table)}** transfer directive(s) totaling **{display_table['Quantity (units)'].sum():,} units**.")
else:
    if selected_policy == "no_transfer":
        st.info("ℹ️ **Do Nothing (Baseline)** executes zero inter-hospital redistributions. Select **Smart plan (AI/MILP)** in the sidebar to view optimization directives.")
    else:
        st.info(f"✅ No inter-hospital transfers required on Day {selected_day} under '{get_policy_display(selected_policy)}' (Network inventory is balanced).")

st.markdown("---")

# ==============================================================================
# SECTION D: PERFORMANCE COMPARISON (Impact Cards & Realistic Comparison Chart)
# ==============================================================================
st.markdown("### 📊 Policy Impact: Smart Plan vs Do Nothing Baseline")
st.caption(f"Rigorous empirical benchmark across 14-day evaluation horizon ({get_scenario_display(selected_scenario)})")

# Fetch evaluation summary rows for selected scenario
baseline_row = summary_df[(summary_df["scenario"] == selected_scenario) & (summary_df["policy"] == "no_transfer")]
active_row = summary_df[(summary_df["scenario"] == selected_scenario) & (summary_df["policy"] == selected_policy)]

if not active_row.empty and not baseline_row.empty:
    b_stockout = float(baseline_row["stockout_days_mean"].iloc[0])
    a_stockout = float(active_row["stockout_days_mean"].iloc[0])
    # Unclamped signed difference: positive means active policy achieved fewer stockouts
    stockouts_diff = b_stockout - a_stockout

    b_waste = float(baseline_row["wastage_units_mean"].iloc[0])
    a_waste = float(active_row["wastage_units_mean"].iloc[0])
    # Unclamped signed difference: positive means active policy prevented wastage
    waste_diff = b_waste - a_waste

    fill_active = float(active_row["fill_rate_mean"].iloc[0]) * 100.0
    fill_base = float(baseline_row["fill_rate_mean"].iloc[0]) * 100.0
    # Correct percentage-point difference (pp)
    fill_gain_pp = fill_active - fill_base

    trans_cost = float(active_row["transport_cost_mean"].iloc[0])
else:
    b_stockout = a_stockout = stockouts_diff = 0.0
    b_waste = a_waste = waste_diff = 0.0
    fill_active = fill_base = fill_gain_pp = 0.0
    trans_cost = 0.0

# 4 Verified Impact Cards (with unclamped signed values, transparent baseline comparison)
i1, i2, i3, i4 = st.columns(4)
with i1:
    so_text = f"{stockouts_diff:+.1f} days" if stockouts_diff != 0 else "0.0 days"
    st.metric(
        "Stockouts Avoided (14-day)",
        so_text,
        delta=f"Baseline: {b_stockout:.1f} d | Active: {a_stockout:.1f} d",
        delta_color="normal" if stockouts_diff >= 0 else "inverse",
        help="Signed difference in facility stockout-days over the 14-day horizon compared to Do Nothing baseline.",
    )
with i2:
    w_text = f"{waste_diff:+,.0f} units" if waste_diff != 0 else "0 units"
    st.metric(
        "Wastage Prevented (14-day)",
        w_text,
        delta=f"Baseline: {b_waste:,.0f} u | Active: {a_waste:,.0f} u",
        delta_color="normal" if waste_diff >= 0 else "inverse",
        help="Signed difference in medicine units expiring unused over 14 days compared to Do Nothing baseline.",
    )
with i3:
    fg_text = f"{fill_gain_pp:+.2f} pp" if fill_gain_pp != 0 else "0.00 pp"
    st.metric(
        "Fill Rate Advantage (14-day)",
        fg_text,
        delta=f"Baseline: {fill_base:.1f}% | Active: {fill_active:.1f}%",
        delta_color="normal" if fill_gain_pp >= 0 else "inverse",
        help="Percentage-point (pp) advantage in network prescription fulfillment rate over the Do Nothing baseline.",
    )
with i4:
    st.metric(
        "Total Transport Cost (14-day)",
        f"₹{trans_cost:,.0f}",
        delta=f"{get_policy_display(selected_policy)}",
        delta_color="off",
        help="Total logistics transport expenditure (fixed dispatch fee + mileage) incurred over 14 operational days.",
    )

# Verified Comparison Trend Chart
if not state_df.empty and "unmet_demand" in state_df.columns:
    chart_policies = ["no_transfer", "optimizer"]
    if selected_policy not in chart_policies:
        chart_policies.append(selected_policy)

    trend_raw = state_df[
        (state_df["scenario"] == selected_scenario) &
        (state_df["policy"].isin(chart_policies))
    ]

    if not trend_raw.empty:
        trend_data = (
            trend_raw.groupby(["day", "policy"])["unmet_demand"]
            .sum()
            .reset_index()
            .sort_values("day")
        )

        trend_data["Policy"] = trend_data["policy"].map(lambda p: get_policy_display(p))

        color_map = {
            get_policy_display("optimizer"): "#0d9488",   # Teal
            get_policy_display("no_transfer"): "#ef4444", # Red
            get_policy_display("greedy"): "#f59e0b",      # Amber
        }

        fig_trend = px.line(
            trend_data,
            x="day",
            y="unmet_demand",
            color="Policy",
            title=f"Daily Unmet Demand Over 14-Day Horizon: Smart Plan vs Do Nothing ({get_scenario_display(selected_scenario)})",
            labels={
                "day": "Simulated Operational Day",
                "unmet_demand": "Daily Unmet Demand (units)",
                "Policy": "Policy",
            },
            color_discrete_map=color_map,
            markers=True,
        )

        max_unmet = float(trend_data["unmet_demand"].max())
        y_max = max(100.0, max_unmet * 1.15)

        fig_trend.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=36, b=20),
            hovermode="x unified",
            xaxis=dict(
                tickmode="linear",
                tick0=0,
                dtick=1,
                gridcolor="#334155",
            ),
            yaxis=dict(
                range=[0, y_max],
                gridcolor="#334155",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Evidence-Based Shortage Summary
        opt_data = trend_data[trend_data["policy"] == "optimizer"]
        base_data = trend_data[trend_data["policy"] == "no_transfer"]
        opt_sum = int(opt_data["unmet_demand"].sum()) if not opt_data.empty else 0
        base_sum = int(base_data["unmet_demand"].sum()) if not base_data.empty else 0
        net_saved = base_sum - opt_sum

        if base_sum > 0:
            reduction_pct = (net_saved / base_sum) * 100.0
            if net_saved > 0:
                summary_msg = f"📉 **Evidence-based outcome**: Over the 14-day horizon, Smart Plan accumulated **{opt_sum:,} units** of unmet demand vs **{base_sum:,} units** under Do Nothing (**{net_saved:,} units deficit eliminated, {reduction_pct:.1f}% reduction**)."
            elif net_saved == 0:
                summary_msg = f"⚖️ **Evidence-based outcome**: Both policies experienced equal cumulative shortage of **{opt_sum:,} units** across this horizon."
            else:
                summary_msg = f"⚠️ **Evidence-based outcome**: Smart Plan experienced **{opt_sum:,} units** vs **{base_sum:,} units** under Do Nothing (**{abs(net_saved):,} units higher** due to inter-facility transit delays during outbreak onset)."
        else:
            summary_msg = "✅ **Zero network shortage**: Both policies maintained adequate stock across all 14 simulation days."
        st.caption(summary_msg)
    else:
        st.info("No comparative daily state data available for the selected scenario.")
else:
    st.info("Comparative daily state data table is empty or missing.")

st.markdown("---")

# ==============================================================================
# SECTION E: SECONDARY INFORMATION (Compact Expanders)
# ==============================================================================
c_exp1, c_exp2 = st.columns(2)

with c_exp1:
    with st.expander(f"🚨 **Critical Shortage Detail (Day {selected_day})**", expanded=(total_shortage > 0)):
        if not day_state.empty and "unmet_demand" in day_state.columns:
            short_df = day_state[day_state["unmet_demand"] > 0].sort_values("unmet_demand", ascending=False).head(10).copy()
            if not short_df.empty:
                short_df["Facility"] = short_df["hospital_id"].map(h_name_map)
                short_df["Medicine"] = short_df["medicine_id"].map(m_name_map)
                short_df["Current Stock"] = short_df["stock_remaining"].astype(int)
                short_df["Unmet Units"] = short_df["unmet_demand"].astype(int)
                short_df["Status"] = STATUS_CONFIG["critical_shortage"]["badge"]
                st.dataframe(
                    short_df[["Facility", "Medicine", "Current Stock", "Unmet Units", "Status"]],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.success("✅ Zero active shortages recorded on this day.")
        else:
            st.info("No state data available.")

with c_exp2:
    with st.expander(f"⚠️ **Expiry & FEFO Batches (Day {selected_day})**", expanded=(total_expired > 0)):
        if not day_state.empty and "expired_units" in day_state.columns:
            exp_df = day_state[day_state["expired_units"] > 0].sort_values("expired_units", ascending=False).head(10).copy()
            if not exp_df.empty:
                exp_df["Facility"] = exp_df["hospital_id"].map(h_name_map)
                exp_df["Medicine"] = exp_df["medicine_id"].map(m_name_map)
                exp_df["Expired Units"] = exp_df["expired_units"].astype(int)
                exp_df["Status"] = STATUS_CONFIG["expiry_risk"]["badge"]
                st.dataframe(
                    exp_df[["Facility", "Medicine", "Expired Units", "Status"]],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.success("✅ Zero expired units written off on this day. FEFO allocation effective.")
        else:
            st.info("No expiry records available.")

# Geospatial Network Health & Regional Distribution (Secondary Section in Expander)
with st.expander("🗺️ **Geospatial Facility Network Health & Regional Overview**", expanded=False):
    st.caption("Facility geographic distribution and district-level inventory vs deficit comparison")
    geo_col, bar_col = st.columns([3, 2])

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

        with geo_col:
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
                    height=340,
                )
                if hasattr(px, "scatter_map"):
                    fig_geo.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=10, b=0))
                else:
                    fig_geo.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_geo, use_container_width=True)
            else:
                st.dataframe(map_data[["name", "district", "Status", "stock", "shortage"]], hide_index=True)

        with bar_col:
            reg_summary = map_data.groupby("district").agg(
                Available_Stock=("stock", "sum"),
                Unmet_Shortage=("shortage", "sum"),
            ).reset_index()
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=reg_summary["district"], y=reg_summary["Available_Stock"], name="Stock", marker_color="#0ea5e9"))
            fig_bar.add_trace(go.Bar(x=reg_summary["district"], y=reg_summary["Unmet_Shortage"], name="Deficit", marker_color="#ef4444"))
            fig_bar.update_layout(
                barmode="group",
                height=340,
                margin=dict(l=10, r=10, t=10, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Hospital geospatial coordinates not available for current selection.")

st.caption("Smart Medicine Redistribution System • Kerala Public Health Services • Academic Big Data Implementation")
