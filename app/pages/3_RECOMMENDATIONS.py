"""RECOMMENDATIONS Page: Recommended Medicine Transfers, Why? Audit Log,
Before vs After Impact Simulation, Simple Route Map, and Policy Benchmarks.
Retro desktop portal aesthetics.
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from app.retro_theme import apply_retro_theme, retro_header, format_retro_chart
    from app.data_loader import load_serving_table
    from app.data_manager import (
        get_active_dataset,
        compute_redistributions,
        compute_before_after_simulation,
        KERALA_HOSPITALS,
        HOSPITALS_NAME_LOOKUP,
    )
except ImportError:
    from retro_theme import apply_retro_theme, retro_header, format_retro_chart
    from data_loader import load_serving_table
    from data_manager import (
        get_active_dataset,
        compute_redistributions,
        compute_before_after_simulation,
        KERALA_HOSPITALS,
        HOSPITALS_NAME_LOOKUP,
    )

st.set_page_config(page_title="RECOMMENDATIONS — Medicine Transfers", page_icon="🚚", layout="wide")
apply_retro_theme()

retro_header(
    title="RECOMMENDED MEDICINE TRANSFERS",
    subtitle="Mathematical Optimization Directives, Decision Justification & Before/After Impact Simulation"
)

active_df = get_active_dataset()
if active_df.empty:
    st.warning("No inventory records found. Please load data on the DATA page.")
    st.stop()

recommendations = compute_redistributions(active_df)
sim_impact = compute_before_after_simulation(active_df, recommendations)

# ==============================================================================
# SECTION 16: SHOW MOST IMPORTANT RESULT FIRST
# ==============================================================================
if recommendations:
    primary = recommendations[0]
    st.markdown("### 🏆 Top Recommended Medicine Transfer Directive")
    st.markdown(f"""
    <div class="hero-transfer-box">
        <div class="hero-transfer-header">
            <span>RECOMMENDED TRANSFER</span>
            <span class="badge-retro badge-retro-danger">PRIORITY: {primary['priority']}</span>
        </div>
        <div class="hero-transfer-grid">
            <div class="hero-transfer-field">
                <div class="hero-transfer-label">Medicine</div>
                <div class="hero-transfer-value">{primary['medicine']}</div>
            </div>
            <div class="hero-transfer-field">
                <div class="hero-transfer-label">FROM (Source Surplus)</div>
                <div class="hero-transfer-value">{primary['from_hospital']}</div>
            </div>
            <div class="hero-transfer-field">
                <div class="hero-transfer-label">TO (Destination Shortage)</div>
                <div class="hero-transfer-value">{primary['to_hospital']}</div>
            </div>
            <div class="hero-transfer-field">
                <div class="hero-transfer-label">QUANTITY</div>
                <div class="hero-transfer-value">{primary['quantity']:,} units</div>
            </div>
            <div class="hero-transfer-field">
                <div class="hero-transfer-label">TRANSIT & DISTANCE</div>
                <div class="hero-transfer-value">{primary['distance_km']} km ({primary['transit_days']} transit day)</div>
            </div>
            <div class="hero-transfer-field">
                <div class="hero-transfer-label">ESTIMATED LOGISTICS COST</div>
                <div class="hero-transfer-value">₹{primary['transport_cost']:,}</div>
            </div>
        </div>
        <div style="font-size:12px; margin-top:8px; color:#1e3a5f;">
            <strong>REASON:</strong> {primary['reason']}
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.success("✅ Network inventory is currently balanced. No inter-hospital transfers required.")

st.markdown("---")

# ==============================================================================
# TABS: DIRECTIVES TABLE, WHY? AUDIT, BEFORE/AFTER SIMULATION, SIMPLE MAP
# ==============================================================================
tab_table, tab_why, tab_sim, tab_map, tab_bench = st.tabs([
    "📋 All Recommendations",
    "🔍 Why? Decision Justification",
    "⚖️ Before / After Simulation",
    "🗺️ Simple Transfer Map",
    "📊 Policy Benchmarks"
])

# 1. RECOMMENDATIONS TABLE (Section 16)
with tab_table:
    st.markdown("#### All Recommended Transfer Orders")
    if recommendations:
        tbl_data = []
        for idx, r in enumerate(recommendations, 1):
            tbl_data.append({
                "Order #": f"TR-{idx:02d}",
                "Priority": r["priority"],
                "Medicine": r["medicine"],
                "From": r["from_hospital"],
                "To": r["to_hospital"],
                "Quantity": f"{r['quantity']:,} units",
                "Distance": f"{r['distance_km']} km",
                "Cost": f"₹{r['transport_cost']}",
                "Reason": r["reason"],
            })
        st.dataframe(pd.DataFrame(tbl_data), hide_index=True, use_container_width=True)
    else:
        st.info("No active transfer recommendations.")

# 2. ADD "WHY?" TO EVERY RECOMMENDATION (Section 17)
with tab_why:
    st.markdown("#### Decision Justification Engine (Why was this recommended?)")
    st.caption("Inspect exact inventory, demand, safety stock, and distance values used in the mathematical optimization decision.")

    if recommendations:
        for idx, r in enumerate(recommendations, 1):
            w = r["why_details"]
            with st.expander(f"Order #{idx:02d} — {r['medicine']}: {r['quantity']} units from {r['from_hospital']} to {r['to_hospital']} [ WHY? ]", expanded=(idx <= 2)):
                st.markdown(f"""
                <div class="retro-status-box" style="margin:0;">
WHY WAS THIS TRANSFER RECOMMENDED?
-------------------------------------------------------------
Medicine            : {w['medicine']}
Destination Hospital: {w['to_hosp']}
Destination stock   : {w['dest_stock']:,} units
Predicted demand    : {w['predicted_demand']:,} units
Safety stock        : {w['safety_stock']:,} units
Expected shortage   : {w['expected_shortage']:,} units

Source Hospital     : {w['from_hosp']}
Source surplus      : {w['source_surplus']:,} units
Expiry risk         : {w['expiry_risk']} ({w['source_expiry_days']} days remaining shelf life)
Transit distance    : {w['distance_km']} km ({w['transit_days']} transit day)
Facility Priority   : {'RURAL PHC (Higher Critical Weight)' if w['is_rural'] else 'Standard Facility'}

THEREFORE:
Transfer {w['quantity']:,} units from {w['from_hosp']} to {w['to_hosp']}.
Dest. shortage is mitigated before stockout occurs; Source surplus is consumed before expiry.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No recommendations to justify.")

# 3. BEFORE / AFTER SIMULATION (Section 18)
with tab_sim:
    st.markdown("#### Before / After Redistribution Simulation")
    st.caption("Quantitative impact assessment: calculated dynamically from the inventory and solved transfer schedule.")

    b = sim_impact["before"]
    a = sim_impact["after"]
    imp = sim_impact["impact"]

    col_sim_b, col_sim_a = st.columns(2)
    with col_sim_b:
        st.markdown(f"""
        <div class="retro-panel" style="border-top: 3px solid #c5221f;">
            <div class="retro-panel-header" style="color:#c5221f;">BEFORE REDISTRIBUTION</div>
            <div style="font-family:'Courier New', monospace; font-size:14px; line-height:2.0;">
                Critical shortages : <strong>{b['critical_shortages']}</strong> facilities<br>
                Shortage deficit   : <strong>{b['shortage_units']:,}</strong> units<br>
                Expiry-risk stock  : <strong>{b['expiry_risk_stock']:,}</strong> units<br>
                Surplus stock      : <strong>{b['surplus_stock']:,}</strong> units
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_sim_a:
        st.markdown(f"""
        <div class="retro-panel" style="border-top: 3px solid #16a34a;">
            <div class="retro-panel-header" style="color:#16a34a;">AFTER REDISTRIBUTION</div>
            <div style="font-family:'Courier New', monospace; font-size:14px; line-height:2.0;">
                Critical shortages : <strong>{a['critical_shortages']}</strong> facilities<br>
                Shortage deficit   : <strong>{a['shortage_units']:,}</strong> units<br>
                Expiry-risk stock  : <strong>{a['expiry_risk_stock']:,}</strong> units<br>
                Surplus stock      : <strong>{a['surplus_stock']:,}</strong> units
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##### System Impact Summary Metrics:")
    c_k1, c_k2, c_k3, c_k4 = st.columns(4)
    with c_k1:
        st.metric("Shortage Reduction", f"{imp['shortage_reduction_pct']}%", delta="Net Deficits Resolved")
    with c_k2:
        st.metric("Potential Wastage Reduction", f"{imp['wastage_reduction_pct']}%", delta="Expiring Stock Saved")
    with c_k3:
        st.metric("Medicine Redistributed", f"{imp['medicine_redistributed']:,} units")
    with c_k4:
        st.metric("Transport Cost Incurred", f"₹{imp['transport_cost']:,}")

# 4. SIMPLE MAP (Section 19)
with tab_map:
    st.markdown("#### Simple Geographic Transfer Routes Map")
    st.caption("Showing hospitals, sources (blue), destinations (red), and recommended logistics paths.")

    # Build simple map points and transfer lines
    sources = set(r["from_hospital"] for r in recommendations)
    destinations = set(r["to_hospital"] for r in recommendations)

    map_rows = []
    for h in KERALA_HOSPITALS:
        name = h["name"]
        role = "Neutral Facility"
        if name in sources and name in destinations:
            role = "Transit Hub"
        elif name in sources:
            role = "Source Hospital (Surplus)"
        elif name in destinations:
            role = "Destination Hospital (Shortage)"

        map_rows.append({
            "Hospital": name,
            "District": h["district"],
            "Latitude": h["latitude"],
            "Longitude": h["longitude"],
            "Role": role,
            "Type": "Rural PHC" if h["is_rural"] else "Urban Facility",
        })

    geo_df = pd.DataFrame(map_rows)

    color_map = {
        "Source Hospital (Surplus)": "#2563eb",
        "Destination Hospital (Shortage)": "#dc2626",
        "Transit Hub": "#9333ea",
        "Neutral Facility": "#64748b",
    }

    # Scatter plot on coordinates
    fig_map = px.scatter(
        geo_df,
        x="Longitude",
        y="Latitude",
        color="Role",
        text="Hospital",
        hover_data=["District", "Type"],
        color_discrete_map=color_map,
        title="Hospital Locations & Active Transfer Route Network",
    )
    fig_map.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1, color="#1e293b")))

    # Add transfer line routes
    for r in recommendations:
        src = HOSPITALS_NAME_LOOKUP.get(r["from_hospital"])
        dst = HOSPITALS_NAME_LOOKUP.get(r["to_hospital"])
        if src and dst:
            fig_map.add_trace(go.Scatter(
                x=[src["longitude"], dst["longitude"]],
                y=[src["latitude"], dst["latitude"]],
                mode="lines+markers",
                line=dict(color="#2b547e", width=2, dash="dash"),
                name=f"{r['medicine']} ({r['quantity']}u)",
                hoverinfo="text",
                hovertext=f"Transfer: {r['quantity']} units of {r['medicine']} ({r['distance_km']} km)",
                showlegend=False,
            ))

    format_retro_chart(fig_map, height=450)
    st.plotly_chart(fig_map, use_container_width=True)

# 5. BENCHMARK POLICY COMPARISON (Preserved Scenario Comparison)
with tab_bench:
    st.markdown("#### Mathematical Optimizer vs Heuristic Benchmarks")
    eval_opt_df = load_serving_table("evaluation_optimizer")
    if not eval_opt_df.empty:
        st.markdown("##### Multi-Scenario Cross-Policy Performance:")
        st.dataframe(
            eval_opt_df[[
                "scenario", "policy", "seeds_count", "wastage_units_mean", "stockout_days_mean", "fill_rate_mean", "transport_cost_mean"
            ]].rename(columns={
                "scenario": "Scenario",
                "policy": "Policy Architecture",
                "seeds_count": "Simulated Seeds",
                "wastage_units_mean": "Mean Wastage (Units)",
                "stockout_days_mean": "Mean Stockout Days",
                "fill_rate_mean": "Mean Fill Rate",
                "transport_cost_mean": "Mean Transport Cost (₹)",
            }),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Optimizer benchmark table is ready in serving data.")
