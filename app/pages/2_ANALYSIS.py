"""ANALYSIS Page: Answers the core question 'What is happening now?'
Shows Total Stock, Shortage Risks, Surplus Stock, and Expiry Risks.
Classic tables, clean focused retro charts, and facility explorer.
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
    from app.data_manager import get_active_dataset, analyze_inventory, KERALA_HOSPITALS
except ImportError:
    from retro_theme import apply_retro_theme, retro_header, format_retro_chart
    from data_manager import get_active_dataset, analyze_inventory, KERALA_HOSPITALS

st.set_page_config(page_title="ANALYSIS — Medicine Inventory Health", page_icon="📊", layout="wide")
apply_retro_theme()

retro_header(
    title="INVENTORY ANALYSIS & RISK SURVEILLANCE",
    subtitle="Statewide Medicine Stock Status, Impending Shortages, Usable Surplus & Expiry Risks"
)

active_df = get_active_dataset()
if active_df.empty:
    st.warning("No inventory records found. Please load demo data or import records from the DATA page.")
    if st.button("Go to DATA Page"):
        st.switch_page("pages/1_DATA.py")
    st.stop()

analysis = analyze_inventory(active_df)

# ==============================================================================
# SECTION: WHAT IS HAPPENING NOW? (Section 11)
# ==============================================================================
st.markdown("### ❓ What is happening now?")
st.caption("Live aggregate status across all monitored government hospitals.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="TOTAL STOCK",
        value=f"{analysis['total_stock']:,} units",
        help="Total physical medicine inventory currently available across all facilities."
    )
with col2:
    shortage_count = analysis["shortage_count"]
    st.metric(
        label="SHORTAGE RISKS",
        value=f"{shortage_count} items",
        delta=f"-{shortage_count} deficits" if shortage_count > 0 else "Normal",
        delta_color="inverse",
        help="Medicines with less than 4 days of stock remaining or below safety stock target."
    )
with col3:
    surplus_count = analysis["surplus_count"]
    st.metric(
        label="SURPLUS STOCK",
        value=f"{surplus_count} items",
        delta="Available for Transfer",
        help="Medicine inventory in excess of 7-day demand plus local safety stock buffer."
    )
with col4:
    expiry_count = analysis["expiry_risk_count"]
    st.metric(
        label="EXPIRY RISKS",
        value=f"{expiry_count} items",
        delta="≤ 30 Days Shelf Life",
        delta_color="inverse",
        help="Medicines expiring within 30 days that risk being wasted if not redistributed."
    )

st.markdown("---")

# ==============================================================================
# THE THREE FOCUSED TABLES (Section 11)
# ==============================================================================
tab_short, tab_surp, tab_exp, tab_fac = st.tabs([
    "🚨 Shortage Medicines",
    "📦 Surplus Medicines",
    "⏳ Expiry Risks",
    "🏥 Facility Explorer"
])

with tab_short:
    st.markdown("#### Shortage Medicines")
    st.markdown("""
    > **Deficit Criteria**: Days Remaining < 4.0 days or current stock below required safety buffer.
    > Facilities in rural areas receive prioritized replenishment ranking.
    """)
    shortages_df = analysis["shortages_df"]
    if not shortages_df.empty:
        disp_short = shortages_df[[
            "medicine_name", "hospital_name", "current_stock", "daily_consumption", "days_remaining", "risk"
        ]].rename(columns={
            "medicine_name": "Medicine",
            "hospital_name": "Hospital",
            "current_stock": "Stock",
            "daily_consumption": "Daily Usage",
            "days_remaining": "Days Remaining",
            "risk": "Risk",
        })
        st.dataframe(disp_short, hide_index=True, use_container_width=True)
    else:
        st.success("No critical shortages detected across the network.")

with tab_surp:
    st.markdown("#### Surplus Medicines")
    st.markdown("""
    > **Surplus Criteria**: Inventory remaining after reserving 7 days of operational demand plus safety stock.
    > Available for redistribution to facilities facing deficits.
    """)
    surplus_df = analysis["surplus_df"]
    if not surplus_df.empty:
        disp_surp = surplus_df[[
            "medicine_name", "hospital_name", "surplus", "days_to_expiry", "action"
        ]].rename(columns={
            "medicine_name": "Medicine",
            "hospital_name": "Hospital",
            "surplus": "Surplus (Units)",
            "days_to_expiry": "Expiry (Days)",
            "action": "Action",
        })
        st.dataframe(disp_surp, hide_index=True, use_container_width=True)
    else:
        st.info("No surplus stock identified beyond local safety requirements.")

with tab_exp:
    st.markdown("#### Expiry Risks")
    st.markdown("""
    > **Expiry Criteria**: Batches with shelf life of 30 days or less.
    > Must be transferred to high-consumption facilities under First Expired, First Out (FEFO) rules.
    """)
    expiry_df = analysis["expiry_df"]
    if not expiry_df.empty:
        disp_exp = expiry_df[[
            "medicine_name", "hospital_name", "current_stock", "days_to_expiry", "risk"
        ]].rename(columns={
            "medicine_name": "Medicine",
            "hospital_name": "Hospital",
            "current_stock": "Stock",
            "days_to_expiry": "Days Until Expiry",
            "risk": "Risk",
        })
        st.dataframe(disp_exp, hide_index=True, use_container_width=True)
    else:
        st.success("No medicines approaching shelf life expiry within the 30-day window.")

with tab_fac:
    # Preserving Hospital Administrator local view
    st.markdown("#### Facility-Level Formulary Inspection")
    hosp_list = sorted(active_df["hospital_name"].unique().tolist())
    selected_hosp = st.selectbox("Select Hospital Facility to Audit", hosp_list, index=0)

    h_sub = active_df[active_df["hospital_name"] == selected_hosp].copy()
    if not h_sub.empty:
        h_info = h_sub.iloc[0]
        col_hi1, col_hi2, col_hi3 = st.columns(3)
        with col_hi1:
            st.metric("Total Facility Stock", f"{int(h_sub['current_stock'].sum()):,} units")
        with col_hi2:
            h_deficits = (h_sub["current_stock"] < h_sub["safety_stock"]).sum()
            st.metric("Facility Deficits", f"{h_deficits} items", delta="Deficits" if h_deficits > 0 else "Balanced", delta_color="inverse")
        with col_hi3:
            st.metric("Location Type", "Rural PHC" if h_info["is_rural"] else "Urban Facility")

        st.markdown("##### On-Hand Medicine Formulary:")
        h_sub["Days Remaining"] = (h_sub["current_stock"] / h_sub["daily_consumption"].clip(lower=1)).round(1)
        st.dataframe(
            h_sub[[
                "medicine_name", "category", "current_stock", "daily_consumption", "safety_stock", "Days Remaining", "days_to_expiry"
            ]].rename(columns={
                "medicine_name": "Medicine",
                "category": "Category",
                "current_stock": "Stock",
                "daily_consumption": "Daily Usage",
                "safety_stock": "Safety Stock",
                "days_to_expiry": "Expiry (Days)",
            }),
            hide_index=True,
            use_container_width=True,
        )

st.markdown("---")

# ==============================================================================
# FOCUSED RETRO CHARTS (Section 11)
# ==============================================================================
st.markdown("### 📊 Focused Visual Summaries")
st.caption("Clean, informative visualizations without unnecessary dashboard clutter.")

c_c1, c_c2 = st.columns(2)

with c_c1:
    # Chart 1: Stock vs Daily Usage across medicines
    med_agg = active_df.groupby("medicine_name").agg(
        total_stock=("current_stock", "sum"),
        total_usage=("daily_consumption", "sum"),
    ).reset_index().sort_values(by="total_stock", ascending=False).head(8)

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=med_agg["medicine_name"],
        y=med_agg["total_stock"],
        name="Current Stock (Units)",
        marker_color="#2b547e",
    ))
    fig1.add_trace(go.Bar(
        x=med_agg["medicine_name"],
        y=med_agg["total_usage"] * 7,
        name="7-Day Consumption Requirement",
        marker_color="#d97706",
    ))
    format_retro_chart(fig1, height=320, title="Top Medicines: Total Stock vs 7-Day Network Need")
    st.plotly_chart(fig1, use_container_width=True)

with c_c2:
    # Chart 2: Days of Stock Distribution by Facility
    hosp_agg = active_df.groupby("hospital_name").agg(
        avg_days=("current_stock", lambda x: np.mean(x / active_df.loc[x.index, "daily_consumption"].clip(lower=1)))
    ).reset_index().sort_values(by="avg_days")

    colors = ["#c5221f" if d < 4.0 else ("#2b547e" if d > 14.0 else "#16a34a") for d in hosp_agg["avg_days"]]
    fig2 = go.Figure(go.Bar(
        x=hosp_agg["hospital_name"],
        y=hosp_agg["avg_days"].round(1),
        marker_color=colors,
        text=hosp_agg["avg_days"].round(1),
        textposition="outside",
    ))
    fig2.add_hline(y=4.0, line_dash="dash", line_color="#c5221f", annotation_text="Shortage Risk Line (4 Days)")
    fig2.add_hline(y=14.0, line_dash="dash", line_color="#16a34a", annotation_text="Surplus Margin Line (14 Days)")
    format_retro_chart(fig2, height=320, title="Average Days of Stock per Hospital Facility")
    st.plotly_chart(fig2, use_container_width=True)
