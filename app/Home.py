"""Government Health Department Command Center Dashboard."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Ensure repository root and app directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from app.data_loader import load_serving_table
except ImportError:
    from data_loader import load_serving_table

st.set_page_config(
    page_title="Government Health Department Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-seen {
        background-color: #065f46;
        color: #34d399;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
    }
    .badge-unseen {
        background-color: #831843;
        color: #f472b6;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# Load Published Serving Tables
catalog_df = load_serving_table("scenario_catalog")
hospitals_df = load_serving_table("hospitals")
medicines_df = load_serving_table("medicines")
state_df = load_serving_table("state_daily")
transfers_df = load_serving_table("transfers")
kpis_df = load_serving_table("kpis_daily")
runs_df = load_serving_table("runs")

if state_df.empty or hospitals_df.empty:
    st.error("Serving data tables not found. Please run: `python -m pipeline.export --profile dev`")
    st.stop()

# Sidebar Controls
st.sidebar.title("🏥 Network Controls")
scenarios_available = catalog_df["name"].tolist() if not catalog_df.empty else ["normal"]
selected_scenario = st.sidebar.selectbox("Select Scenario", scenarios_available, index=0)

policies_available = ["optimizer", "greedy", "no_transfer"]
selected_policy = st.sidebar.selectbox("Active Redistribution Policy", policies_available, index=0)

# Scenario metadata
sc_meta = catalog_df[catalog_df["name"] == selected_scenario].iloc[0] if not catalog_df.empty else {}
sc_type = sc_meta.get("type", "seen")
sc_badge = f'<span class="badge-seen">SEEN CONDITION</span>' if sc_type == "seen" else f'<span class="badge-unseen">UNSEEN STRESS TEST</span>'
st.sidebar.markdown(f"**Condition Type**: {sc_badge}", unsafe_allow_html=True)
st.sidebar.caption(sc_meta.get("description", "Standard operating environment."))

# Filter data
scenario_state = state_df[(state_df["scenario"] == selected_scenario) & (state_df["policy"] == selected_policy)]
if scenario_state.empty:
    scenario_state = state_df

max_day = int(scenario_state["day"].max()) if not scenario_state.empty else 13
selected_day = st.sidebar.slider("Simulated Operational Day", min_value=0, max_value=max_day, value=min(7, max_day))

day_state = scenario_state[scenario_state["day"] == selected_day]
if day_state.empty:
    day_state = scenario_state

# Main Header
st.title("Smart Medicine Redistribution System")
st.subheader("Government Health Department — State Command Center")

# Top Level KPI Row
total_stock = int(day_state["stock_remaining"].sum()) if "stock_remaining" in day_state.columns else 0
total_shortage = int(day_state["unmet_demand"].sum()) if "unmet_demand" in day_state.columns else 0
total_expired = int(day_state["expired_units"].sum()) if "expired_units" in day_state.columns else 0
total_demand = int(day_state["actual_demand"].sum()) if "actual_demand" in day_state.columns else 1
dispensed = max(0, total_demand - total_shortage)
fill_rate = (dispensed / total_demand) * 100.0 if total_demand > 0 else 100.0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Network Stock Available</div>
        <div class="metric-val">{total_stock:,} <span style="font-size:14px;color:#94a3b8">units</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    shortage_color = "#ef4444" if total_shortage > 0 else "#10b981"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Unmet Shortage Units</div>
        <div class="metric-val" style="color:{shortage_color}">{total_shortage:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Expired Write-offs</div>
        <div class="metric-val" style="color:#f59e0b">{total_expired:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    rate_color = "#10b981" if fill_rate >= 95.0 else ("#f59e0b" if fill_rate >= 80.0 else "#ef4444")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Network Fill Rate</div>
        <div class="metric-val" style="color:{rate_color}">{fill_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    # Estimate total transport cost incurred
    day_transfers = transfers_df[transfers_df["day"] == selected_day] if not transfers_df.empty else pd.DataFrame()
    est_cost = len(day_transfers) * 100.0 + (day_transfers["distance_km"].sum() * 2.5 if not day_transfers.empty else 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Transport Cost Incurred</div>
        <div class="metric-val" style="color:#a855f7">₹{est_cost:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Geospatial Network Health Map & Regional Breakdown
c_map, c_reg = st.columns([3, 2])

with c_map:
    st.markdown("### 🗺️ Hospital Stock-Health & Geographical Distribution")
    h_summary = day_state.groupby("hospital_id").agg(
        stock=("stock_remaining", "sum"),
        shortage=("unmet_demand", "sum"),
    ).reset_index()

    map_df = hospitals_df.merge(h_summary, on="hospital_id", how="left").fillna(0)
    map_df["health_status"] = np.where(
        map_df["shortage"] > 0, "Stockout Alert",
        np.where(map_df["stock"] < 200, "Low Inventory", "Adequate Surplus")
    )
    map_df["color"] = map_df["health_status"].map({
        "Stockout Alert": "#ef4444",
        "Low Inventory": "#f59e0b",
        "Adequate Surplus": "#10b981",
    })

    map_fn = getattr(px, "scatter_map", getattr(px, "scatter_mapbox", None))
    if map_fn is not None:
        fig_map = map_fn(
            map_df,
            lat="latitude",
            lon="longitude",
            color="health_status",
            size="bed_count",
            hover_name="name",
            hover_data={"hospital_type": True, "stock": True, "shortage": True, "latitude": False, "longitude": False},
            color_discrete_map={"Stockout Alert": "#ef4444", "Low Inventory": "#f59e0b", "Adequate Surplus": "#10b981"},
            zoom=7.5,
            height=450,
            title=f"Hospital Status on Day {selected_day}",
        )
        if hasattr(px, "scatter_map"):
            fig_map.update_layout(
                map_style="carto-darkmatter",
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
        else:
            fig_map.update_layout(
                mapbox_style="carto-darkmatter",
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
        st.plotly_chart(fig_map, use_container_width=True)

with c_reg:
    st.markdown("### 📊 Regional Inventory Health by District")
    reg_df = map_df.groupby("district").agg(
        total_stock=("stock", "sum"),
        total_shortage=("shortage", "sum"),
    ).reset_index()

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Bar(
        x=reg_df["district"],
        y=reg_df["total_stock"],
        name="Surplus Stock",
        marker_color="#38bdf8",
    ))
    fig_reg.add_trace(go.Bar(
        x=reg_df["district"],
        y=reg_df["total_shortage"],
        name="Unmet Need",
        marker_color="#ef4444",
    ))
    fig_reg.update_layout(
        barmode="group",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="District",
        yaxis_title="Medicine Units",
    )
    st.plotly_chart(fig_reg, use_container_width=True)

st.markdown("---")

# Recommended Transfers & Expiring Medicines
c_tr, c_exp = st.columns([3, 2])

with c_tr:
    st.markdown(f"### 🚚 Recommended Inter-Hospital Transfers (Day {selected_day})")
    day_tr = transfers_df[transfers_df["day"] == selected_day] if not transfers_df.empty else pd.DataFrame()
    if not day_tr.empty:
        # Join hospital and medicine names
        h_names = hospitals_df.set_index("hospital_id")["name"].to_dict()
        m_names = medicines_df.set_index("medicine_id")["name"].to_dict()
        disp_tr = day_tr.copy()
        disp_tr["From Hospital"] = disp_tr["from_hospital"].map(h_names)
        disp_tr["To Destination"] = disp_tr["to_hospital"].map(h_names)
        disp_tr["Medicine"] = disp_tr["medicine_id"].map(m_names)
        disp_tr["Quantity"] = disp_tr["quantity"].astype(int)
        disp_tr["Transit (Days)"] = disp_tr["transit_days"]
        disp_tr["Distance (km)"] = disp_tr["distance_km"].round(1)

        st.dataframe(
            disp_tr[["From Hospital", "To Destination", "Medicine", "Quantity", "Transit (Days)", "Distance (km)"]],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No transfers recommended for Day " + str(selected_day) + " (Network inventory balanced).")

with c_exp:
    st.markdown("### ⚠️ Medicines at Imminent Expiry Risk")
    exp_summary = day_state[day_state["expired_units"] > 0]
    if not exp_summary.empty:
        m_names = medicines_df.set_index("medicine_id")["name"].to_dict()
        exp_table = exp_summary.groupby("medicine_id")["expired_units"].sum().reset_index()
        exp_table["Medicine Name"] = exp_table["medicine_id"].map(m_names)
        exp_table = exp_table.sort_values("expired_units", ascending=False).rename(columns={"expired_units": "Units Expired"})
        st.dataframe(exp_table[["Medicine Name", "Units Expired"]], hide_index=True, use_container_width=True)
    else:
        st.success("Zero expired units written off on Day " + str(selected_day) + ". FEFO policy holding firm.")

# Hospital Clusters Breakdown
st.markdown("---")
st.markdown("### 🏥 Facility Clustering Analysis (K-Means)")
if "cluster_label" in hospitals_df.columns:
    c_summary = hospitals_df.groupby("cluster_label").agg(
        count=("hospital_id", "count"),
        avg_beds=("bed_count", "mean"),
        rural_share=("is_rural", "mean"),
    ).reset_index()
    c_summary["avg_beds"] = c_summary["avg_beds"].round(1)
    c_summary["rural_share"] = (c_summary["rural_share"] * 100.0).round(1).astype(str) + "%"
    st.dataframe(
        c_summary.rename(columns={
            "cluster_label": "Operational Cluster",
            "count": "Facilities Count",
            "avg_beds": "Average Bed Capacity",
            "rural_share": "Rural Facility Share",
        }),
        hide_index=True,
        use_container_width=True,
    )
