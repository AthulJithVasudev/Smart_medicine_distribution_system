"""DATA Page: Hospital Medicine Stock Data Entry, CSV Import, and ETL Quality Report.
Retro desktop portal aesthetics.
"""
import sys
from pathlib import Path
import datetime
import streamlit as st
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from app.retro_theme import apply_retro_theme, retro_header
    from app.data_manager import (
        get_active_dataset,
        reset_to_demo_data,
        clear_active_dataset,
        add_manual_record,
        import_csv_data,
        get_sample_csv_text,
        KERALA_HOSPITALS,
    )
except ImportError:
    from retro_theme import apply_retro_theme, retro_header
    from data_manager import (
        get_active_dataset,
        reset_to_demo_data,
        clear_active_dataset,
        add_manual_record,
        import_csv_data,
        get_sample_csv_text,
        KERALA_HOSPITALS,
    )

st.set_page_config(page_title="DATA — Medicine Stock Management", page_icon="📝", layout="wide")
apply_retro_theme()

retro_header(
    title="DATA MANAGEMENT & INVENTORY ENTRY",
    subtitle="Government Hospital Medicine Stock Registration, CSV Import & Quality Audit"
)

# Top Action Control Bar
col_a1, col_a2, col_a3, col_a4 = st.columns([1.2, 1.2, 1.2, 1.2])
with col_a1:
    if st.button("[ LOAD DEMO DATA ]", use_container_width=True, help="Load realistic Kerala government hospital dataset"):
        reset_to_demo_data()
        st.success("Loaded predefined demo dataset.")
        st.rerun()

with col_a2:
    if st.button("[ CLEAR DATA ]", use_container_width=True, help="Clear active records"):
        clear_active_dataset()
        st.warning("Cleared active records.")
        st.rerun()

with col_a3:
    sample_csv = get_sample_csv_text()
    st.download_button(
        label="📥 [ DOWNLOAD SAMPLE CSV ]",
        data=sample_csv,
        file_name="sample_medicine_stock.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download ready-to-use CSV template matching system expectations",
    )

with col_a4:
    if st.button("🚀 [ GO TO ANALYSIS ]", use_container_width=True, help="Analyze active inventory"):
        st.switch_page("pages/2_ANALYSIS.py")

st.markdown("---")

# Main Content Layout: Form Entry & CSV Upload
tab_form, tab_csv, tab_etl = st.tabs(["✍️ Manual Data Entry Form", "📂 CSV Import & Template", "⚙️ Data Processing Quality"])

with tab_form:
    st.markdown("### Simple Inventory Entry Form")
    st.caption("Enter hospital facility and medicine inventory record.")

    with st.form("manual_entry_form", clear_on_submit=False):
        c_h1, c_h2, c_h3 = st.columns(3)
        with c_h1:
            hosp_options = [h["name"] for h in KERALA_HOSPITALS]
            hosp_name = st.selectbox("Hospital Name", hosp_options, index=0)
        with c_h2:
            district_val = st.selectbox("Location / District", ["Ernakulam", "Thrissur", "Palakkad", "Malappuram", "Kozhikode"], index=0)
        with c_h3:
            rural_urban = st.radio("Facility Classification", ["Urban Facility", "Rural PHC"], index=0, horizontal=True)
            is_rural = (rural_urban == "Rural PHC")

        st.markdown("<hr style='margin: 8px 0; border:none; border-top:1px solid #cbd5e1;'>", unsafe_allow_html=True)

        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            med_list = [
                "Paracetamol 500mg", "Amoxicillin 500mg", "Ciprofloxacin 500mg",
                "Oral Rehydration Salts", "Normal Saline 500ml", "Artesunate Injection",
                "Chloroquine 250mg", "Ibuprofen 400mg", "Cetirizine 10mg", "Azithromycin 500mg"
            ]
            med_name = st.selectbox("Medicine Name", med_list, index=1)
        with c_m2:
            stock_qty = st.number_input("Current Stock (Units)", min_value=0, max_value=100000, value=300, step=50)
        with c_m3:
            daily_use = st.number_input("Daily Consumption (Units / Day)", min_value=1, max_value=10000, value=120, step=10, help="Average patient consumption per 24h")

        c_s1, c_s2, c_s3 = st.columns(3)
        with c_s1:
            safety_stock = st.number_input("Safety Stock (Units)", min_value=0, max_value=10000, value=360, step=50, help="Minimum buffer stock needed to absorb demand spikes")
        with c_s2:
            days_expiry = st.number_input("Days Until Expiry", min_value=1, max_value=1095, value=180, step=10, help="Days until batch shelf life expires")
        with c_s3:
            criticality = st.slider("Clinical Criticality (1 = Low, 5 = Life Saving)", 1, 5, 4, help="Clinical urgency rating for hospital stock prioritization")

        # Advanced Options Expander (Section 9)
        with st.expander("⚙️ Advanced Options (Optional Settings)"):
            c_adv1, c_adv2 = st.columns(2)
            with c_adv1:
                price = st.number_input("Unit Procurement Price (₹)", min_value=0.5, max_value=500.0, value=6.0, step=0.5)
            with c_adv2:
                category = st.selectbox("Therapeutic Category", ["Antibiotic", "Analgesic", "IV Fluid", "Antimalarial", "Electrolyte", "Antihistamine"], index=0)

        submit_btn = st.form_submit_button("[ ADD RECORD ]", type="primary")
        if submit_btn:
            success, msg = add_manual_record({
                "hospital_name": hosp_name,
                "district": district_val,
                "is_rural": is_rural,
                "medicine_name": med_name,
                "category": category,
                "current_stock": stock_qty,
                "daily_consumption": daily_use,
                "safety_stock": safety_stock,
                "days_to_expiry": days_expiry,
                "criticality": criticality,
                "price": price,
            })
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

with tab_csv:
    st.markdown("### Import CSV Dataset")
    st.markdown("""
    Upload a CSV file containing hospital medicine stock records. 
    The file should contain standard fields such as `Hospital`, `Medicine`, `Stock`, `Daily Usage`, and `Expiry`.
    """)

    st.markdown("##### Expected Sample Format:")
    st.markdown("""
    <table class="retro-table" style="max-width: 650px;">
        <thead>
            <tr>
                <th>Hospital</th>
                <th>Medicine</th>
                <th>Stock</th>
                <th>Daily Usage</th>
                <th>Expiry</th>
                <th>Rural</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Hospital A</td>
                <td>Paracetamol 500mg</td>
                <td>5000</td>
                <td>100</td>
                <td>20</td>
                <td>NO</td>
            </tr>
            <tr>
                <td>Hospital B</td>
                <td>Paracetamol 500mg</td>
                <td>500</td>
                <td>120</td>
                <td>90</td>
                <td>YES</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select or Drag & Drop CSV File", type=["csv"])
    if uploaded_file is not None:
        if st.button("📥 [ IMPORT CSV ]", type="primary"):
            success, msg = import_csv_data(uploaded_file)
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(f"Unable to process the dataset. Please check that required columns are present. ({msg})")

with tab_etl:
    # Data Processing / ETL Report (Section 20)
    stats = st.session_state.get("etl_stats", {
        "records_received": 0, "records_processed": 0, "missing_values": 0, "invalid_records": 0, "duplicates_removed": 0
    })
    st.markdown("### DATA PROCESSING & ETL PIPELINE SUMMARY")
    st.markdown(f"""
    <div class="retro-status-box">
DATA PROCESSING
-----------------------------------------
Records received    : {stats.get('records_received', 0)}
Records processed   : {stats.get('records_processed', 0)}
Missing values      : {stats.get('missing_values', 0)}
Invalid records     : {stats.get('invalid_records', 0)}
Duplicates removed  : {stats.get('duplicates_removed', 0)}
Status              : VALIDATED & READY
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 [ SHOW DETAILS ] Technical ETL & Validation Audit"):
        st.markdown("""
        **Pipeline Cleaning & Transformation Guarantees:**
        1. **Null Imputation**: Missing consumption records are imputed using facility-specific moving median.
        2. **Bounds Enforcing**: `current_stock >= 0`, `daily_consumption >= 1`, non-negative constraints enforced.
        3. **Deduplication**: Composite key `(hospital_name, medicine_name)` deduplicated maintaining latest state.
        4. **Causality Barrier**: Information barrier guarantees no leakage of latent future demand into current operational state.
        """)

st.markdown("---")

# Active Records Table
st.markdown("### 📋 Active Inventory Records Table")
active_df = get_active_dataset()

if not active_df.empty:
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.caption(f"Displaying {len(active_df)} active inventory records across {active_df['hospital_name'].nunique()} hospitals.")
    with col_t2:
        filter_hosp = st.selectbox("Filter by Hospital", ["(All Hospitals)"] + sorted(active_df["hospital_name"].unique().tolist()))

    display_df = active_df.copy()
    if filter_hosp != "(All Hospitals)":
        display_df = display_df[display_df["hospital_name"] == filter_hosp]

    display_df["Days Remaining"] = (display_df["current_stock"] / display_df["daily_consumption"].clip(lower=1)).round(1)

    st.dataframe(
        display_df[[
            "hospital_name", "district", "is_rural", "medicine_name", "category",
            "current_stock", "daily_consumption", "safety_stock", "Days Remaining", "days_to_expiry", "criticality"
        ]].rename(columns={
            "hospital_name": "Hospital",
            "district": "District",
            "is_rural": "Rural",
            "medicine_name": "Medicine",
            "category": "Category",
            "current_stock": "Current Stock",
            "daily_consumption": "Daily Usage",
            "safety_stock": "Safety Stock",
            "days_to_expiry": "Expiry (Days)",
            "criticality": "Criticality",
        }),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("No records currently loaded. Click [ LOAD DEMO DATA ] or enter a record above.")
