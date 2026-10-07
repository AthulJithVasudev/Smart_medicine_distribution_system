"""About page documenting system architecture, methodology, synthetic data disclaimer, and stack."""
import streamlit as st

st.set_page_config(page_title="About System Architecture", page_icon="ℹ️", layout="wide")

st.title("About the Smart Medicine Redistribution System")
st.caption("System Architecture v4: Cloud-Hosted, Scenario-Driven Multi-Hospital Medicine Redistribution")

# PROMINENT DISCLAIMER (CRITICAL ARCHITECTURE REQUIREMENT)
st.warning(
    "⚠️ **ACADEMIC & SYNTHETIC DATA DISCLAIMER**:\n\n"
    "All hospital facilities, patient demand records, inventory levels, OPD visits, and batch numbers displayed in this system "
    "are **SYNTHETIC DATA** generated deterministically by the discrete-event simulator engine. "
    "These values are produced for research and operational simulation purposes and do **NOT** represent real, official government hospital statistics. "
    "Data provenance is strictly tagged across all records (`SIMULATED`, `DERIVED`, `OFFICIAL`)."
)

st.markdown("---")

# Project Objective
st.subheader("1. 🎯 Project Objective & Core Business Problem")
st.markdown("""
Government hospital networks across regions frequently face asymmetric inventory imbalances:
- **Hospital A** has an excess surplus of critical medicines approaching expiration.
- **Hospital B** (often a rural or peripheral Primary Health Centre) is facing a stockout.

The goal of this system is to determine **whether, when, what medicine, how much, and from where** to transfer items,
minimizing total wastage and stockouts while accounting for transit times, shelf lives, and logistics costs.
""")

st.markdown("---")

# Logical Architecture Diagram
st.subheader("2. 🏗️ Logical Architecture Pipeline")
st.code("""
SIMULATOR (Stateful discrete-event dynamics)
       ↓
RAW PARQUET (Partitioned by report_day)
       ↓
PYSPARK ETL (Deduplication, imputation, window lags, unit normalization)
       ↓
CURATED PARQUET (Features strictly causal at time t)
       ↓
DUCKDB (Serverless analytical SQL layer)
       ↓
ML + FORECASTING (SARIMAX dedicated + Global Trees + K-Means + Isolation Forest)
       ↓
EVALUATION + MLFLOW (Walk-forward cross validation & Champion/Challenger registry)
       ↓
OR-TOOLS OPTIMIZER (Mixed-Integer Linear Program with receiving caps & FEFO tie-break)
       ↓
SERVING PARQUET (12 high-performance published snapshot tables)
       ↓
STREAMLIT DASHBOARD (Zero-compute presentation layer)
""", language="text")

st.markdown("---")

# Key Technological Pillars
st.subheader("3. ⚙️ Technology Stack")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    - **Simulation Core**: NumPy SeedSequence (independent streams for demand, supply, faults, observation)
    - **ETL & Big Data**: PySpark local-mode (8 shuffle partitions), DuckDB SQL engine
    - **Storage Format**: Apache Parquet with Snappy compression
    - **Forecasting & ML**: statsmodels SARIMAX, Scikit-Learn HistGradientBoosting & RandomForest
    """)
with c2:
    st.markdown("""
    - **Optimization**: Google OR-Tools (CBC / SCIP / GLOP Mixed Integer Linear Programming)
    - **Experiment Tracking**: Local MLflow with SQLite backend
    - **User Interface**: Streamlit with Plotly and Mapbox geospatial visualization
    - **Automation**: GitHub Actions scheduled CI/CD runners (Java 17 + Python 3.12)
    """)

st.markdown("---")

# Constraints & Information Barrier
st.subheader("4. 🛡️ Information Barrier & Causality Guarantees")
st.markdown("""
1. **Information Barrier**: Latent true demand, outbreak flags, and hidden unmet shortages are strictly hidden from ML models. The models receive only observation-model output with reporting noise, delays, and demand censoring.
2. **Censored Demand**: `observed_consumption = min(true_demand, available_stock)`.
3. **Mandatory Receiving Cap**: Destination facilities cannot receive more than their forecast deficit; excess stock is never dumped on receiving clinics.
4. **FEFO Allocation**: Batches are strictly consumed and transferred using First Expired, First Out priority.
""")

st.markdown("---")

# Limitations
st.subheader("5. 📌 System Limitations")
st.markdown("""
- Geospatial routing currently estimates road network distance via Haversine great-circle distance with road curvature factor (1.2x - 1.5x for rural roads) rather than live GPS turn-by-turn routing.
- Cold-chain temperature breaches during transit are modeled as discrete failure rates rather than continuous ambient sensor streams.
""")
