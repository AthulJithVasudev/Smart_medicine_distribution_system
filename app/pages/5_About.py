"""About page: 2-minute quickstart guide, project glossary, methodology, and limitations."""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.data_loader import load_serving_table, render_sample_banner
from app.labels import GLOSSARY, SCENARIO_INFO, POLICY_LABELS, POLICY_DESCRIPTIONS
from app.state import render_sidebar_filters

st.set_page_config(page_title="About & System Architecture", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About the Smart Medicine Redistribution System")
st.markdown("**Architecture v4 • Academic Data Mining, Data Analysis & Big Data Implementation**")

render_sample_banner()

# Load Serving Data
hospitals_df = load_serving_table("hospitals")
state_df = load_serving_table("state_daily")
render_sidebar_filters(hospitals_df, state_df)

# ==============================================================================
# SECTION 1: USE THIS DASHBOARD IN 2 MINUTES
# ==============================================================================
st.markdown("### ⏱️ Use This Dashboard in 2 Minutes")
st.markdown("""
1. **Executive Overview (Home)**:
   - Check **What needs attention now** for critical shortage deficits and medicines nearing expiry.
   - Inspect **What should be done** for recommended transfer directives and download logistics dispatch CSVs.
   - Review **What it achieves** to see quantified stockout and wastage reductions compared with 'Do nothing'.
2. **Hospital Administrator**:
   - Select any hospital in Kerala to inspect local inventory levels, days of stock, and incoming/outgoing transfer manifests.
3. **Scenario Comparison**:
   - Compare the **Smart plan** against **Nearest surplus** and **Do nothing** across baseline conditions and emergency stress situations.
4. **Replay & Audit**:
   - Pick any past day and verify that real patient consumption matched the 95% forecast prediction interval.
5. **Model Evaluation**:
   - Review the empirical ML metrics (MAE, RMSE, WAPE), OPD ablation gains, and classification recall.
""")

st.markdown("---")

# ==============================================================================
# SECTION 2: PROJECT GLOSSARY
# ==============================================================================
st.markdown("### 📖 Project Glossary & Terminology")
glossary_cols = st.columns(2)
items = list(GLOSSARY.items())
half = len(items) // 2

with glossary_cols[0]:
    for term, definition in items[:half]:
        with st.expander(f"🔹 **{term}**", expanded=False):
            st.write(definition)

with glossary_cols[1]:
    for term, definition in items[half:]:
        with st.expander(f"🔹 **{term}**", expanded=False):
            st.write(definition)

st.markdown("---")

# ==============================================================================
# SECTION 3: SYSTEM METHODOLOGY & ZERO-COMPUTE ARCHITECTURE
# ==============================================================================
st.markdown("### ⚙️ System Architecture & Zero-Compute Serving Layer")
st.markdown("""
This system implements a decoupled Big Data intelligence pipeline:
- **Stateful Daily Simulation**: Generates multi-facility demand, OPD clinic visits, batch shelf-life aging, and inventory balances.
- **PySpark ETL Engine**: Executes windowed lags, deduplication, and feature engineering (with automatic local DuckDB/Arrow fallback).
- **Hybrid Machine Learning Models**: Combines high-volume seasonal estimators with global gradient-boosted trees and Random Forest risk classifiers.
- **OR-Tools Mathematical Optimization (MILP)**: Solves multi-facility inventory redistribution using Mixed-Integer Linear Programming, enforcing receiving caps and soft FEFO tie-breaking.
- **Zero-Compute Serving Presentation**: This Streamlit application operates purely as a high-speed presentation layer reading precomputed Parquet serving tables. **No model training, simulation, or heavy optimization runs inside the UI.**
""")

st.markdown("---")

# ==============================================================================
# SECTION 4: SYSTEM LIMITATIONS (HONEST ACADEMIC TRANSPARENCY)
# ==============================================================================
st.markdown("### ⚠️ Methodological Limitations")
st.markdown("""
In accordance with academic research standards, the following design boundaries apply:
1. **Synthetic Data**: All hospital facilities, demand figures, and stock counts are generated reproducibly by the simulation engine; they do not represent official government statistics.
2. **Simulation Assumptions**: Lead times, cost-per-kilometer rates (₹2.5/km), and OPD lead-lag correlations represent calibrated assumptions rather than live telemetry.
3. **Local Spark Demonstration**: PySpark local-mode illustrates API competence and distributed data architecture, but does not provide multi-node cluster speedup on a single laptop.
4. **Precomputed Presentation Layer**: The hosted dashboard displays published serving tables and does not retrain models on the fly during user sessions.
""")

st.caption("Smart Medicine Redistribution System • Kerala Health Services • Academic Big Data Project v4.0")
