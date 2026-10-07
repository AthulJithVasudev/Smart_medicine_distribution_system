"""Data loader utility for Streamlit dashboard with intelligent caching and fallback."""
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVING_DIR = REPO_ROOT / "data" / "serving"
DEV_DIR = REPO_ROOT / "data" / "dev"


@st.cache_data(ttl=300)
def load_serving_table(table_name: str) -> pd.DataFrame:
    """Load a specific table from published serving Parquet directory."""
    p = SERVING_DIR / f"{table_name}.parquet"
    if p.exists():
        return pd.read_parquet(p)

    # Fallback to dev data directory
    fallback_p = DEV_DIR / f"{table_name}.parquet"
    if fallback_p.exists():
        return pd.read_parquet(fallback_p)

    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_all_dashboard_data() -> Dict[str, pd.DataFrame]:
    """Load all published serving tables."""
    table_names = [
        "scenario_catalog",
        "hospitals",
        "medicines",
        "state_daily",
        "transfers",
        "kpis_daily",
        "replay",
        "evaluation_forecasting",
        "evaluation_classification",
        "evaluation_anomaly",
        "evaluation_optimizer",
        "runs",
    ]
    data = {}
    for name in table_names:
        data[name] = load_serving_table(name)
    return data
