"""Unit and integration tests for Phase 2: PySpark ETL, Parquet Storage, and DuckDB Queries."""
import shutil
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.simulator.engine import Simulator
from pipeline.etl.storage import ParquetStorage, query_duckdb
from pipeline.etl.cleaning import clean_observations
from pipeline.etl.features import engineer_features
from pipeline.etl.pipeline import run_etl


@pytest.fixture
def temp_data_dir(tmp_path):
    d = tmp_path / "data_test"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_simulator():
    cfg = {
        "profile": "dev",
        "hospitals": {"count": 4, "ratio_phc": 0.5, "ratio_taluk": 0.5, "ratio_district": 0.0, "rural_ratio": 0.5},
        "medicines": {"count": 5},
        "simulation": {"history_days": 10, "reorder_cycle_days": 5, "base_lead_time_days": 3, "safety_stock_days": 4},
        "observation": {"reporting_delay_prob": 0.1, "max_delay_days": 2, "missing_record_prob": 0.05, "duplicate_record_prob": 0.02, "unit_error_prob": 0.01, "noise_std": 0.05},
    }
    sim = Simulator(config=cfg, seed=42)
    obs_df, _ = sim.run(days=10)
    return sim, obs_df


def test_raw_parquet_production_and_partitioning(temp_data_dir, sample_simulator):
    """Verify raw observations are written to Parquet partitioned by report_day."""
    _, obs_df = sample_simulator
    storage = ParquetStorage(base_dir=temp_data_dir)
    storage.save_raw_observations(obs_df, partition_col="report_day")

    raw_dir = temp_data_dir / "raw"
    assert raw_dir.exists()
    partition_dirs = list(raw_dir.glob("report_day=*"))
    assert len(partition_dirs) > 0, "No partitioned directories found!"

    # Verify reload of partitioned raw data
    loaded_raw = storage.load_raw_observations()
    assert not loaded_raw.empty
    assert len(loaded_raw) == len(obs_df)


def test_duplicate_removal_cleaning():
    """Verify duplicate removal drops duplicate records while keeping official/primary entries."""
    records = [
        {"report_day": 1, "hospital_id": "H001", "medicine_id": "M001", "observed_stock": 100, "observed_consumption": 10, "provenance": "SIMULATED", "day": 1},
        {"report_day": 1, "hospital_id": "H001", "medicine_id": "M001", "observed_stock": 105, "observed_consumption": 12, "provenance": "OFFICIAL", "day": 1},
        {"report_day": 1, "hospital_id": "H001", "medicine_id": "M001", "observed_stock": 95, "observed_consumption": 9, "provenance": "DERIVED", "day": 1},
    ]
    df = pd.DataFrame(records)
    cleaned = clean_observations(df)

    assert len(cleaned) == 1
    # OFFICIAL provenance has highest priority
    assert cleaned.iloc[0]["provenance"] == "OFFICIAL"
    assert cleaned.iloc[0]["observed_stock"] == 105


def test_missing_value_handling():
    """Verify null and missing values are properly imputed without crashing."""
    records = [
        {"report_day": 1, "hospital_id": "H001", "medicine_id": "M001", "observed_stock": None, "observed_consumption": None, "provenance": "SIMULATED", "day": 1, "nearest_expiry_days": None},
        {"report_day": 2, "hospital_id": "H001", "medicine_id": "M001", "observed_stock": 50, "observed_consumption": 10, "provenance": "SIMULATED", "day": 2, "nearest_expiry_days": 100},
    ]
    df = pd.DataFrame(records)
    cleaned = clean_observations(df)

    assert not cleaned["observed_stock"].isnull().any()
    assert not cleaned["observed_consumption"].isnull().any()


def test_unit_normalization():
    """Verify unit errors (10x outlier) are detected and scaled back down."""
    # Series of normal observations around 10 units, with one 100-unit error
    rows = []
    for d in range(1, 10):
        val = 100 if d == 5 else 10
        rows.append({
            "report_day": d, "hospital_id": "H001", "medicine_id": "M001",
            "observed_stock": 200, "observed_consumption": val,
            "provenance": "SIMULATED", "day": d, "nearest_expiry_days": 180,
            "opd_visits_total": 40, "opd_visits_fever": 10, "opd_visits_gastro": 5,
        })
    df = pd.DataFrame(rows)
    cleaned = clean_observations(df)
    day5_val = cleaned[cleaned["report_day"] == 5]["observed_consumption"].iloc[0]
    assert day5_val == 10, f"Expected 10x unit error to be corrected to 10, got {day5_val}"


def test_curated_parquet_and_duckdb_queries(temp_data_dir, sample_simulator):
    """Verify curated table is stored in Parquet and can be queried directly with DuckDB."""
    sim, obs_df = sample_simulator
    curated_df = run_etl(
        base_dir=str(temp_data_dir),
        obs_df=obs_df,
        hospitals_map=sim.hospitals,
        medicines_map=sim.medicines,
    )

    assert not curated_df.empty
    storage = ParquetStorage(base_dir=temp_data_dir)
    curated_path = storage.curated_dir / "curated_daily.parquet"
    assert curated_path.exists()

    # Query with DuckDB
    query = """
        SELECT
            hospital_id,
            medicine_id,
            AVG(observed_consumption) as avg_consumption,
            AVG(days_of_stock) as avg_dos
        FROM PARQUET_TABLE
        GROUP BY hospital_id, medicine_id
    """
    duck_res = query_duckdb(query=query, parquet_path=curated_path)
    assert not duck_res.empty
    assert "avg_consumption" in duck_res.columns
    assert "avg_dos" in duck_res.columns


def test_no_future_information_in_temporal_features():
    """
    Verify that temporal features (lags and rolling means) use ONLY historical data at time t.
    Specifically: consumption_lag_1 at day t must equal observed_consumption at day t-1.
    Rolling mean at day t must NOT include day t's consumption.
    """
    rows = []
    # 5 consecutive days with distinct known consumption values: 10, 20, 30, 40, 50
    test_consumptions = [10, 20, 30, 40, 50]
    for idx, c in enumerate(test_consumptions):
        d = idx + 1
        rows.append({
            "report_day": d,
            "hospital_id": "H001",
            "medicine_id": "M001",
            "observed_stock": 100,
            "observed_consumption": c,
            "opd_visits_total": 50,
            "opd_visits_fever": 15,
            "opd_visits_gastro": 10,
            "nearest_expiry_days": 100,
            "provenance": "SIMULATED",
            "day": d,
        })
    df = pd.DataFrame(rows)
    curated = engineer_features(df)

    # For day 2 (idx=1, c=20): lag 1 must equal day 1's consumption (10)
    day2 = curated[curated["report_day"] == 2].iloc[0]
    assert day2["consumption_lag_1"] == 10

    # For day 3 (idx=2, c=30): lag 1 must equal day 2's consumption (20), lag 2 must equal 10
    day3 = curated[curated["report_day"] == 3].iloc[0]
    assert day3["consumption_lag_1"] == 20
    assert day3["consumption_lag_2"] == 10

    # For day 3: rolling_mean_7d must be based only on days 1 and 2 (mean(10, 20) = 15.0)
    # It must NOT include day 3's value of 30!
    assert day3["rolling_mean_7d_consumption"] == 15.0


def test_dev_etl_completes_successfully(temp_data_dir):
    """Verify DEV profile full ETL run from raw simulator generation to curated Parquet."""
    sim = Simulator(config="configs/dev.yaml", seed=42)
    obs_df, _ = sim.run(days=14)

    curated = run_etl(
        base_dir=str(temp_data_dir),
        obs_df=obs_df,
        hospitals_map=sim.hospitals,
        medicines_map=sim.medicines,
    )

    assert not curated.empty
    assert "days_of_stock" in curated.columns
    assert "expiry_risk_score" in curated.columns
    assert "consumption_lag_1" in curated.columns
    assert "rolling_mean_7d_consumption" in curated.columns
    assert "is_rural" in curated.columns
    assert "criticality" in curated.columns
