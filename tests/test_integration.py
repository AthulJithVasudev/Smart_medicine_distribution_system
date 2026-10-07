"""Phase 9 Full Integration Test: End-to-End Execution of Entire System from Scratch."""
import shutil
from pathlib import Path
import pytest
import pandas as pd

from pipeline.prepare import prepare_profile_data
from pipeline.train import train_models_pipeline
from pipeline.run_scenario import run_scenario_matrix
from pipeline.export import export_serving_data
from pipeline.etl.storage import ParquetStorage, query_duckdb


def test_full_dev_pipeline_end_to_end_integration(tmp_path, monkeypatch):
    """
    Execute clean full DEV pipeline from Simulator through ETL, ML, Scenarios,
    OR-Tools Optimizer, MLflow, Export, and DuckDB serving queries.
    """
    # 1. Prepare DEV Data & PySpark/DuckDB ETL
    prepare_profile_data(profile="dev")

    dev_dir = Path("data/dev")
    assert (dev_dir / "hospitals.parquet").exists()
    assert (dev_dir / "medicines.parquet").exists()
    assert (dev_dir / "curated" / "curated_daily.parquet").exists()

    # Query curated data with DuckDB
    duck_res = query_duckdb(
        "SELECT count(*) as row_count FROM PARQUET_TABLE",
        parquet_path=dev_dir / "curated" / "curated_daily.parquet",
    )
    assert duck_res["row_count"].iloc[0] > 0

    # 2. Train ML Models & Baselines & MLflow
    train_models_pipeline(profile="dev")
    assert (dev_dir / "models" / "forecaster.pkl").exists()
    assert (dev_dir / "models" / "eval_forecasting.parquet").exists()
    assert (dev_dir / "models" / "hospital_clusters.parquet").exists()

    # 3. Run Scenario Matrix (Optimizer vs Greedy vs No-Transfer)
    matrix_res = run_scenario_matrix(
        profile="dev",
        scenarios_to_run=["normal", "dengue_2x"],
        policies_to_run=["optimizer", "greedy", "no_transfer"],
        seeds=2,
    )
    assert not matrix_res["summary"].empty
    assert not matrix_res["replay"].empty

    # 4. Export & Publish Serving Tables
    serving_dir = dev_dir / "serving"
    export_serving_data(profile="dev", target_dir=str(serving_dir))

    # 5. Verify Streamlit Serving Tables
    required_tables = [
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
    for tbl in required_tables:
        t_file = serving_dir / f"{tbl}.parquet"
        assert t_file.exists(), f"Published serving file {t_file} missing!"
        df = pd.read_parquet(t_file)
        assert not df.empty, f"Published table {tbl} is empty!"
