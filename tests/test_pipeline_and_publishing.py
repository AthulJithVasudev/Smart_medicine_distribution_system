"""Unit and integration tests for Phase 7: GitHub Actions YAML validation, Pipeline stages, and Serving Export."""
import pytest
import yaml
from pathlib import Path
import pandas as pd

from pipeline.prepare import prepare_profile_data
from pipeline.train import train_models_pipeline
from pipeline.run_scenario import run_scenario_matrix
from pipeline.export import export_serving_data


def test_github_actions_workflows_valid_yaml():
    """Verify that CI and pipeline workflow YAML files exist and are valid syntax."""
    ci_path = Path(".github/workflows/ci.yml")
    pipe_path = Path(".github/workflows/pipeline.yml")

    assert ci_path.exists(), "ci.yml does not exist"
    assert pipe_path.exists(), "pipeline.yml does not exist"

    with open(ci_path, "r") as f:
        ci_cfg = yaml.safe_load(f)
        assert "jobs" in ci_cfg
        assert "test" in ci_cfg["jobs"]

    with open(pipe_path, "r") as f:
        pipe_cfg = yaml.safe_load(f)
        assert "jobs" in pipe_cfg
        assert "pipeline" in pipe_cfg["jobs"]


def test_dev_pipeline_end_to_end_and_serving_export(tmp_path):
    """
    Execute full pipeline stages on DEV profile and verify all 12 serving Parquet tables exist
    and contain real values.
    """
    serving_dir = tmp_path / "serving_test"

    # Stage 1: Prepare
    prepare_profile_data(profile="dev")
    assert Path("data/dev/hospitals.parquet").exists()
    assert Path("data/dev/medicines.parquet").exists()
    assert Path("data/dev/curated/curated_daily.parquet").exists()

    # Stage 2: Train
    train_models_pipeline(profile="dev")
    assert Path("data/dev/models/forecaster.pkl").exists()
    assert Path("data/dev/models/eval_forecasting.parquet").exists()

    # Stage 3: Scenario Matrix (run subset for speed)
    run_scenario_matrix(
        profile="dev",
        scenarios_to_run=["normal", "monsoon_surge"],
        policies_to_run=["optimizer", "greedy", "no_transfer"],
        seeds=2,
    )
    assert Path("data/dev/matrix/summary.parquet").exists()
    assert Path("data/dev/matrix/replay.parquet").exists()

    # Stage 4: Export to serving
    export_serving_data(profile="dev", target_dir=str(serving_dir))

    # Verify all 12 required tables exist
    required_tables = [
        "scenario_catalog.parquet",
        "hospitals.parquet",
        "medicines.parquet",
        "state_daily.parquet",
        "transfers.parquet",
        "kpis_daily.parquet",
        "replay.parquet",
        "evaluation_forecasting.parquet",
        "evaluation_classification.parquet",
        "evaluation_anomaly.parquet",
        "evaluation_optimizer.parquet",
        "runs.parquet",
    ]

    for table_name in required_tables:
        t_path = serving_dir / table_name
        assert t_path.exists(), f"Missing required serving table: {table_name}"
        df = pd.read_parquet(t_path)
        assert not df.empty, f"Serving table {table_name} is empty!"
        assert len(df) > 0

    # Verify real numbers in evaluation
    opt_eval = pd.read_parquet(serving_dir / "evaluation_optimizer.parquet")
    assert "wastage_units_mean" in opt_eval.columns
    assert "fill_rate_mean" in opt_eval.columns
    # Check that fill rates are positive reals
    assert (opt_eval["fill_rate_mean"] > 0.0).all()
