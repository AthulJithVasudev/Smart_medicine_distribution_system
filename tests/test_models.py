"""Unit and integration tests for Phase 4: Machine Learning models (Forecasting, Clustering, Risk Classification, Anomaly)."""
import pytest
import numpy as np
import pandas as pd

from pipeline.simulator.engine import Simulator
from pipeline.etl.pipeline import run_etl
from pipeline.models.forecasting import HybridMedicineForecaster, train_forecasters
from pipeline.models.clustering import HospitalClusterer
from pipeline.models.classification import RiskClassifier
from pipeline.models.anomaly import ConsumptionAnomalyDetector
from pipeline.evaluation.metrics import calculate_forecast_metrics, calculate_classification_metrics


@pytest.fixture(scope="module")
def dev_curated_data():
    sim = Simulator(config="configs/dev.yaml", seed=42)
    obs_df, _ = sim.run(days=21)
    curated = run_etl(
        base_dir="scratch/data_models_test",
        obs_df=obs_df,
        hospitals_map=sim.hospitals,
        medicines_map=sim.medicines,
    )
    return curated, sim


def test_hybrid_forecasting_training_and_prediction(dev_curated_data):
    """Verify hybrid forecasting model fits and produces non-empty point and bound predictions."""
    curated, _ = dev_curated_data
    train_df = curated[curated["report_day"] <= 14]
    test_df = curated[curated["report_day"] > 14]

    forecaster = train_forecasters(train_df, with_opd=True)
    preds = forecaster.predict(test_df)

    assert not preds.empty
    assert "predicted_demand" in preds.columns
    assert "lower_bound" in preds.columns
    assert "upper_bound" in preds.columns

    # Check bounds sanity
    assert (preds["lower_bound"] <= preds["predicted_demand"] + 1e-4).all()
    assert (preds["predicted_demand"] <= preds["upper_bound"] + 1e-4).all()

    # Calculate metrics
    merged = pd.merge(test_df, preds, on=["report_day", "hospital_id", "medicine_id"])
    assert not merged.empty
    metrics = calculate_forecast_metrics(merged["observed_consumption"], merged["predicted_demand"])
    assert metrics["mae"] > 0.0
    assert metrics["wape"] > 0.0


def test_forecasting_opd_ablation(dev_curated_data):
    """Verify forecaster supports training with and without OPD leading indicators."""
    curated, _ = dev_curated_data
    train_df = curated[curated["report_day"] <= 14]
    test_df = curated[curated["report_day"] > 14]

    # Model with OPD
    forecaster_with = train_forecasters(train_df, with_opd=True)
    preds_with = forecaster_with.predict(test_df)

    # Model without OPD
    forecaster_without = train_forecasters(train_df, with_opd=False)
    preds_without = forecaster_without.predict(test_df)

    assert not preds_with.empty
    assert not preds_without.empty
    assert "predicted_demand" in preds_with.columns
    assert "predicted_demand" in preds_without.columns


def test_hospital_clustering(dev_curated_data):
    """Verify K-Means clustering groups hospitals by operational profile."""
    curated, _ = dev_curated_data
    clusterer = HospitalClusterer(n_clusters=3, random_state=42)
    clusters_df = clusterer.fit_predict(curated)

    assert not clusters_df.empty
    assert len(clusters_df) == 10  # 10 hospitals in dev profile
    assert "cluster_id" in clusters_df.columns
    assert "cluster_label" in clusters_df.columns
    assert len(clusters_df["cluster_id"].unique()) >= 2


def test_risk_classification_with_realized_future_labels(dev_curated_data):
    """Verify Random Forest classification on forward-window realized outcomes."""
    curated, _ = dev_curated_data
    clf = RiskClassifier(horizon_days=5, random_state=42)

    labeled_data = clf.create_realized_future_labels(curated)
    assert not labeled_data.empty
    assert "label_future_stockout" in labeled_data.columns
    assert "label_future_surplus" in labeled_data.columns
    assert "label_future_expiry" in labeled_data.columns

    train_subset = labeled_data[labeled_data["report_day"] <= 10]
    test_subset = labeled_data[labeled_data["report_day"] > 10]

    fit_stats = clf.fit(train_subset)
    assert fit_stats["samples"] > 0

    predictions = clf.predict_risks(test_subset)
    assert not predictions.empty
    assert "p_stockout" in predictions.columns
    assert "p_surplus" in predictions.columns
    assert "risk_stockout" in predictions.columns


def test_unsupervised_anomaly_detection(dev_curated_data):
    """Verify Isolation Forest anomaly detector trains unsupervised and scores records."""
    curated, _ = dev_curated_data
    detector = ConsumptionAnomalyDetector(contamination=0.04, random_state=42)
    detector.fit(curated)

    anomaly_preds = detector.predict_anomalies(curated)
    assert not anomaly_preds.empty
    assert "is_anomaly" in anomaly_preds.columns
    assert "anomaly_score" in anomaly_preds.columns

    # Verify binary 0/1 indicator
    unique_vals = set(anomaly_preds["is_anomaly"].unique())
    assert unique_vals.issubset({0, 1})
