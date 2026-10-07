"""Tests specifically proving prevention of target and temporal leakage."""
import warnings
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from pipeline.simulator.engine import Simulator
from pipeline.etl.pipeline import run_etl
from pipeline.models.leakage import DataLeakageAuditor
from pipeline.models.forecasting import GlobalTreeForecaster
from pipeline.models.classification import RiskClassifier
from pipeline.models.anomaly import ConsumptionAnomalyDetector


def test_leakage_audit_rejects_hidden_truth():
    """Verify DataLeakageAuditor catches leaked simulator truth."""
    leaked_df = pd.DataFrame([{"hospital_id": "H001", "observed_stock": 10, "true_demand": 50}])
    with pytest.raises(ValueError, match="Simulator hidden truth detected"):
        DataLeakageAuditor.check_observation_barrier(leaked_df)


def test_feature_target_overlap_prevention():
    """Verify DataLeakageAuditor flags overlap between features and future target labels."""
    features = ["observed_stock", "rolling_mean_7d_consumption", "label_future_stockout"]
    targets = ["label_future_stockout"]
    with pytest.raises(ValueError, match="Target variable.*found in feature set"):
        DataLeakageAuditor.check_feature_target_separation(features, targets)


def test_label_shuffle_leakage_test():
    """Verify that shuffling target labels destroys performance of an un-leaked model."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feat1": np.random.normal(0, 1, n),
        "feat2": np.random.normal(0, 1, n),
    })
    # Synthetic target with genuine modest correlation
    y = (X["feat1"] * 2.0 + np.random.normal(0, 1, n) > 0).astype(int).values

    def train_eval(X_sub, y_sub):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X_sub[:150], y_sub[:150])
        preds = clf.predict(X_sub[150:])
        return float(np.mean(preds == y_sub[150:]))

    passed = DataLeakageAuditor.label_shuffle_leakage_test(
        X, y, train_eval_fn=train_eval, threshold=0.75
    )
    assert passed is True


def test_suspicious_near_perfect_score_warning():
    """Verify diagnostic warning is issued if score is suspiciously close to 1.0 (leakage signal)."""
    with pytest.warns(UserWarning, match="SUSPICIOUS NEAR-PERFECT SCORE"):
        DataLeakageAuditor.check_near_perfect_score(0.999, metric_name="R2")


def test_isolation_forest_is_strictly_unsupervised():
    """Verify Isolation Forest training takes only X features and no target label y."""
    detector = ConsumptionAnomalyDetector()
    df = pd.DataFrame({
        "observed_consumption": [10, 12, 11, 50, 9],
        "observed_stock": [100, 90, 85, 20, 80],
        "consumption_lag_1": [10, 10, 12, 11, 50],
        "rolling_mean_7d_consumption": [10, 10, 11, 11, 19],
        "rolling_std_7d_consumption": [0, 1, 1, 1, 16],
        "days_of_stock": [10, 9, 8, 1.8, 4],
        "hospital_id": ["H001"] * 5,
        "medicine_id": ["M001"] * 5,
        "report_day": list(range(1, 6)),
    })
    detector.fit(df)
    preds = detector.predict_anomalies(df)
    assert "is_anomaly" in preds.columns
    # Model object itself must have no 'classes_' or classification supervised parameters
    assert not hasattr(detector.iso_forest, "classes_")


def test_unseen_scenario_generalization():
    """Verify model trained on normal scenario can predict on unseen dengue surge scenario."""
    # 1. Normal scenario simulation
    sim_normal = Simulator(config="configs/dev.yaml", seed=42)
    obs_normal, _ = sim_normal.run(days=14)
    curated_normal = run_etl(base_dir="scratch/unseen_normal", obs_df=obs_normal)

    # 2. Outbreak scenario simulation (unseen shift)
    sim_surge = Simulator(config="configs/dev.yaml", seed=99)
    sim_surge.set_scenario({"name": "dengue_4x", "outbreak_multiplier": 4.0, "outbreak_window": [1, 14]})
    obs_surge, _ = sim_surge.run(days=14)
    curated_surge = run_etl(base_dir="scratch/unseen_surge", obs_df=obs_surge)

    forecaster = GlobalTreeForecaster(with_opd=True)
    forecaster.fit(curated_normal)

    # Predict on unseen scenario
    preds_surge, lower, upper = forecaster.predict(curated_surge)
    assert len(preds_surge) == len(curated_surge)
    assert np.all(preds_surge >= 0.0)
