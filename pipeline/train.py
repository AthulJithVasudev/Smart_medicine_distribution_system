"""Training script: fits forecasting, clustering, risk classification, and anomaly models with MLflow logging."""
import argparse
import logging
import pickle
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from pipeline.etl.storage import ParquetStorage
from pipeline.models.forecasting import HybridMedicineForecaster, train_forecasters
from pipeline.models.clustering import HospitalClusterer
from pipeline.models.classification import RiskClassifier
from pipeline.models.anomaly import ConsumptionAnomalyDetector
from pipeline.evaluation.metrics import (
    calculate_forecast_metrics,
    calculate_classification_metrics,
)
from pipeline.evaluation.baselines import (
    LastValueForecastBaseline,
    SeasonalNaiveForecastBaseline,
    MovingAverageForecastBaseline,
    DaysOfStockClassifierBaseline,
)
from pipeline.evaluation.validation import WalkForwardValidator
from pipeline.evaluation.tracking import ExperimentTracker, ChampionChallengerRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_models_pipeline(profile: str = "dev") -> None:
    """Train all models for profile and track results in MLflow."""
    config_path = f"configs/{profile}.yaml"
    with open(config_path, "r") as fp:
        cfg = yaml.safe_load(fp)

    out_dir = Path(cfg.get("data_dir", f"data/{profile}"))
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    storage = ParquetStorage(base_dir=str(out_dir))
    curated_df = storage.load_curated_data()
    if curated_df.empty:
        logger.error(f"No curated data found in {out_dir}. Please run 'python -m pipeline.prepare --profile {profile}' first.")
        return

    tracker = ExperimentTracker(
        tracking_uri=cfg.get("mlflow_tracking_uri", "sqlite:///mlflow.db"),
        experiment_name=f"medicine_distribution_{profile}",
    )
    registry = ChampionChallengerRegistry(registry_file=str(models_dir / "champion_registry.json"))

    # Train/Validation Split by time (Walk-Forward validation origin)
    max_day = curated_df["report_day"].max()
    split_day = max(7, max_day - 7)
    train_df = curated_df[curated_df["report_day"] <= split_day].copy()
    val_df = curated_df[curated_df["report_day"] > split_day].copy()
    if val_df.empty:
        val_df = train_df.copy()

    # 1. Forecasting: Baselines vs ML
    logger.info("Evaluating forecasting baselines...")
    last_base = LastValueForecastBaseline()
    seas_base = SeasonalNaiveForecastBaseline()
    ma_base = MovingAverageForecastBaseline(window=7)

    base_preds_last = []
    base_preds_seas = []
    base_preds_ma = []
    for (h_id, m_id), grp in val_df.groupby(["hospital_id", "medicine_id"]):
        hist = train_df[(train_df["hospital_id"] == h_id) & (train_df["medicine_id"] == m_id)]["observed_consumption"]
        base_preds_last.extend(last_base.predict(hist, horizon=len(grp)))
        base_preds_seas.extend(seas_base.predict(hist, horizon=len(grp)))
        base_preds_ma.extend(ma_base.predict(hist, horizon=len(grp)))

    y_val = val_df["observed_consumption"].values
    m_last = calculate_forecast_metrics(y_val, base_preds_last)
    m_seas = calculate_forecast_metrics(y_val, base_preds_seas)
    m_ma = calculate_forecast_metrics(y_val, base_preds_ma)

    # 2. Train Hybrid Forecaster (With OPD & Without OPD Ablation)
    logger.info("Training Hybrid Forecaster with OPD...")
    forecaster_with_opd = train_forecasters(train_df, with_opd=True)
    preds_with = forecaster_with_opd.predict(val_df)
    m_ml_with = calculate_forecast_metrics(val_df["observed_consumption"].values, preds_with["predicted_demand"].values)

    logger.info("Training Hybrid Forecaster without OPD (Ablation)...")
    forecaster_without_opd = train_forecasters(train_df, with_opd=False)
    preds_without = forecaster_without_opd.predict(val_df)
    m_ml_without = calculate_forecast_metrics(val_df["observed_consumption"].values, preds_without["predicted_demand"].values)

    # Log to MLflow
    run_id_forecast = tracker.log_experiment_run(
        run_name="hybrid_forecaster_opd",
        model_type="HybridForecaster",
        params={"with_opd": True, "profile": profile},
        metrics=m_ml_with,
        artifacts_df=preds_with,
    )
    logger.info(f"Forecasting WAPE with OPD: {m_ml_with['wape']} vs without OPD: {m_ml_without['wape']} (LastValue: {m_last['wape']})")

    # Evaluate Champion/Challenger
    champ_res = registry.evaluate_candidate(
        task_name="demand_forecasting",
        candidate_model_type="HybridForecaster",
        candidate_metric_value=m_ml_with["wape"],
        metric_name="wape",
        lower_is_better=True,
    )

    # 3. Hospital Clustering
    logger.info("Fitting Hospital K-Means Clusters...")
    clusterer = HospitalClusterer(n_clusters=3, random_state=42)
    clusters_df = clusterer.fit_predict(curated_df)
    clusters_df.to_parquet(models_dir / "hospital_clusters.parquet", index=False)

    # 4. Risk Classification
    logger.info("Training Risk Classifier on forward realized outcomes...")
    clf = RiskClassifier(horizon_days=5, random_state=42)
    labeled_curated = clf.create_realized_future_labels(curated_df)
    if not labeled_curated.empty:
        clf.fit(labeled_curated)
        risk_preds = clf.predict_risks(val_df)
    else:
        risk_preds = pd.DataFrame()

    # 5. Anomaly Detection (Unsupervised)
    logger.info("Fitting Unsupervised Isolation Forest Anomaly Detector...")
    detector = ConsumptionAnomalyDetector(contamination=0.03, random_state=42)
    detector.fit(curated_df)
    anomaly_preds = detector.predict_anomalies(curated_df)

    # Save models and evaluation reports
    eval_forecasting_df = pd.DataFrame([
        {"model": "Last Value Baseline", **m_last},
        {"model": "Seasonal Naive (7d)", **m_seas},
        {"model": "Moving Average (7d)", **m_ma},
        {"model": "Hybrid ML (without OPD)", **m_ml_without},
        {"model": "Hybrid ML (with OPD)", **m_ml_with},
    ])
    eval_forecasting_df.to_parquet(models_dir / "eval_forecasting.parquet", index=False)

    # Save pickled model objects
    with open(models_dir / "forecaster.pkl", "wb") as f:
        pickle.dump(forecaster_with_opd, f)
    with open(models_dir / "risk_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(models_dir / "anomaly_detector.pkl", "wb") as f:
        pickle.dump(detector, f)

    logger.info("Model training pipeline completed successfully.")


def main():
    import os
    default_profile = os.environ.get("MEDSYS_PROFILE", "dev")
    parser = argparse.ArgumentParser(description="Train ML models and evaluate baselines")
    parser.add_argument("--profile", default=default_profile, choices=["dev", "demo", "full"], help=f"Profile to train (default: {default_profile})")
    args = parser.parse_args()
    train_models_pipeline(profile=args.profile)


if __name__ == "__main__":
    main()

