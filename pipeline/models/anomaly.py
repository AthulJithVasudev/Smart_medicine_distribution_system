"""Unsupervised anomaly detection for sudden spikes, drops, and inventory discrepancies."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
from pipeline.evaluation.metrics import calculate_classification_metrics


class ConsumptionAnomalyDetector:
    """
    Unsupervised Isolation Forest anomaly detector.
    Trained strictly WITHOUT anomaly labels.
    Identifies sudden demand surges, severe consumption drops, or inventory shifts.
    """

    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self.feature_cols: List[str] = []

    def _get_features(self, df: pd.DataFrame) -> List[str]:
        cols = [
            "observed_consumption",
            "observed_stock",
            "consumption_lag_1",
            "rolling_mean_7d_consumption",
            "rolling_std_7d_consumption",
            "days_of_stock",
            "opd_visits_total",
            "opd_visits_fever",
        ]
        return [c for c in cols if c in df.columns]

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit Isolation Forest model in purely unsupervised mode."""
        self.feature_cols = self._get_features(train_df)
        X = train_df[self.feature_cols].fillna(0.0)
        self.iso_forest.fit(X)

    def predict_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomaly flag and anomaly score.
        is_anomaly = 1 for anomalies, 0 for normal.
        """
        if df.empty:
            return pd.DataFrame()

        X = df[self.feature_cols].fillna(0.0)
        # IsolationForest: -1 for anomaly, +1 for inlier
        raw_preds = self.iso_forest.predict(X)
        scores = self.iso_forest.decision_function(X)

        res = df[["hospital_id", "medicine_id", "report_day"]].copy()
        res["anomaly_score"] = np.round(-scores, 4)  # Higher score = more anomalous
        res["is_anomaly"] = (raw_preds == -1).astype(int)
        return res

    def evaluate_against_injected_labels(
        self,
        predicted_anomalies_df: pd.DataFrame,
        ground_truth_labels: pd.Series,
    ) -> Dict[str, Any]:
        """
        Score unsupervised detector against known/injected anomaly labels for benchmark evaluation only.
        """
        y_true = ground_truth_labels.values
        y_pred = predicted_anomalies_df["is_anomaly"].values
        return calculate_classification_metrics(y_true, y_pred)
