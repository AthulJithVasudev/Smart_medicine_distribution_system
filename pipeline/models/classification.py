"""Risk classification module predicting future realized stockouts, surpluses, and expiries."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from pipeline.evaluation.metrics import calculate_classification_metrics


class RiskClassifier:
    """
    Random Forest multi-target risk classifier.
    Predicts whether a stockout, surplus, or expiry waste event will actually occur
    in the upcoming k-day horizon.
    Labels are derived strictly from REALIZED FUTURE OUTCOMES in the simulation forward window.
    """

    def __init__(self, horizon_days: int = 7, random_state: int = 42):
        self.horizon_days = horizon_days
        self.random_state = random_state
        self.rf_stockout = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=random_state)
        self.rf_surplus = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=random_state)
        self.rf_expiry = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=random_state)
        self.feature_cols: List[str] = []

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        cols = [
            "observed_stock",
            "observed_consumption",
            "days_of_stock",
            "days_to_nearest_expiry",
            "expiry_risk_score",
            "rolling_mean_7d_consumption",
            "rolling_std_7d_consumption",
            "consumption_lag_1",
            "consumption_lag_7",
            "stock_lag_1",
            "bed_count",
            "is_rural",
            "criticality",
            "price",
            "calendar_dow",
            "is_monsoon_season",
        ]
        return [c for c in cols if c in df.columns]

    def create_realized_future_labels(self, curated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive ground-truth future realized outcome labels from forward simulation window [t+1, t+k].
        """
        df = curated_df.sort_values(["hospital_id", "medicine_id", "report_day"]).copy()
        grouped = df.groupby(["hospital_id", "medicine_id"])

        # Forward rolling window to observe future realization
        # Shift -1 then rolling(k) forward
        k = self.horizon_days

        # Forward minimum stock in next k days
        fwd_indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=k)
        future_min_stock = grouped["observed_stock"].apply(
            lambda s: s.shift(-1).rolling(window=fwd_indexer, min_periods=1).min()
        ).reset_index(level=[0, 1], drop=True)

        # Forward minimum days of stock
        future_min_dos = grouped["days_of_stock"].apply(
            lambda s: s.shift(-1).rolling(window=fwd_indexer, min_periods=1).min()
        ).reset_index(level=[0, 1], drop=True)

        # Forward expired units if recorded
        exp_col = "expired_units_recorded" if "expired_units_recorded" in df.columns else "observed_stock"
        future_expired = grouped[exp_col].apply(
            lambda s: s.shift(-1).rolling(window=fwd_indexer, min_periods=1).sum()
        ).reset_index(level=[0, 1], drop=True)

        df["label_future_stockout"] = (future_min_stock <= 0).astype(int)
        df["label_future_surplus"] = (future_min_dos >= 35.0).astype(int)
        # Expiry event: expiry within k days and stock remaining
        df["label_future_expiry"] = (
            (df["days_to_nearest_expiry"] <= k) & (df["observed_stock"] > 0)
        ).astype(int)

        # Drop rows near the end of dataset where forward window cannot be observed
        max_day = df["report_day"].max()
        valid_df = df[df["report_day"] <= max_day - k].copy()
        return valid_df

    def fit(self, labeled_train_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit Random Forest classifiers for stockout, surplus, and expiry risk."""
        self.feature_cols = self._get_feature_cols(labeled_train_df)
        X = labeled_train_df[self.feature_cols].fillna(0.0)

        # Fit stockout
        y_stockout = labeled_train_df["label_future_stockout"].values
        if len(np.unique(y_stockout)) > 1:
            self.rf_stockout.fit(X, y_stockout)
        else:
            self.rf_stockout = None

        # Fit surplus
        y_surplus = labeled_train_df["label_future_surplus"].values
        if len(np.unique(y_surplus)) > 1:
            self.rf_surplus.fit(X, y_surplus)
        else:
            self.rf_surplus = None

        # Fit expiry
        y_expiry = labeled_train_df["label_future_expiry"].values
        if len(np.unique(y_expiry)) > 1:
            self.rf_expiry.fit(X, y_expiry)
        else:
            self.rf_expiry = None

        return {"features_count": len(self.feature_cols), "samples": len(X)}

    def predict_risks(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk probabilities and binary flags for items in test_df.
        Returns DataFrame with [hospital_id, medicine_id, report_day, stockout_risk, surplus_risk, expiry_risk, p_stockout, p_surplus, p_expiry].
        """
        if test_df.empty:
            return pd.DataFrame()

        X = test_df[self.feature_cols].fillna(0.0)
        n = len(X)

        if self.rf_stockout is not None:
            p_stockout = self.rf_stockout.predict_proba(X)[:, 1] if self.rf_stockout.classes_.shape[0] > 1 else np.zeros(n)
        else:
            p_stockout = (test_df["days_of_stock"] < 3.0).astype(float).values

        if self.rf_surplus is not None:
            p_surplus = self.rf_surplus.predict_proba(X)[:, 1] if self.rf_surplus.classes_.shape[0] > 1 else np.zeros(n)
        else:
            p_surplus = (test_df["days_of_stock"] > 35.0).astype(float).values

        if self.rf_expiry is not None:
            p_expiry = self.rf_expiry.predict_proba(X)[:, 1] if self.rf_expiry.classes_.shape[0] > 1 else np.zeros(n)
        else:
            p_expiry = (test_df["days_to_nearest_expiry"] < self.horizon_days).astype(float).values

        res = test_df[["hospital_id", "medicine_id", "report_day"]].copy()
        res["p_stockout"] = np.round(p_stockout, 3)
        res["p_surplus"] = np.round(p_surplus, 3)
        res["p_expiry"] = np.round(p_expiry, 3)
        res["risk_stockout"] = (p_stockout >= 0.5).astype(int)
        res["risk_surplus"] = (p_surplus >= 0.5).astype(int)
        res["risk_expiry"] = (p_expiry >= 0.5).astype(int)

        return res
