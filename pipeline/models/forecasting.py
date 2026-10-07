"""Forecasting module with dedicated SARIMAX models for high-priority series and global tree models for the long tail."""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import HistGradientBoostingRegressor
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


class DedicatedTimeSeriesForecaster:
    """SARIMAX time series model with optional OPD exogenous regressor."""

    def __init__(self, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), with_opd: bool = True):
        self.order = order
        self.seasonal_order = seasonal_order
        self.with_opd = with_opd
        self.model_res = None
        self.last_exog = None

    def fit(self, y: pd.Series, exog: Optional[pd.Series] = None):
        """Fit SARIMAX model."""
        y_vals = y.astype(float).values
        exog_vals = None
        if self.with_opd and exog is not None and not exog.empty:
            exog_vals = exog.astype(float).values.reshape(-1, 1)

        try:
            mod = SARIMAX(
                y_vals,
                exog=exog_vals,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_res = mod.fit(disp=False, maxiter=50)
            if exog_vals is not None:
                self.last_exog = float(exog_vals[-1, 0])
        except Exception as e:
            logger.debug(f"SARIMAX fit fallback: {e}")
            self.model_res = None
            self.mean_val = float(np.mean(y_vals)) if len(y_vals) > 0 else 10.0

    def predict(self, horizon: int = 7, future_exog: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate point forecast and 80% confidence interval."""
        if self.model_res is None:
            preds = np.full(horizon, getattr(self, "mean_val", 10.0))
            return preds, preds * 0.7, preds * 1.3

        try:
            exog_future = None
            if self.with_opd and self.last_exog is not None:
                if future_exog is not None:
                    exog_future = future_exog.reshape(-1, 1)
                else:
                    exog_future = np.full((horizon, 1), self.last_exog)

            forecast_res = self.model_res.get_forecast(steps=horizon, exog=exog_future)
            preds = np.maximum(0.0, forecast_res.predicted_mean)
            ci = forecast_res.conf_int(alpha=0.20)
            lower = np.maximum(0.0, ci[:, 0] if ci.ndim > 1 else ci)
            upper = np.maximum(preds, ci[:, 1] if ci.ndim > 1 else ci)
            return preds, lower, upper
        except Exception:
            preds = np.maximum(0.0, self.model_res.forecast(steps=horizon))
            return preds, preds * 0.7, preds * 1.3


class GlobalTreeForecaster:
    """
    Global gradient boosting model for long-tail series.
    Trained across all hospital-medicine series using lag, rolling, calendar, and OPD features.
    """

    def __init__(self, with_opd: bool = True):
        self.with_opd = with_opd
        self.model = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
        )
        self.feature_names: List[str] = []

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        cols = [
            "calendar_dow",
            "calendar_doy",
            "sin_doy",
            "cos_doy",
            "is_monsoon_season",
            "consumption_lag_1",
            "consumption_lag_2",
            "consumption_lag_3",
            "consumption_lag_7",
            "stock_lag_1",
            "stock_lag_7",
            "rolling_mean_7d_consumption",
            "rolling_std_7d_consumption",
            "rolling_mean_14d_consumption",
            "days_to_nearest_expiry",
            "days_of_stock",
            "bed_count",
            "is_rural",
            "criticality",
            "price",
        ]
        if self.with_opd:
            cols.extend(["opd_total_lag_1", "opd_fever_lag_1", "opd_gastro_lag_1"])
        return [c for c in cols if c in df.columns]

    def fit(self, train_df: pd.DataFrame, target_col: str = "observed_consumption") -> None:
        self.feature_names = self._get_feature_cols(train_df)
        X = train_df[self.feature_names].fillna(0.0)
        y = train_df[target_col].fillna(0.0)
        self.model.fit(X, y)

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = test_df.reindex(columns=self.feature_names, fill_value=0.0).fillna(0.0)
        preds = np.maximum(0.0, self.model.predict(X))
        # Estimate empirical prediction intervals
        lower = np.maximum(0.0, preds * 0.75)
        upper = preds * 1.25
        return preds, lower, upper


class HybridMedicineForecaster:
    """
    Hybrid forecaster:
    - Dedicated SARIMAX for critical life-saving items (criticality >= 4)
    - Global Tree model for all general and long-tail items
    """

    def __init__(self, with_opd: bool = True):
        self.with_opd = with_opd
        self.dedicated_models: Dict[Tuple[str, str], DedicatedTimeSeriesForecaster] = {}
        self.global_model = GlobalTreeForecaster(with_opd=with_opd)

    def fit(self, curated_df: pd.DataFrame) -> None:
        if curated_df.empty:
            return

        # Train global model on all data
        self.global_model.fit(curated_df, target_col="observed_consumption")

        # Train dedicated SARIMAX only for high-criticality pairs with sufficient history
        for (h_id, m_id), grp in curated_df.groupby(["hospital_id", "medicine_id"]):
            crit = grp["criticality"].iloc[0] if "criticality" in grp.columns else 1
            if crit >= 4 and len(grp) >= 14:
                series = grp.sort_values("report_day")
                sarima = DedicatedTimeSeriesForecaster(with_opd=self.with_opd)
                exog = series["opd_fever_lag_1"] if self.with_opd and "opd_fever_lag_1" in series.columns else None
                sarima.fit(series["observed_consumption"], exog=exog)
                self.dedicated_models[(h_id, m_id)] = sarima

    def predict(self, test_df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Generate forecasts for items in test_df."""
        if test_df.empty:
            return pd.DataFrame()

        preds_list = []
        # Run global predictions first
        global_point, global_lower, global_upper = self.global_model.predict(test_df)
        pred_dict = {
            (row["hospital_id"], row["medicine_id"], row["report_day"]): (
                global_point[idx], global_lower[idx], global_upper[idx]
            )
            for idx, (_, row) in enumerate(test_df.iterrows())
        }

        # Override with dedicated models where available
        for (h_id, m_id), model in self.dedicated_models.items():
            sub = test_df[(test_df["hospital_id"] == h_id) & (test_df["medicine_id"] == m_id)]
            if not sub.empty:
                d_p, d_l, d_u = model.predict(horizon=len(sub))
                for idx, (_, row) in enumerate(sub.iterrows()):
                    if idx < len(d_p):
                        pred_dict[(h_id, m_id, row["report_day"])] = (d_p[idx], d_l[idx], d_u[idx])

        # Construct result dataframe
        results = []
        for _, row in test_df.iterrows():
            h_id = row["hospital_id"]
            m_id = row["medicine_id"]
            r_day = row["report_day"]
            p, l, u = pred_dict.get((h_id, m_id, r_day), (10.0, 7.0, 13.0))
            results.append({
                "report_day": r_day,
                "day": row.get("day", r_day),
                "hospital_id": h_id,
                "medicine_id": m_id,
                "predicted_demand": round(float(p), 2),
                "lower_bound": round(float(l), 2),
                "upper_bound": round(float(u), 2),
            })

        return pd.DataFrame(results)


def train_forecasters(curated_df: pd.DataFrame, with_opd: bool = True) -> HybridMedicineForecaster:
    """Helper to train and return fitted hybrid forecaster."""
    forecaster = HybridMedicineForecaster(with_opd=with_opd)
    forecaster.fit(curated_df)
    return forecaster
