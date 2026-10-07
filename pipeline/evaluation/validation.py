"""Validation module implementing rolling-origin / walk-forward splits and oracle evaluation."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Tuple
from pipeline.evaluation.metrics import calculate_forecast_metrics, calculate_optimization_metrics


class WalkForwardValidator:
    """
    Rolling-origin / walk-forward time series validation.
    Strictly forbids random splitting of time series data.
    """

    def __init__(self, min_train_days: int = 14, step_size: int = 7, horizon: int = 7):
        self.min_train_days = min_train_days
        self.step_size = step_size
        self.horizon = horizon

    def split(self, df: pd.DataFrame, time_col: str = "report_day"):
        """
        Generate train/test index slices respecting chronological order.
        Yields (train_indices, test_indices, split_origin_day).
        """
        unique_days = sorted(df[time_col].unique())
        total_days = len(unique_days)

        if total_days < self.min_train_days + self.horizon:
            # Degenerate case: yield single split
            split_point = max(1, int(total_days * 0.7))
            train_idx = df[df[time_col] <= unique_days[split_point - 1]].index
            test_idx = df[df[time_col] > unique_days[split_point - 1]].index
            yield train_idx, test_idx, unique_days[split_point - 1]
            return

        origin_idx = self.min_train_days
        while origin_idx + self.horizon <= total_days:
            origin_day = unique_days[origin_idx - 1]
            test_end_day = unique_days[origin_idx + self.horizon - 1]

            train_idx = df[df[time_col] <= origin_day].index
            test_idx = df[(df[time_col] > origin_day) & (df[time_col] <= test_end_day)].index

            yield train_idx, test_idx, origin_day
            origin_idx += self.step_size

    def evaluate_model(
        self,
        df: pd.DataFrame,
        model_fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
        target_col: str = "observed_consumption",
        time_col: str = "report_day",
    ) -> Dict[str, Any]:
        """
        Evaluate a model function across all walk-forward folds.
        Returns average and per-fold metrics.
        """
        fold_metrics: List[Dict[str, float]] = []

        for train_idx, test_idx, origin_day in self.split(df, time_col=time_col):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            train_df = df.loc[train_idx]
            test_df = df.loc[test_idx]

            y_pred = model_fit_predict_fn(train_df, test_df)
            y_true = test_df[target_col].values

            metrics = calculate_forecast_metrics(y_true, y_pred)
            metrics["origin_day"] = origin_day
            metrics["test_size"] = len(test_idx)
            fold_metrics.append(metrics)

        if not fold_metrics:
            return {"mae": 0.0, "rmse": 0.0, "wape": 0.0, "smape": 0.0, "folds_count": 0}

        avg_mae = float(np.mean([m["mae"] for m in fold_metrics]))
        avg_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
        avg_wape = float(np.mean([m["wape"] for m in fold_metrics]))
        avg_smape = float(np.mean([m["smape"] for m in fold_metrics]))

        return {
            "mae": round(avg_mae, 4),
            "rmse": round(avg_rmse, 4),
            "wape": round(avg_wape, 4),
            "smape": round(avg_smape, 4),
            "folds_count": len(fold_metrics),
            "folds": fold_metrics,
        }


class OracleEvaluator:
    """
    Theoretical Oracle bound.
    Uses latent TRUE demand (accessible only in evaluation code) to establish
    the theoretical upper bound on prediction and distribution efficiency.
    """

    def evaluate_forecast_oracle(
        self,
        observations_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        model_predictions_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Compare model predictions against:
        1. Observable consumption (censored)
        2. True latent demand (Oracle ground truth)
        """
        merged = pd.merge(
            model_predictions_df,
            ground_truth_df[["day", "hospital_id", "medicine_id", "true_demand"]],
            on=["day", "hospital_id", "medicine_id"],
            how="inner",
        )
        if merged.empty:
            return {"oracle_mae": 0.0, "oracle_wape": 0.0}

        metrics_vs_true = calculate_forecast_metrics(
            y_true=merged["true_demand"].values,
            y_pred=merged["predicted_demand"].values,
        )
        return {
            "oracle_mae": metrics_vs_true["mae"],
            "oracle_rmse": metrics_vs_true["rmse"],
            "oracle_wape": metrics_vs_true["wape"],
        }
