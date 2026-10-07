"""Evaluation metrics for forecasting, classification, anomaly detection, and redistribution optimization."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def calculate_forecast_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Calculate MAE, RMSE, WAPE, and sMAPE for time-series forecasting."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) == 0:
        return {"mae": 0.0, "rmse": 0.0, "wape": 0.0, "smape": 0.0}

    errors = y_t - y_p
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    sum_true = float(np.sum(np.abs(y_t)))
    wape = float(np.sum(np.abs(errors)) / max(1e-6, sum_true))

    denom = np.abs(y_t) + np.abs(y_p) + 1e-6
    smape = float(100.0 * np.mean(2.0 * np.abs(errors) / denom))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4),
        "smape": round(smape, 4),
    }


def calculate_classification_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, Any]:
    """Calculate Precision, Recall, F1-score, and confusion matrix counts."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    if len(y_t) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}}

    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def calculate_optimization_metrics(
    ground_truth_df: pd.DataFrame,
    transfers_df: Optional[pd.DataFrame] = None,
    prices_map: Optional[Dict[str, float]] = None,
    fixed_transport_cost: float = 100.0,
    cost_per_km: float = 2.5,
) -> Dict[str, float]:
    """
    Calculate real optimization performance metrics from simulator ground truth:
    - Wastage units (expired inventory)
    - Wastage in rupees (economic loss)
    - Stockout occurrences / days
    - Total unmet demand units
    - Fill rate (dispensed units / true demand)
    - Transport costs incurred
    """
    if ground_truth_df.empty:
        return {
            "wastage_units": 0,
            "wastage_rupees": 0.0,
            "stockout_days": 0,
            "unmet_demand_units": 0,
            "fill_rate": 1.0,
            "transport_cost": 0.0,
            "transfers_count": 0,
        }

    wastage_units = int(ground_truth_df["expired_units"].sum())
    unmet_demand = int(ground_truth_df["unmet_demand"].sum())
    stockout_days = int((ground_truth_df["unmet_demand"] > 0).sum())
    total_demand = int(ground_truth_df["true_demand"].sum())
    dispensed = int(ground_truth_df["dispensed_units"].sum())
    fill_rate = float(dispensed / max(1.0, float(total_demand)))

    # Calculate monetary wastage
    wastage_rupees = 0.0
    if prices_map:
        for m_id, grp in ground_truth_df.groupby("medicine_id"):
            unit_price = prices_map.get(m_id, 10.0)
            wastage_rupees += grp["expired_units"].sum() * unit_price
    else:
        wastage_rupees = wastage_units * 10.0

    # Transport costs
    transport_cost = 0.0
    transfers_count = 0
    if transfers_df is not None and not transfers_df.empty:
        transfers_count = len(transfers_df)
        dist_sum = transfers_df["distance_km"].sum() if "distance_km" in transfers_df.columns else 0.0
        transport_cost = (transfers_count * fixed_transport_cost) + (dist_sum * cost_per_km)

    return {
        "wastage_units": wastage_units,
        "wastage_rupees": round(float(wastage_rupees), 2),
        "stockout_days": stockout_days,
        "unmet_demand_units": unmet_demand,
        "fill_rate": round(fill_rate, 4),
        "transport_cost": round(float(transport_cost), 2),
        "transfers_count": transfers_count,
    }
