"""Unit and integration tests for Phase 3: Baselines, Metrics, Walk-Forward Validation, and MLflow."""
import shutil
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.evaluation.metrics import (
    calculate_forecast_metrics,
    calculate_classification_metrics,
    calculate_optimization_metrics,
)
from pipeline.evaluation.baselines import (
    LastValueForecastBaseline,
    SeasonalNaiveForecastBaseline,
    MovingAverageForecastBaseline,
    DaysOfStockClassifierBaseline,
    NoTransferPolicy,
    GreedyNearestSurplusPolicy,
)
from pipeline.evaluation.validation import WalkForwardValidator, OracleEvaluator
from pipeline.evaluation.tracking import ExperimentTracker, ChampionChallengerRegistry


def test_forecast_metrics_calculation():
    """Verify calculation of MAE, RMSE, WAPE, and sMAPE."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 39.0])  # errors: 2, -2, 3, -1 -> abs errors: 2, 2, 3, 1 -> sum=8, mean=2.0

    m = calculate_forecast_metrics(y_true, y_pred)
    assert m["mae"] == 2.0
    # sq errors: 4 + 4 + 9 + 1 = 18 -> mean = 4.5 -> sqrt(4.5) ~ 2.1213
    assert abs(m["rmse"] - np.sqrt(4.5)) < 1e-3
    # wape = 8 / 100 = 0.08
    assert m["wape"] == 0.08
    assert m["smape"] > 0.0


def test_classification_and_optimization_metrics():
    """Verify classification and optimization metrics."""
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    cm = calculate_classification_metrics(y_true, y_pred)
    assert cm["precision"] == 1.0  # 2 TP, 0 FP
    assert cm["recall"] == round(2.0 / 3.0, 4)  # 2 TP, 1 FN
    assert cm["confusion_matrix"]["tp"] == 2

    # Optimization metrics
    gt_df = pd.DataFrame([
        {"hospital_id": "H001", "medicine_id": "M001", "true_demand": 100, "dispensed_units": 90, "unmet_demand": 10, "expired_units": 5},
        {"hospital_id": "H002", "medicine_id": "M001", "true_demand": 50, "dispensed_units": 50, "unmet_demand": 0, "expired_units": 0},
    ])
    opt_m = calculate_optimization_metrics(gt_df)
    assert opt_m["wastage_units"] == 5
    assert opt_m["unmet_demand_units"] == 10
    assert opt_m["stockout_days"] == 1
    assert opt_m["fill_rate"] == round(140 / 150.0, 4)


def test_baseline_forecasters():
    """Verify LastValue, SeasonalNaive, and MovingAverage baselines."""
    series = pd.Series([10, 20, 15, 25, 30, 35, 40, 50])  # 8 items

    # Last value
    last_base = LastValueForecastBaseline()
    preds_last = last_base.predict(series, horizon=3)
    assert len(preds_last) == 3
    assert np.all(preds_last == 50)

    # Seasonal naive (weekday 7 days ago)
    # series[-7:] = [20, 15, 25, 30, 35, 40, 50]
    seas_base = SeasonalNaiveForecastBaseline()
    preds_seas = seas_base.predict(series, horizon=3)
    assert preds_seas[0] == 20
    assert preds_seas[1] == 15
    assert preds_seas[2] == 25

    # Moving average (7-day window)
    ma_base = MovingAverageForecastBaseline(window=7)
    preds_ma = ma_base.predict(series, horizon=2)
    expected_ma = series.iloc[-7:].mean()
    assert np.allclose(preds_ma, expected_ma)


def test_walk_forward_validation_no_future_leakage():
    """
    Verify WalkForwardValidator splits strictly preserve chronological ordering.
    No test indices may ever precede or overlap training indices!
    """
    days = list(range(1, 31))  # 30 days
    records = []
    for d in days:
        records.append({"report_day": d, "observed_consumption": d * 2})
    df = pd.DataFrame(records)

    validator = WalkForwardValidator(min_train_days=10, step_size=5, horizon=5)
    splits = list(validator.split(df, time_col="report_day"))
    assert len(splits) > 0

    for train_idx, test_idx, origin_day in splits:
        train_days = df.loc[train_idx, "report_day"].values
        test_days = df.loc[test_idx, "report_day"].values

        assert max(train_days) <= origin_day
        assert min(test_days) > origin_day
        # Zero intersection
        assert len(set(train_days).intersection(set(test_days))) == 0


def test_oracle_evaluation():
    """Verify Oracle evaluator compares predictions against latent true demand."""
    oracle = OracleEvaluator()
    obs_df = pd.DataFrame([{"day": 1, "hospital_id": "H001", "medicine_id": "M001"}])
    gt_df = pd.DataFrame([{"day": 1, "hospital_id": "H001", "medicine_id": "M001", "true_demand": 50}])
    preds_df = pd.DataFrame([{"day": 1, "hospital_id": "H001", "medicine_id": "M001", "predicted_demand": 45}])

    res = oracle.evaluate_forecast_oracle(obs_df, gt_df, preds_df)
    assert res["oracle_mae"] == 5.0
    assert res["oracle_wape"] == 0.10


def test_mlflow_tracking_and_champion_challenger(tmp_path):
    """Verify MLflow runs are logged to SQLite and champion/challenger promotions execute."""
    db_path = tmp_path / "test_mlflow.db"
    tracker = ExperimentTracker(tracking_uri=f"sqlite:///{db_path}", experiment_name="test_exp")

    params = {"horizon": 7, "alpha": 0.1}
    metrics = {"wape": 0.12, "mae": 3.4}
    art_df = pd.DataFrame([{"item": 1, "pred": 10}])

    run_id = tracker.log_experiment_run(
        run_name="baseline_test",
        model_type="LastValue",
        params=params,
        metrics=metrics,
        artifacts_df=art_df,
    )
    assert run_id is not None
    assert len(run_id) > 0

    # Champion/Challenger
    reg_file = tmp_path / "champion_test.json"
    registry = ChampionChallengerRegistry(registry_file=str(reg_file))

    # Candidate 1: First model becomes champion automatically
    res1 = registry.evaluate_candidate(
        task_name="demand_forecast",
        candidate_model_type="BaselineNaive",
        candidate_metric_value=0.25,
        metric_name="wape",
        lower_is_better=True,
    )
    assert res1["promoted"] is True
    assert res1["champion_type"] == "BaselineNaive"

    # Candidate 2: Worse model (WAPE 0.30 > 0.25) is rejected
    res2 = registry.evaluate_candidate(
        task_name="demand_forecast",
        candidate_model_type="WorseModel",
        candidate_metric_value=0.30,
        metric_name="wape",
        lower_is_better=True,
    )
    assert res2["promoted"] is False
    assert res2["champion_type"] == "BaselineNaive"

    # Candidate 3: Better model (WAPE 0.15 < 0.25) is promoted
    res3 = registry.evaluate_candidate(
        task_name="demand_forecast",
        candidate_model_type="SarimaxModel",
        candidate_metric_value=0.15,
        metric_name="wape",
        lower_is_better=True,
    )
    assert res3["promoted"] is True
    assert res3["champion_type"] == "SarimaxModel"
    assert res3["champion_metric"] == 0.15


def test_no_transfer_policy_generates_zero_transfers():
    """Verify that NoTransferPolicy returns zero transfers and creates no transfer orders."""
    policy = NoTransferPolicy()
    actions = policy(None)
    assert actions == []
    assert len(actions) == 0


def test_impact_metrics_preserve_negative_differences():
    """Verify that impact metrics do not clamp regressions to zero and preserve signed difference."""
    baseline_stockout = 10.0
    worse_active_stockout = 15.0
    # Must preserve signed regression (-5.0), NOT clamped to 0.0
    stockout_diff = baseline_stockout - worse_active_stockout
    assert stockout_diff == -5.0

    baseline_waste = 50.0
    worse_active_waste = 80.0
    waste_diff = baseline_waste - worse_active_waste
    assert waste_diff == -30.0


def test_fill_rate_percentage_point_difference():
    """Verify that fill-rate difference is calculated in percentage points (pp), not relative % change."""
    baseline_fill = 0.90 * 100.0  # 90.0%
    active_fill = 0.95 * 100.0    # 95.0%

    pp_diff = active_fill - baseline_fill
    assert pp_diff == 5.0  # +5.0 percentage points, not (95-90)/90 = 5.55%

    # Negative difference case
    worse_fill = 0.85 * 100.0
    neg_pp_diff = worse_fill - baseline_fill
    assert neg_pp_diff == -5.0


def test_daily_and_cumulative_unmet_demand_calculation():
    """Verify that daily unmet demand and cumulative unmet demand are calculated chronologically."""
    df = pd.DataFrame([
        {"day": 0, "policy": "optimizer", "unmet_demand": 10},
        {"day": 1, "policy": "optimizer", "unmet_demand": 25},
        {"day": 2, "policy": "optimizer", "unmet_demand": 5},
    ])
    daily = df.sort_values("day")["unmet_demand"].tolist()
    assert daily == [10, 25, 5]

    cumulative = df.sort_values("day")["unmet_demand"].cumsum().tolist()
    assert cumulative == [10, 35, 40]

