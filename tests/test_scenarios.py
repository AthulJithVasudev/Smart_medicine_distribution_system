"""Unit and integration tests for Phase 6: Scenario Engine, Policy Comparison, and Closed-Loop Dynamics."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.utils.scenario_loader import load_all_scenarios, get_scenario_catalog
from pipeline.simulator.engine import Simulator
from pipeline.run_scenario import execute_closed_loop_run, run_scenario_matrix


def test_all_yaml_scenarios_load():
    """Verify that all 7 required YAML scenario configurations load without errors."""
    scenarios = load_all_scenarios()
    expected_scenarios = {
        "normal",
        "monsoon_surge",
        "dengue_2x",
        "dengue_4x",
        "supplier_delay",
        "low_stock",
        "rural_access",
    }
    assert expected_scenarios.issubset(set(scenarios.keys()))

    for sc_name in expected_scenarios:
        sc = scenarios[sc_name]
        assert "type" in sc
        assert sc["type"] in ("seen", "unseen")
        assert "description" in sc
        assert "outbreak_multiplier" in sc


def test_seen_unseen_classification():
    """Verify scenarios are properly categorized into seen vs unseen stress tests."""
    catalog = get_scenario_catalog()
    seen_names = {item["name"] for item in catalog if item["type"] == "seen"}
    unseen_names = {item["name"] for item in catalog if item["type"] == "unseen"}

    assert "normal" in seen_names
    assert "monsoon_surge" in seen_names
    assert "dengue_2x" in seen_names
    assert "low_stock" in seen_names

    assert "dengue_4x" in unseen_names
    assert "supplier_delay" in unseen_names
    assert "rural_access" in unseen_names


def test_scenario_parameters_change_simulator_behavior():
    """Verify scenario multipliers and initial stock constraints actually change simulator dynamics."""
    scenarios = load_all_scenarios()

    # Normal vs Dengue 4X: dengue 4x must have higher antipyretic demand
    sim_normal = Simulator(config="configs/dev.yaml", seed=42)
    sim_normal.set_scenario(scenarios["normal"])
    _, gt_normal = sim_normal.run(days=5)

    sim_dengue = Simulator(config="configs/dev.yaml", seed=42)
    sim_dengue.set_scenario(scenarios["dengue_4x"])
    _, gt_dengue = sim_dengue.run(days=5)

    fever_meds = [m.medicine_id for m in sim_normal.medicines.values() if m.is_fever_related]
    demand_norm = gt_normal[gt_normal["medicine_id"].isin(fever_meds)]["true_demand"].sum()
    demand_dengue = gt_dengue[gt_dengue["medicine_id"].isin(fever_meds)]["true_demand"].sum()

    assert demand_dengue > demand_norm, "Dengue 4X did not produce higher fever demand than normal!"


def test_identical_seed_identical_demand_across_policies():
    """Verify that running with identical seeds produces identical underlying demand across different policies."""
    scenarios = load_all_scenarios()
    sc = scenarios["normal"]

    # Run No Transfer
    run_no = execute_closed_loop_run(
        config_path="configs/dev.yaml",
        scenario_cfg=sc,
        policy_name="no_transfer",
        seed=100,
        horizon_days=5,
    )
    # Run Greedy
    run_gr = execute_closed_loop_run(
        config_path="configs/dev.yaml",
        scenario_cfg=sc,
        policy_name="greedy",
        seed=100,
        horizon_days=5,
    )

    demand_no = run_no["state_daily"].groupby("day")["true_demand"].sum()
    demand_gr = run_gr["state_daily"].groupby("day")["true_demand"].sum()

    pd.testing.assert_series_equal(demand_no, demand_gr)


def test_closed_loop_transfers_affect_future_stock():
    """Verify that transfers dispatched on day t alter future stock at source and destination."""
    scenarios = load_all_scenarios()
    sc = scenarios["low_stock"]

    # In low stock, optimizer transfers surplus to hospitals in need
    run_opt = execute_closed_loop_run(
        config_path="configs/dev.yaml",
        scenario_cfg=sc,
        policy_name="optimizer",
        seed=42,
        horizon_days=7,
    )

    run_none = execute_closed_loop_run(
        config_path="configs/dev.yaml",
        scenario_cfg=sc,
        policy_name="no_transfer",
        seed=42,
        horizon_days=7,
    )

    # State at end of horizon must diverge because transfers were executed in optimizer run
    end_stock_opt = run_opt["state_daily"][run_opt["state_daily"]["day"] == 6]["current_stock"].values
    end_stock_none = run_none["state_daily"][run_none["state_daily"]["day"] == 6]["current_stock"].values

    assert not np.array_equal(end_stock_opt, end_stock_none)


def test_multiple_seeds_produce_different_realizations():
    """Verify different seeds generate different stochastic realizations."""
    scenarios = load_all_scenarios()
    sc = scenarios["normal"]

    run_s1 = execute_closed_loop_run(
        config_path="configs/dev.yaml",
        scenario_cfg=sc,
        policy_name="no_transfer",
        seed=11,
        horizon_days=5,
    )
    run_s2 = execute_closed_loop_run(
        config_path="configs/dev.yaml",
        scenario_cfg=sc,
        policy_name="no_transfer",
        seed=99,
        horizon_days=5,
    )

    d1 = run_s1["state_daily"]["true_demand"].sum()
    d2 = run_s2["state_daily"]["true_demand"].sum()
    assert d1 != d2


def test_replay_data_and_scenario_matrix_execution(tmp_path):
    """Verify replay data schema and matrix execution produces summary KPIs."""
    out_dir = tmp_path / "matrix_test"
    res = run_scenario_matrix(
        profile="dev",
        scenarios_to_run=["normal"],
        policies_to_run=["optimizer", "no_transfer"],
        seeds=2,
        output_dir=str(out_dir),
    )

    summary = res["summary"]
    replay = res["replay"]

    assert not summary.empty
    assert len(summary) == 2  # 1 scenario x 2 policies
    assert "wastage_units_mean" in summary.columns
    assert "fill_rate_mean" in summary.columns
    assert "transport_cost_mean" in summary.columns

    assert not replay.empty
    assert "forecast" in replay.columns
    assert "actual_demand" in replay.columns
    assert "lower_bound" in replay.columns
    assert "upper_bound" in replay.columns
