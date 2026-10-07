"""Unit and integration tests for Phase 5: OR-Tools Redistribution Optimizer."""
import pytest
import numpy as np
import pandas as pd

from pipeline.simulator.engine import Simulator
from pipeline.simulator.entities import Hospital, Medicine, Batch
from pipeline.optimizer.solver import RedistributionOptimizer, OptimizationResult
from pipeline.optimizer.rolling import RollingHorizonOptimizer


@pytest.fixture
def mini_network():
    h1 = Hospital("H001", "Dist 1", "District", False, 100, 10.0, 76.0, "DistA")
    h2 = Hospital("H002", "PHC 2", "PHC", True, 10, 10.1, 76.1, "DistA")
    hospitals = {"H001": h1, "H002": h2}

    m1 = Medicine("M001", "Paracetamol", "Antipyretic", 2.0, 3, 365, True, False)
    medicines = {"M001": m1}

    # H001 has large surplus (1000 units), H002 has 0 units
    b1 = Batch("B1", "M001", "H001", 1000, 1000, 0, 180, "active")
    batches = {"B1": b1}

    pred_demand = {("H001", "M001"): 5.0, ("H002", "M001"): 10.0}

    return hospitals, medicines, batches, pred_demand


def test_optimizer_transfers_surplus_to_need(mini_network):
    """Verify optimizer identifies surplus at H001, shortfall at H002, and generates valid transfer."""
    hospitals, medicines, batches, pred_demand = mini_network
    optimizer = RedistributionOptimizer()

    result = optimizer.solve(
        current_day=0,
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
        safety_stock_days=5,
    )

    assert result.solved is True
    assert result.status in ("OPTIMAL", "FEASIBLE")
    assert len(result.transfers) > 0

    tr = result.transfers[0]
    assert tr.from_hospital == "H001"
    assert tr.to_hospital == "H002"
    assert tr.medicine_id == "M001"
    assert tr.quantity > 0


def test_mandatory_receiving_cap(mini_network):
    """
    Verify receiving cap: Destination H002 must NEVER receive more than its actual need,
    even when source H001 has 10,000 surplus units!
    """
    hospitals, medicines, batches, pred_demand = mini_network
    # Massive source surplus
    batches["B1"].current_quantity = 10000
    optimizer = RedistributionOptimizer(horizon_days=7)

    # Destination need = horizon_demand (10*7) + safety (10*5) - stock (0) = 120 units
    result = optimizer.solve(
        current_day=0,
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
        safety_stock_days=5,
    )

    assert result.solved is True
    total_received = sum(t.quantity for t in result.transfers if t.to_hospital == "H002")
    expected_need = int(10.0 * 7 + 10.0 * 5)
    assert total_received <= expected_need, f"Receiving cap violated! Received {total_received} > need {expected_need}"


def test_source_and_batch_limits(mini_network):
    """Verify transfers out of source do not exceed source surplus or batch stock."""
    hospitals, medicines, batches, pred_demand = mini_network
    # Limit batch to exactly 40 units
    batches["B1"].current_quantity = 40
    # Expected demand at H001 is 5*7=35 + safety 25 = 60. So stock 40 < 60 -> availability is 0!
    optimizer = RedistributionOptimizer()

    result = optimizer.solve(
        current_day=0,
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
        safety_stock_days=5,
    )

    assert result.solved is True
    # Zero transfers should occur because source H001 has no surplus!
    assert len(result.transfers) == 0


def test_expiry_feasibility_constraint(mini_network):
    """
    Verify batches that cannot reach destination before expiry or minimum shelf-life
    are strictly forbidden from being transferred.
    """
    hospitals, medicines, batches, pred_demand = mini_network
    # Batch expires on day 3, but transit + min_shelf_life (7) requires at least 8-9 days!
    batches["B1"].expiry_day = 3
    optimizer = RedistributionOptimizer(min_destination_shelf_life=7)

    result = optimizer.solve(
        current_day=0,
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
    )

    assert result.solved is True
    assert len(result.transfers) == 0, "Expiring batch should not have been transferred!"


def test_fefo_tie_breaking():
    """
    Verify that when two batches at the same source hospital have equal cost and quantities,
    the earlier-expiring batch is selected (FEFO tie-breaker preference).
    """
    h1 = Hospital("H001", "Hosp 1", "District", False, 100, 10.0, 76.0, "DistA")
    h2 = Hospital("H002", "Hosp 2", "PHC", False, 10, 10.05, 76.05, "DistA")
    hospitals = {"H001": h1, "H002": h2}

    m1 = Medicine("M001", "Paracetamol", "Antipyretic", 2.0, 3, 365)
    medicines = {"M001": m1}

    # Two equal-size batches at H001: B_EARLY expires day 40, B_LATE expires day 180
    b_early = Batch("B_EARLY", "M001", "H001", 500, 500, 0, 40, "active")
    b_late = Batch("B_LATE", "M001", "H001", 500, 500, 0, 180, "active")
    batches = {"B_EARLY": b_early, "B_LATE": b_late}

    pred_demand = {("H001", "M001"): 2.0, ("H002", "M001"): 10.0}

    optimizer = RedistributionOptimizer(fefo_epsilon=0.005)
    result = optimizer.solve(
        current_day=0,
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
    )

    assert result.solved is True
    assert len(result.transfers) > 0
    # Must pick the earlier batch B_EARLY!
    batches_used = [t.batch_id for t in result.transfers]
    assert "B_EARLY" in batches_used
    assert "B_LATE" not in batches_used


def test_rolling_horizon_optimizer_integration():
    """Verify RollingHorizonOptimizer executes seamlessly with Simulator."""
    sim = Simulator(config="configs/dev.yaml", seed=42)
    rolling_opt = RollingHorizonOptimizer(safety_stock_days=5)

    # Run for 3 days using rolling optimizer
    for d in range(3):
        actions = rolling_opt(sim)
        assert isinstance(actions, list)
        sim.step(actions)

    assert sim.current_day == 3
    # Check executed transfers log in simulator
    tr_log = sim.get_transfers_log()
    assert isinstance(tr_log, pd.DataFrame)
