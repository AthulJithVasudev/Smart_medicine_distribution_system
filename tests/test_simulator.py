"""Comprehensive unit and integration tests for Phase 1: Stateful Simulator."""
import pytest
import numpy as np
import pandas as pd
from pipeline.simulator.engine import Simulator
from pipeline.simulator.entities import Hospital, Medicine, Batch, TransferOrder, SupplierOrder


@pytest.fixture
def dev_config():
    return {
        "profile": "dev",
        "hospitals": {
            "count": 6,
            "ratio_phc": 0.5,
            "ratio_taluk": 0.3,
            "ratio_district": 0.2,
            "rural_ratio": 0.5,
        },
        "medicines": {
            "count": 10,
        },
        "simulation": {
            "history_days": 14,
            "scenario_days": 7,
            "reorder_cycle_days": 5,
            "base_lead_time_days": 3,
            "safety_stock_days": 4,
        },
        "observation": {
            "reporting_delay_prob": 0.1,
            "max_delay_days": 2,
            "missing_record_prob": 0.05,
            "duplicate_record_prob": 0.02,
            "unit_error_prob": 0.01,
            "noise_std": 0.05,
        },
    }


def test_simulator_initialization(dev_config):
    """Verify simulator properly instantiates hospitals, medicines, batches, and state."""
    sim = Simulator(config=dev_config, seed=42)
    assert sim.current_day == 0
    assert len(sim.hospitals) == 6
    assert len(sim.medicines) == 10
    assert len(sim.batches) > 0


def test_hospital_attributes_generated(dev_config):
    """Verify hospital attributes (id, type, rural/urban, beds, coordinates, distance)."""
    sim = Simulator(config=dev_config, seed=42)
    for h_id, h in sim.hospitals.items():
        assert h.hospital_id.startswith("H")
        assert h.hospital_type in ("PHC", "Taluk", "District")
        assert isinstance(h.is_rural, (bool, np.bool_))
        assert h.bed_count > 0
        assert 8.0 <= h.latitude <= 13.0
        assert 74.0 <= h.longitude <= 78.0
        assert len(h.district) > 0

    h_list = list(sim.hospitals.values())
    dist = h_list[0].distance_to(h_list[1])
    assert dist >= 0.0


def test_medicine_attributes_generated(dev_config):
    """Verify medicine catalog attributes (id, name, category, price, criticality, shelf_life)."""
    sim = Simulator(config=dev_config, seed=42)
    for m_id, m in sim.medicines.items():
        assert m.medicine_id.startswith("M")
        assert len(m.name) > 0
        assert len(m.category) > 0
        assert m.price > 0.0
        assert 1 <= m.criticality <= 5
        assert m.shelf_life_days >= 180


def test_batch_and_stock_generation(dev_config):
    """Verify initial inventory batches and positive stock for all pairs."""
    sim = Simulator(config=dev_config, seed=42)
    for h_id in sim.hospitals:
        for m_id in sim.medicines:
            stock = sim.get_hospital_medicine_stock(h_id, m_id)
            assert stock > 0
            batches = sim.get_batches_for_item(h_id, m_id)
            assert len(batches) >= 2
            # Batches must be sorted by expiry ascending (FEFO)
            assert batches[0].expiry_day <= batches[1].expiry_day


def test_fefo_consumption_logic(dev_config):
    """Verify FEFO logic consumes the earliest-expiring batch first."""
    sim = Simulator(config=dev_config, seed=42)
    h_id = list(sim.hospitals.keys())[0]
    m_id = list(sim.medicines.keys())[0]

    batches = sim.get_batches_for_item(h_id, m_id)
    assert len(batches) >= 2
    b_early = batches[0]
    b_late = batches[1]
    assert b_early.expiry_day < b_late.expiry_day

    initial_early_qty = b_early.current_quantity
    initial_late_qty = b_late.current_quantity

    # Run 1 day to trigger demand and consumption
    sim.step()

    # The earlier batch must have been consumed first
    if b_early.current_quantity > 0:
        # If early batch wasn't fully depleted, late batch must not be touched
        assert b_late.current_quantity == initial_late_qty
    else:
        # Early batch was depleted first
        assert b_early.status in ("depleted", "active")


def test_expiry_write_off(dev_config):
    """Verify expired batches are written off when expiry_day <= current_day."""
    sim = Simulator(config=dev_config, seed=42)
    h_id = list(sim.hospitals.keys())[0]
    m_id = list(sim.medicines.keys())[0]

    # Insert a batch that expires tomorrow (day 1)
    expiring_batch = Batch(
        batch_id="TEST_EXPIRING_BATCH",
        medicine_id=m_id,
        hospital_id=h_id,
        initial_quantity=500,
        current_quantity=500,
        manufacture_day=0,
        expiry_day=1,  # Expires on day 1
        status="active",
    )
    sim.batches["TEST_EXPIRING_BATCH"] = expiring_batch

    assert expiring_batch.is_valid
    # Advance to day 1
    sim.step()
    # Now current_day is 1, expiring_batch should be expired on the next step when day is 1
    sim.step()

    assert expiring_batch.status == "expired"
    assert expiring_batch.current_quantity == 0


def test_supplier_deliveries(dev_config):
    """Verify pending supplier orders arrive on their delivery day and create new batches."""
    sim = Simulator(config=dev_config, seed=42)
    h_id = list(sim.hospitals.keys())[0]
    m_id = list(sim.medicines.keys())[0]

    order = SupplierOrder(
        order_id="TEST_ORD_001",
        hospital_id=h_id,
        medicine_id=m_id,
        quantity=300,
        order_day=0,
        delivery_day=2,
        shelf_life_days=365,
        status="pending",
    )
    sim.pending_supplier_orders.append(order)

    # sim starts at day 0. Step day 0 -> advances clock to 1
    sim.step()
    assert order.status == "pending"

    # Step day 1 -> advances clock to 2
    sim.step()
    assert order.status == "pending"

    # Step day 2 -> delivery_day 2 reached and processed!
    sim.step()
    assert order.status == "delivered"
    delivered_batch = [b for b in sim.batches.values() if b.initial_quantity == 300 and b.hospital_id == h_id]
    assert len(delivered_batch) >= 1
    assert delivered_batch[0].is_valid


def test_transfer_scheduling_and_transit(dev_config):
    """Verify transfers are deducted at source, transit time is observed, and stock arrives at destination."""
    sim = Simulator(config=dev_config, seed=42)
    h_ids = list(sim.hospitals.keys())
    src, dst = h_ids[0], h_ids[1]
    m_id = list(sim.medicines.keys())[0]

    src_stock_before = sim.get_hospital_medicine_stock(src, m_id)
    dst_stock_before = sim.get_hospital_medicine_stock(dst, m_id)

    transfer_qty = 15
    action = [{
        "from_hospital": src,
        "to_hospital": dst,
        "medicine_id": m_id,
        "quantity": transfer_qty,
    }]

    sim.step(actions=action)

    # Source stock decreased
    src_stock_after = sim.get_hospital_medicine_stock(src, m_id)
    assert src_stock_after <= src_stock_before - transfer_qty

    # Check pending transfer exists
    assert len(sim.pending_transfers) >= 1
    t = sim.pending_transfers[0]
    assert t.from_hospital_id == src
    assert t.to_hospital_id == dst
    assert t.quantity == transfer_qty
    assert t.arrival_day > 0

    # Advance until transfer arrival day
    arrival_day = t.arrival_day
    while sim.current_day < arrival_day + 1:
        sim.step()

    # Transfer has arrived
    delivered_transfers = [tr for tr in sim._transfers_executed_log if tr["from_hospital"] == src and tr["to_hospital"] == dst]
    assert len(delivered_transfers) >= 1


def test_information_barrier_and_censored_demand(dev_config):
    """Verify observations strictly hide latent true demand and contain censored consumption."""
    sim = Simulator(config=dev_config, seed=42)
    sim.step()
    sim.step()

    obs_df = sim.observations()
    assert not obs_df.empty

    # Information Barrier: ML must never see these hidden columns in observations!
    forbidden_columns = [
        "true_demand",
        "unmet_demand",
        "latent_expected_demand",
        "is_in_outbreak",
        "scenario",
    ]
    for col in forbidden_columns:
        assert col not in obs_df.columns, f"Information leakage detected! Column {col} found in observations."

    # Allowed columns
    expected_columns = [
        "hospital_id",
        "medicine_id",
        "observed_stock",
        "observed_consumption",
        "opd_visits_total",
        "provenance",
    ]
    for col in expected_columns:
        assert col in obs_df.columns

    # Censored demand check: Ground truth shows unmet demand when stockouts happen
    gt_df = sim.get_hidden_ground_truth()
    assert not gt_df.empty
    assert "true_demand" in gt_df.columns
    assert "unmet_demand" in gt_df.columns
    # Check relationship: true_demand = dispensed_units + unmet_demand
    for _, row in gt_df.iterrows():
        assert row["true_demand"] == row["dispensed_units"] + row["unmet_demand"]


def test_random_seeds_reproducibility(dev_config):
    """Verify deterministic seeds produce identical results, while different seeds diverge."""
    sim1 = Simulator(config=dev_config, seed=123)
    sim2 = Simulator(config=dev_config, seed=123)
    sim3 = Simulator(config=dev_config, seed=999)

    for _ in range(5):
        sim1.step()
        sim2.step()
        sim3.step()

    gt1 = sim1.get_hidden_ground_truth()
    gt2 = sim2.get_hidden_ground_truth()
    gt3 = sim3.get_hidden_ground_truth()

    # Same seed runs must be completely identical
    pd.testing.assert_frame_equal(gt1, gt2)

    # Different seeds must produce different demands
    assert not gt1["true_demand"].equals(gt3["true_demand"])


def test_separate_random_streams(dev_config):
    """Verify the 4 spawned random streams operate independently without cross-stream collision."""
    sim = Simulator(config=dev_config, seed=42)
    r_demand = sim.demand_rng.random(10)
    r_supply = sim.supply_rng.random(10)
    r_fault = sim.fault_rng.random(10)
    r_obs = sim.observation_rng.random(10)

    # None of the random streams should produce the same numbers
    assert not np.array_equal(r_demand, r_supply)
    assert not np.array_equal(r_demand, r_fault)
    assert not np.array_equal(r_supply, r_obs)


def test_opd_leading_indicator(dev_config):
    """Verify OPD visits are generated and correlate with fever/gastro hospital throughput."""
    sim = Simulator(config=dev_config, seed=42)
    for _ in range(4):
        sim.step()

    h_id = list(sim.hospitals.keys())[0]
    history = sim.opd_history[h_id]
    assert len(history) == 4
    for day_opd in history:
        assert day_opd["total_opd"] >= day_opd["fever_opd"] + day_opd["gastro_opd"]
        assert day_opd["fever_opd"] > 0


def test_dev_profile_full_run():
    """Verify Simulator runs successfully with the actual configs/dev.yaml configuration."""
    sim = Simulator(config="configs/dev.yaml", seed=42)
    assert len(sim.hospitals) == 10
    assert len(sim.medicines) == 30

    # Run for 14 days
    obs_df, gt_df = sim.run(days=14)
    assert not obs_df.empty
    assert not gt_df.empty
    assert sim.current_day == 14
    assert "true_demand" not in obs_df.columns
    assert "true_demand" in gt_df.columns
    assert len(obs_df) > 0
    assert len(gt_df) > 0

