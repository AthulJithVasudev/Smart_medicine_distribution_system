"""Stateful simulator engine executing daily hospital medicine inventory dynamics."""
import math
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

from pipeline.simulator.entities import Hospital, Medicine, Batch, TransferOrder, SupplierOrder
from pipeline.simulator.demand import DemandGenerator
from pipeline.simulator.supply import SupplyManager
from pipeline.simulator.observation import ObservationModel


class Simulator:
    """
    Stateful daily time-stepped simulation of medicine distribution across government hospitals.
    Maintains true latent state, executes FEFO consumption, manages transit delays,
    and applies an information barrier to emit noisy, censored observations.
    """

    def __init__(self, config: Union[Dict[str, Any], str], seed: int = 42):
        if isinstance(config, str):
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = dict(config)

        self.seed = seed
        # Spawn 4 deterministic independent random streams
        self.seed_seq = np.random.SeedSequence(seed)
        child_seeds = self.seed_seq.spawn(4)
        self.demand_rng = np.random.default_rng(child_seeds[0])
        self.supply_rng = np.random.default_rng(child_seeds[1])
        self.fault_rng = np.random.default_rng(child_seeds[2])
        self.observation_rng = np.random.default_rng(child_seeds[3])

        # Subsystems
        self.demand_gen = DemandGenerator(self.demand_rng)
        self.supply_mgr = SupplyManager(self.supply_rng)
        self.obs_model = ObservationModel(self.observation_rng, self.config.get("observation", {}))

        # Simulation state
        self.current_day: int = 0
        self.hospitals: Dict[str, Hospital] = {}
        self.medicines: Dict[str, Medicine] = {}
        self.batches: Dict[str, Batch] = {}  # batch_id -> Batch
        self.pending_transfers: List[TransferOrder] = []
        self.pending_supplier_orders: List[SupplierOrder] = []
        self.opd_history: Dict[str, List[Dict[str, int]]] = {}  # hospital_id -> list of daily OPD records

        # Logs
        self._observations_log: List[Dict[str, Any]] = []
        self._hidden_ground_truth_log: List[Dict[str, Any]] = []
        self._transfers_executed_log: List[Dict[str, Any]] = []

        # Scenario overrides
        self.scenario_name: str = "normal"
        self.outbreak_multiplier: float = 1.0
        self.is_monsoon: bool = False
        self.supplier_delay_days: int = 0
        self.initial_stock_ratio: float = 1.0
        self.rural_transit_multiplier: float = 1.0
        self.outbreak_window: Tuple[int, int] = (0, 0)

        # Initialize network
        self._init_network()
        self._init_inventory()

    def set_scenario(self, scenario_cfg: Dict[str, Any]) -> None:
        """Apply scenario parameters to the simulation."""
        self.scenario_name = scenario_cfg.get("name", "custom")
        self.outbreak_multiplier = float(scenario_cfg.get("outbreak_multiplier", 1.0))
        self.is_monsoon = bool(scenario_cfg.get("is_monsoon", False))
        self.supplier_delay_days = int(scenario_cfg.get("supplier_delay_days", 0))
        self.initial_stock_ratio = float(scenario_cfg.get("initial_stock_ratio", 1.0))
        self.rural_transit_multiplier = float(scenario_cfg.get("rural_transit_multiplier", 1.0))
        window = scenario_cfg.get("outbreak_window", [0, 0])
        self.outbreak_window = (window[0], window[1]) if len(window) == 2 else (0, 0)

    def _init_network(self) -> None:
        """Generate hospital network and medicine catalog."""
        h_cfg = self.config.get("hospitals", {})
        h_count = h_cfg.get("count", 10)
        rural_ratio = h_cfg.get("rural_ratio", 0.6)
        ratio_phc = h_cfg.get("ratio_phc", 0.5)
        ratio_taluk = h_cfg.get("ratio_taluk", 0.3)

        # Realistic geographic region (Kerala / South India)
        base_lat, base_lon = 10.0, 76.5
        districts = ["Ernakulam", "Thrissur", "Palakkad", "Malappuram", "Kozhikode"]

        for i in range(h_count):
            h_id = f"H{i+1:03d}"
            u = self.fault_rng.random()
            if u < ratio_phc:
                h_type = "PHC"
                bed_count = int(self.fault_rng.integers(8, 25))
                is_rural = self.fault_rng.random() < rural_ratio
            elif u < ratio_phc + ratio_taluk:
                h_type = "Taluk"
                bed_count = int(self.fault_rng.integers(60, 140))
                is_rural = self.fault_rng.random() < (rural_ratio * 0.4)
            else:
                h_type = "District"
                bed_count = int(self.fault_rng.integers(250, 600))
                is_rural = False

            # Cluster coordinates
            lat = base_lat + self.fault_rng.normal(0, 0.4)
            lon = base_lon + self.fault_rng.normal(0, 0.4)
            district = districts[i % len(districts)]
            name = f"{district} {h_type} Hospital #{i+1}"

            self.hospitals[h_id] = Hospital(
                hospital_id=h_id,
                name=name,
                hospital_type=h_type,
                is_rural=is_rural,
                bed_count=bed_count,
                latitude=lat,
                longitude=lon,
                district=district,
            )
            self.opd_history[h_id] = []

        # Generate Medicines
        m_cfg = self.config.get("medicines", {})
        m_count = m_cfg.get("count", 30)
        catalog = [
            ("Paracetamol 500mg", "Antipyretic", 1.5, 3, 540, True, False),
            ("Amoxicillin 500mg", "Antibiotic", 4.2, 4, 365, False, False),
            ("Ciprofloxacin 500mg", "Antibiotic", 5.0, 4, 365, False, True),
            ("Oral Rehydration Salts", "IV Fluid", 12.0, 4, 730, False, True),
            ("Normal Saline 500ml", "IV Fluid", 35.0, 5, 730, True, True),
            ("Artesunate Injection", "Antimalarial", 120.0, 5, 365, True, False),
            ("Chloroquine 250mg", "Antimalarial", 2.8, 4, 540, True, False),
            ("Ibuprofen 400mg", "Analgesic", 2.0, 2, 540, False, False),
            ("Diclofenac 50mg", "Analgesic", 3.0, 2, 540, False, False),
            ("Cetirizine 10mg", "Antihistamine", 1.8, 1, 730, False, False),
            ("Metformin 500mg", "Antidiabetic", 2.5, 4, 730, False, False),
            ("Glimepiride 1mg", "Antidiabetic", 3.5, 3, 730, False, False),
            ("Amlodipine 5mg", "Cardiovascular", 2.2, 4, 730, False, False),
            ("Atenolol 50mg", "Cardiovascular", 3.0, 4, 730, False, False),
            ("Salbutamol Inhaler", "Respiratory", 95.0, 5, 540, False, False),
            ("Azithromycin 500mg", "Antibiotic", 18.0, 4, 365, True, False),
            ("Doxycycline 100mg", "Antibiotic", 4.0, 3, 365, True, False),
            ("Pantoprazole 40mg", "Gastrointestinal", 5.5, 2, 540, False, True),
            ("Domperidone 10mg", "Gastrointestinal", 2.8, 1, 540, False, True),
            ("Ceftriaxone 1g Inj", "Antibiotic", 45.0, 5, 365, True, False),
        ]

        for i in range(m_count):
            m_id = f"M{i+1:03d}"
            if i < len(catalog):
                name, cat, price, crit, shelf, fever, gastro = catalog[i]
            else:
                name = f"Essential Medicine {i+1}"
                cat = "General"
                price = float(round(self.fault_rng.uniform(2.0, 50.0), 2))
                crit = int(self.fault_rng.integers(1, 6))
                shelf = int(self.fault_rng.choice([180, 365, 540, 730]))
                fever = (crit >= 4)
                gastro = False

            self.medicines[m_id] = Medicine(
                medicine_id=m_id,
                name=name,
                category=cat,
                price=price,
                criticality=crit,
                shelf_life_days=shelf,
                is_fever_related=fever,
                is_gastro_related=gastro,
            )

    def _init_inventory(self) -> None:
        """Seed initial batches with staggered expiry across all hospital-medicine pairs."""
        b_idx = 0
        for h_id, hospital in self.hospitals.items():
            bed_scale = max(0.4, hospital.bed_count / 50.0)
            for m_id, med in self.medicines.items():
                base_qty = int(med.criticality * 40 * bed_scale * self.initial_stock_ratio)
                base_qty = max(10, base_qty)

                # Split initial inventory across 2 staggered batches (nearer & further expiry)
                exp_offset_1 = int(self.fault_rng.integers(25, 90))
                exp_offset_2 = int(self.fault_rng.integers(91, max(92, med.shelf_life_days)))

                qty_1 = int(base_qty * 0.4)
                qty_2 = base_qty - qty_1

                b1_id = f"BAT_INIT_{m_id}_{h_id}_{b_idx}"
                b_idx += 1
                b2_id = f"BAT_INIT_{m_id}_{h_id}_{b_idx}"
                b_idx += 1

                self.batches[b1_id] = Batch(
                    batch_id=b1_id,
                    medicine_id=m_id,
                    hospital_id=h_id,
                    initial_quantity=qty_1,
                    current_quantity=qty_1,
                    manufacture_day=0,
                    expiry_day=exp_offset_1,
                    status="active",
                )
                self.batches[b2_id] = Batch(
                    batch_id=b2_id,
                    medicine_id=m_id,
                    hospital_id=h_id,
                    initial_quantity=qty_2,
                    current_quantity=qty_2,
                    manufacture_day=0,
                    expiry_day=exp_offset_2,
                    status="active",
                )

    def get_hospital_medicine_stock(self, hospital_id: str, medicine_id: str) -> int:
        """Calculate total active valid stock for a hospital and medicine."""
        return sum(
            b.current_quantity
            for b in self.batches.values()
            if b.hospital_id == hospital_id
            and b.medicine_id == medicine_id
            and b.is_valid
            and b.expiry_day > self.current_day
        )

    def get_batches_for_item(self, hospital_id: str, medicine_id: str) -> List[Batch]:
        """Get valid active batches sorted by FEFO (ascending expiry day)."""
        batches = [
            b
            for b in self.batches.values()
            if b.hospital_id == hospital_id
            and b.medicine_id == medicine_id
            and b.is_valid
            and b.expiry_day > self.current_day
        ]
        return sorted(batches, key=lambda b: b.expiry_day)

    def step(self, actions: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Execute one simulated day:
        1. Deliver arriving transfers from transit.
        2. Deliver arriving supplier orders.
        3. Write off expired batches.
        4. Validate and dispatch new transfer actions.
        5. Generate daily OPD visits.
        6. Generate true medicine demand.
        7. Consume stock using FEFO (First Expired First Out).
        8. Evaluate supplier reorders.
        9. Record hidden ground truth.
        10. Generate noisy/delayed observations through information barrier.
        11. Advance current_day.
        """
        day = self.current_day
        sim_cfg = self.config.get("simulation", {})
        base_lead_time = sim_cfg.get("base_lead_time_days", 5)
        safety_stock_days = sim_cfg.get("safety_stock_days", 5)
        reorder_cycle_days = sim_cfg.get("reorder_cycle_days", 7)

        # 1. Deliver arriving transfers
        transfers_received_today: Dict[Tuple[str, str], int] = {}
        transfers_dispatched_today: Dict[Tuple[str, str], int] = {}
        remaining_transfers: List[TransferOrder] = []

        for transfer in self.pending_transfers:
            if transfer.arrival_day <= day:
                key = (transfer.to_hospital_id, transfer.medicine_id)
                transfers_received_today[key] = transfers_received_today.get(key, 0) + transfer.quantity
                # Add batch to receiving hospital if still unexpired
                if transfer.original_expiry_day > day:
                    new_b_id = f"BAT_TR_{transfer.order_id}"
                    self.batches[new_b_id] = Batch(
                        batch_id=new_b_id,
                        medicine_id=transfer.medicine_id,
                        hospital_id=transfer.to_hospital_id,
                        initial_quantity=transfer.quantity,
                        current_quantity=transfer.quantity,
                        manufacture_day=transfer.dispatch_day,
                        expiry_day=transfer.original_expiry_day,
                        status="active",
                    )
                    transfer.status = "delivered"
                else:
                    transfer.status = "expired_in_transit"
            else:
                remaining_transfers.append(transfer)
        self.pending_transfers = remaining_transfers

        # 2. Deliver arriving supplier orders
        supplier_receipts_today: Dict[Tuple[str, str], int] = {}
        remaining_supplier_orders: List[SupplierOrder] = []
        for s_order in self.pending_supplier_orders:
            if s_order.delivery_day <= day:
                key = (s_order.hospital_id, s_order.medicine_id)
                supplier_receipts_today[key] = supplier_receipts_today.get(key, 0) + s_order.quantity
                new_batch = self.supply_mgr.create_batch_from_delivery(s_order, day)
                self.batches[new_batch.batch_id] = new_batch
                s_order.status = "delivered"
            else:
                remaining_supplier_orders.append(s_order)
        self.pending_supplier_orders = remaining_supplier_orders

        # 3. Write off expired batches
        expired_units_today: Dict[Tuple[str, str], int] = {}
        for b in self.batches.values():
            if b.is_valid and b.expiry_day <= day:
                key = (b.hospital_id, b.medicine_id)
                expired_units_today[key] = expired_units_today.get(key, 0) + b.current_quantity
                b.status = "expired"
                b.current_quantity = 0

        # 4. Process new transfer actions
        if actions:
            for act in actions:
                from_h = act["from_hospital"]
                to_h = act["to_hospital"]
                m_id = act["medicine_id"]
                req_qty = int(act["quantity"])
                b_id = act.get("batch_id")

                if from_h == to_h or req_qty <= 0:
                    continue

                source_h = self.hospitals.get(from_h)
                dest_h = self.hospitals.get(to_h)
                if not source_h or not dest_h:
                    continue

                # Locate valid batch at source
                candidate_batch = None
                if b_id and b_id in self.batches:
                    b = self.batches[b_id]
                    if b.hospital_id == from_h and b.medicine_id == m_id and b.is_valid:
                        candidate_batch = b

                if candidate_batch is None:
                    # Choose FEFO batch with available stock
                    valid_batches = self.get_batches_for_item(from_h, m_id)
                    for b in valid_batches:
                        if b.current_quantity > 0:
                            candidate_batch = b
                            break

                if not candidate_batch:
                    continue

                actual_qty = min(req_qty, candidate_batch.current_quantity)
                if actual_qty <= 0:
                    continue

                # Calculate transit time based on distance
                distance_km = source_h.distance_to(dest_h)
                base_transit = max(1, int(math.ceil(distance_km / 120.0)))
                if source_h.is_rural or dest_h.is_rural:
                    base_transit = int(math.ceil(base_transit * self.rural_transit_multiplier))
                transit_days = max(1, base_transit)
                arrival_day = day + transit_days

                # Deduct immediately from source batch
                candidate_batch.current_quantity -= actual_qty
                if candidate_batch.current_quantity == 0:
                    candidate_batch.status = "depleted"

                t_order = TransferOrder(
                    order_id=f"TR_{day}_{from_h}_{to_h}_{m_id}_{len(self._transfers_executed_log)}",
                    from_hospital_id=from_h,
                    to_hospital_id=to_h,
                    medicine_id=m_id,
                    batch_id=candidate_batch.batch_id,
                    quantity=actual_qty,
                    dispatch_day=day,
                    arrival_day=arrival_day,
                    transit_days=transit_days,
                    original_expiry_day=candidate_batch.expiry_day,
                    status="in_transit",
                )
                self.pending_transfers.append(t_order)
                key = (from_h, m_id)
                transfers_dispatched_today[key] = transfers_dispatched_today.get(key, 0) + actual_qty

                self._transfers_executed_log.append({
                    "day": day,
                    "order_id": t_order.order_id,
                    "from_hospital": from_h,
                    "to_hospital": to_h,
                    "medicine_id": m_id,
                    "quantity": actual_qty,
                    "transit_days": transit_days,
                    "arrival_day": arrival_day,
                    "distance_km": round(distance_km, 2),
                })

        # 5. Generate OPD visits for all hospitals
        is_in_outbreak = (self.outbreak_window[0] <= day <= self.outbreak_window[1]) if self.outbreak_window[1] > 0 else False
        current_outbreak_mult = self.outbreak_multiplier if is_in_outbreak else 1.0

        daily_opd_records: Dict[str, Dict[str, int]] = {}
        for h_id, hospital in self.hospitals.items():
            opd = self.demand_gen.generate_opd_visits(
                hospital=hospital,
                day=day,
                outbreak_multiplier=current_outbreak_mult,
                is_monsoon=self.is_monsoon,
            )
            self.opd_history[h_id].append(opd)
            daily_opd_records[h_id] = opd

        # 6 & 7. Generate true demand and FEFO consumption
        today_state_records: List[Dict[str, Any]] = []

        for h_id, hospital in self.hospitals.items():
            opd_info = daily_opd_records[h_id]
            for m_id, med in self.medicines.items():
                true_demand, latent_meta = self.demand_gen.generate_medicine_demand(
                    hospital=hospital,
                    medicine=med,
                    day=day,
                    opd_history=self.opd_history[h_id],
                    scenario_multiplier=current_outbreak_mult,
                    is_monsoon=self.is_monsoon,
                )

                # FEFO consumption
                fefo_batches = self.get_batches_for_item(h_id, m_id)
                available_stock_before = sum(b.current_quantity for b in fefo_batches)

                dispensed_units = 0
                remaining_demand = true_demand
                for b in fefo_batches:
                    if remaining_demand <= 0:
                        break
                    can_take = min(remaining_demand, b.current_quantity)
                    b.current_quantity -= can_take
                    dispensed_units += can_take
                    remaining_demand -= can_take
                    if b.current_quantity == 0:
                        b.status = "depleted"

                # Hidden unmet demand (stockout shortfall)
                unmet_demand = max(0, true_demand - dispensed_units)
                current_stock_after = sum(b.current_quantity for b in self.get_batches_for_item(h_id, m_id))

                # Batch statistics
                active_batches = self.get_batches_for_item(h_id, m_id)
                nearest_expiry = min((b.expiry_day - day for b in active_batches), default=999)

                # 8. Check replenishment reorder policy
                on_order_qty = sum(
                    so.quantity
                    for so in self.pending_supplier_orders
                    if so.hospital_id == h_id and so.medicine_id == m_id
                )
                avg_demand_rate = max(1.0, latent_meta["expected_demand"])
                new_s_order = self.supply_mgr.evaluate_reorders(
                    day=day,
                    hospital=hospital,
                    medicine=med,
                    current_stock=current_stock_after,
                    on_order_qty=on_order_qty,
                    avg_daily_demand=avg_demand_rate,
                    base_lead_time=base_lead_time,
                    safety_stock_days=safety_stock_days,
                    reorder_cycle_days=reorder_cycle_days,
                    supplier_delay_days=self.supplier_delay_days,
                )
                if new_s_order:
                    self.pending_supplier_orders.append(new_s_order)

                key = (h_id, m_id)
                expired_units = expired_units_today.get(key, 0)
                transfers_in = transfers_received_today.get(key, 0)
                transfers_out = transfers_dispatched_today.get(key, 0)
                supp_receipts = supplier_receipts_today.get(key, 0)

                # Full true state (INTERNAL - DO NOT LEAK)
                state_record = {
                    "day": day,
                    "hospital_id": h_id,
                    "medicine_id": m_id,
                    "stock_available": available_stock_before,
                    "current_stock": current_stock_after,
                    "true_demand": true_demand,  # HIDDEN TRUTH
                    "unmet_demand": unmet_demand,  # HIDDEN TRUTH
                    "dispensed_units": dispensed_units,
                    "expired_units": expired_units,
                    "transfers_in_received": transfers_in,
                    "transfers_out_dispatched": transfers_out,
                    "supplier_receipts": supp_receipts,
                    "active_batches_count": len(active_batches),
                    "nearest_expiry_days": nearest_expiry,
                    "opd_visits_total": opd_info["total_opd"],
                    "opd_visits_fever": opd_info["fever_opd"],
                    "opd_visits_gastro": opd_info["gastro_opd"],
                    "latent_expected_demand": latent_meta["expected_demand"],
                    "is_in_outbreak": is_in_outbreak,
                    "scenario": self.scenario_name,
                }
                today_state_records.append(state_record)
                self._hidden_ground_truth_log.append(state_record)

        # 10. Generate noisy, delayed observations via Information Barrier
        today_observations = self.obs_model.observe_day(
            current_day=day,
            daily_records=today_state_records,
            hospitals_map=self.hospitals,
        )
        self._observations_log.extend(today_observations)

        # 11. Advance simulation clock
        self.current_day += 1

        return today_observations

    def observations(self, up_to_day: Optional[int] = None) -> pd.DataFrame:
        """
        Return the observation-model dataframe available up to up_to_day.
        Strictly contains only observable features — NO true demand or hidden state!
        """
        if not self._observations_log:
            return pd.DataFrame()
        df = pd.DataFrame(self._observations_log)
        if up_to_day is not None:
            df = df[df["report_day"] <= up_to_day].copy()
        return df

    def get_hidden_ground_truth(self, up_to_day: Optional[int] = None) -> pd.DataFrame:
        """
        Return hidden ground truth (true demand, latent rates, unmet demand).
        FOR EVALUATION CODE ONLY. MUST NEVER BE PASSED TO ML OR PIPELINE!
        """
        if not self._hidden_ground_truth_log:
            return pd.DataFrame()
        df = pd.DataFrame(self._hidden_ground_truth_log)
        if up_to_day is not None:
            df = df[df["day"] <= up_to_day].copy()
        return df

    def get_transfers_log(self) -> pd.DataFrame:
        """Return executed transfers log."""
        if not self._transfers_executed_log:
            return pd.DataFrame(columns=[
                "day", "order_id", "from_hospital", "to_hospital",
                "medicine_id", "quantity", "transit_days", "arrival_day", "distance_km"
            ])
        return pd.DataFrame(self._transfers_executed_log)

    def run(self, days: int, policy: Optional[Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the simulation forward for `days` steps.
        If a policy callable is provided, invokes policy(sim) at each day
        and applies returned actions into sim.step(actions).
        Returns (observations_df, ground_truth_df).
        """
        for _ in range(days):
            actions = None
            if policy is not None:
                actions = policy(self)
            self.step(actions)
        return self.observations(), self.get_hidden_ground_truth()
