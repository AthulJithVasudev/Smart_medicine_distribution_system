"""Supply replenishment, ordering policy, lead times, and batch receipt dynamics."""
import numpy as np
from typing import Dict, List, Optional
from pipeline.simulator.entities import Hospital, Medicine, Batch, SupplierOrder


class SupplyManager:
    """Manages inventory review, supplier reordering, lead times, and batch generation."""

    def __init__(self, supply_rng: np.random.Generator):
        self.rng = supply_rng
        self._order_counter = 0
        self._batch_counter = 0

    def evaluate_reorders(
        self,
        day: int,
        hospital: Hospital,
        medicine: Medicine,
        current_stock: int,
        on_order_qty: int,
        avg_daily_demand: float,
        base_lead_time: int,
        safety_stock_days: int,
        reorder_cycle_days: int,
        supplier_delay_days: int = 0,
    ) -> Optional[SupplierOrder]:
        """
        Evaluate (s, S) periodic inventory review policy.
        Place supplier order if inventory position < reorder point.
        """
        # Periodic check or emergency threshold check
        is_review_day = (day % reorder_cycle_days == 0)
        reorder_point = int(avg_daily_demand * (base_lead_time + safety_stock_days))
        target_stock = int(avg_daily_demand * (base_lead_time + safety_stock_days + reorder_cycle_days * 2))

        inventory_position = current_stock + on_order_qty

        # Trigger order if below reorder point or critically low
        if inventory_position < reorder_point or (is_review_day and inventory_position < target_stock):
            order_qty = max(20, target_stock - inventory_position)
            # Add stochastic lead time jitter
            lead_time_noise = int(self.rng.integers(0, 3))
            lead_time = max(1, base_lead_time + lead_time_noise + supplier_delay_days)
            delivery_day = day + lead_time

            self._order_counter += 1
            order_id = f"ORD_{day}_{hospital.hospital_id}_{medicine.medicine_id}_{self._order_counter}"

            return SupplierOrder(
                order_id=order_id,
                hospital_id=hospital.hospital_id,
                medicine_id=medicine.medicine_id,
                quantity=order_qty,
                order_day=day,
                delivery_day=delivery_day,
                shelf_life_days=medicine.shelf_life_days,
                status="pending",
            )
        return None

    def create_batch_from_delivery(
        self,
        order: SupplierOrder,
        day: int,
    ) -> Batch:
        """Create a new valid batch when a supplier order is delivered."""
        self._batch_counter += 1
        batch_id = f"BAT_{order.medicine_id}_{order.hospital_id}_{day}_{self._batch_counter}"
        
        # Manufacture day is typically recent (10-45 days prior)
        mfg_offset = int(self.rng.integers(10, 45))
        mfg_day = max(0, day - mfg_offset)
        expiry_day = mfg_day + order.shelf_life_days

        return Batch(
            batch_id=batch_id,
            medicine_id=order.medicine_id,
            hospital_id=order.hospital_id,
            initial_quantity=order.quantity,
            current_quantity=order.quantity,
            manufacture_day=mfg_day,
            expiry_day=expiry_day,
            status="active",
        )
