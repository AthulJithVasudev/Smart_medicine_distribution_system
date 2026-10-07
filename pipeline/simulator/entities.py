"""Entities representing stateful objects in the hospital network simulation."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class Hospital:
    hospital_id: str
    name: str
    hospital_type: str  # "PHC", "Taluk", "District"
    is_rural: bool
    bed_count: int
    latitude: float
    longitude: float
    district: str

    def distance_to(self, other: "Hospital") -> float:
        """Haversine distance in kilometers between two hospitals."""
        R = 6371.0  # Earth radius in kilometers
        dlat = math.radians(other.latitude - self.latitude)
        dlon = math.radians(other.longitude - self.longitude)
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(math.radians(self.latitude))
            * math.cos(math.radians(other.latitude))
            * math.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c


@dataclass
class Medicine:
    medicine_id: str
    name: str
    category: str
    price: float
    criticality: int  # 1 (lowest) to 5 (life-saving emergency)
    shelf_life_days: int
    is_fever_related: bool = False
    is_gastro_related: bool = False


@dataclass
class Batch:
    batch_id: str
    medicine_id: str
    hospital_id: str
    initial_quantity: int
    current_quantity: int
    manufacture_day: int
    expiry_day: int
    status: str = "active"  # "active", "depleted", "expired", "in_transit"

    @property
    def is_valid(self) -> bool:
        return self.status == "active" and self.current_quantity > 0

    def days_until_expiry(self, current_day: int) -> int:
        return max(0, self.expiry_day - current_day)


@dataclass
class TransferOrder:
    order_id: str
    from_hospital_id: str
    to_hospital_id: str
    medicine_id: str
    batch_id: str
    quantity: int
    dispatch_day: int
    arrival_day: int
    transit_days: int
    original_expiry_day: int
    status: str = "in_transit"  # "in_transit", "delivered", "rejected"


@dataclass
class SupplierOrder:
    order_id: str
    hospital_id: str
    medicine_id: str
    quantity: int
    order_day: int
    delivery_day: int
    shelf_life_days: int
    status: str = "pending"  # "pending", "delivered"
