"""Observation model implementing the information barrier, noise, delays, and censoring."""
import numpy as np
from typing import Dict, List, Any, Optional
from pipeline.simulator.entities import Hospital


class ObservationModel:
    """
    Applies real-world observation noise, reporting delays, missing values,
    unit conversion errors, and demand censoring (information barrier).
    """

    def __init__(self, observation_rng: np.random.Generator, config: Dict[str, Any]):
        self.rng = observation_rng
        self.reporting_delay_prob = config.get("reporting_delay_prob", 0.15)
        self.max_delay_days = config.get("max_delay_days", 3)
        self.missing_record_prob = config.get("missing_record_prob", 0.05)
        self.duplicate_record_prob = config.get("duplicate_record_prob", 0.02)
        self.unit_error_prob = config.get("unit_error_prob", 0.01)
        self.noise_std = config.get("noise_std", 0.05)

        # Buffer for delayed records: delivery_day -> list of records
        self._delayed_buffer: Dict[int, List[Dict[str, Any]]] = {}

    def observe_day(
        self,
        current_day: int,
        daily_records: List[Dict[str, Any]],
        hospitals_map: Dict[str, Hospital],
    ) -> List[Dict[str, Any]]:
        """
        Process the day's true state records through the information barrier.
        Applies censoring, adds observation imperfections, and manages reporting delays.
        Returns all records that become officially available/observable on current_day.
        """
        emitted_records: List[Dict[str, Any]] = []

        # First, release any delayed records scheduled to arrive on current_day
        if current_day in self._delayed_buffer:
            emitted_records.extend(self._delayed_buffer.pop(current_day))

        for rec in daily_records:
            # Drop missing records
            if self.rng.random() < self.missing_record_prob:
                continue

            h_id = rec["hospital_id"]
            hospital = hospitals_map.get(h_id)
            is_rural = hospital.is_rural if hospital else False

            # Censored demand: observed consumption CANNOT exceed available stock
            # True demand and unmet demand are STRICTLY STRIPPED from observation output!
            stock_available = rec["stock_available"]
            dispensed_units = rec["dispensed_units"]
            current_stock = rec["current_stock"]

            # Add measurement noise to observed stock and consumption
            noise_factor_stock = 1.0 + self.rng.normal(0, self.noise_std)
            observed_stock = max(0, int(round(current_stock * max(0.5, min(1.5, noise_factor_stock)))))

            noise_factor_disp = 1.0 + self.rng.normal(0, self.noise_std * 0.5)
            observed_consumption = max(0, int(round(dispensed_units * max(0.5, min(1.5, noise_factor_disp)))))

            # Rare unit conversion error (e.g. 10 tablets/strip)
            if self.rng.random() < self.unit_error_prob:
                if self.rng.random() < 0.5:
                    observed_consumption = observed_consumption * 10
                else:
                    observed_consumption = max(1, observed_consumption // 10)

            # Build strictly sanitized observation record (INFORMATION BARRIER)
            obs_record = {
                "record_id": f"REC_{current_day}_{rec['hospital_id']}_{rec['medicine_id']}",
                "day": current_day,
                "report_day": current_day,
                "hospital_id": rec["hospital_id"],
                "medicine_id": rec["medicine_id"],
                "observed_stock": observed_stock,
                "observed_consumption": observed_consumption,  # Censored!
                "opd_visits_total": rec["opd_visits_total"],
                "opd_visits_fever": rec["opd_visits_fever"],
                "opd_visits_gastro": rec["opd_visits_gastro"],
                "active_batches_count": rec["active_batches_count"],
                "nearest_expiry_days": rec["nearest_expiry_days"],
                "expired_units_recorded": rec["expired_units"],
                "transfers_in_received": rec["transfers_in_received"],
                "transfers_out_dispatched": rec["transfers_out_dispatched"],
                "supplier_receipts": rec["supplier_receipts"],
                "provenance": "SIMULATED",
            }

            # Check if this record is delayed (rural hospitals have higher delay probability)
            delay_chance = self.reporting_delay_prob * (1.8 if is_rural else 0.8)
            if self.rng.random() < delay_chance:
                delay = int(self.rng.integers(1, self.max_delay_days + 1))
                arrival_day = current_day + delay
                delayed_record = dict(obs_record)
                delayed_record["report_day"] = arrival_day
                if arrival_day not in self._delayed_buffer:
                    self._delayed_buffer[arrival_day] = []
                self._delayed_buffer[arrival_day].append(delayed_record)
            else:
                emitted_records.append(obs_record)

                # Occasional duplicate record
                if self.rng.random() < self.duplicate_record_prob:
                    dup_record = dict(obs_record)
                    dup_record["provenance"] = "DERIVED"
                    emitted_records.append(dup_record)

        return emitted_records
