"""Standard baseline implementations for forecasting, classification, and redistribution policies."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pipeline.simulator.entities import Hospital


class LastValueForecastBaseline:
    """Naive baseline predicting the most recently observed consumption."""

    def predict(self, series: pd.Series, horizon: int = 1) -> np.ndarray:
        if series.empty:
            return np.zeros(horizon)
        last_val = series.iloc[-1]
        return np.full(horizon, last_val)


class SeasonalNaiveForecastBaseline:
    """Seasonal naive baseline predicting the value from 7 days ago (same weekday)."""

    def predict(self, series: pd.Series, horizon: int = 1) -> np.ndarray:
        if len(series) < 7:
            # Fall back to mean or last value
            val = series.iloc[-1] if not series.empty else 0.0
            return np.full(horizon, val)
        # Repeat the last 7 observed days
        recent_week = series.iloc[-7:].values
        preds = []
        for h in range(horizon):
            preds.append(recent_week[h % 7])
        return np.array(preds)


class MovingAverageForecastBaseline:
    """7-day moving average baseline."""

    def __init__(self, window: int = 7):
        self.window = window

    def predict(self, series: pd.Series, horizon: int = 1) -> np.ndarray:
        if series.empty:
            return np.zeros(horizon)
        mean_val = series.iloc[-self.window:].mean()
        return np.full(horizon, mean_val)


class DaysOfStockClassifierBaseline:
    """Rule-based heuristic classifier predicting stockout and surplus using Days-of-Stock."""

    def __init__(self, stockout_dos_threshold: float = 3.0, surplus_dos_threshold: float = 30.0):
        self.stockout_threshold = stockout_dos_threshold
        self.surplus_threshold = surplus_dos_threshold

    def predict_stockout(self, days_of_stock_series: pd.Series) -> np.ndarray:
        return (days_of_stock_series < self.stockout_threshold).astype(int).values

    def predict_surplus(self, days_of_stock_series: pd.Series) -> np.ndarray:
        return (days_of_stock_series > self.surplus_threshold).astype(int).values


class NoTransferPolicy:
    """Optimization baseline: zero transfers."""

    def __call__(self, sim) -> List[Dict[str, Any]]:
        return []


class GreedyNearestSurplusPolicy:
    """
    Greedy optimization baseline:
    Matches hospital shortage with nearest hospital having available surplus.
    """

    def __init__(self, safety_stock_days: int = 5):
        self.safety_stock_days = safety_stock_days

    def __call__(self, sim) -> List[Dict[str, Any]]:
        day = sim.current_day
        actions = []

        # Track in-transit incoming stock from inter-hospital transfers so greedy does not re-request items already on the way
        in_transit_map: Dict[Tuple[str, str], int] = {}
        for tr in getattr(sim, "pending_transfers", []):
            k = (tr.to_hospital_id, tr.medicine_id)
            in_transit_map[k] = in_transit_map.get(k, 0) + tr.quantity

        for m_id, med in sim.medicines.items():
            shortage_hospitals = []
            surplus_hospitals = []

            for h_id, hosp in sim.hospitals.items():
                stock = sim.get_hospital_medicine_stock(h_id, m_id)
                incoming = in_transit_map.get((h_id, m_id), 0)
                inv_pos = stock + incoming

                # Estimate demand rate using bed scale
                est_daily = max(1.0, (hosp.bed_count / 50.0) * med.criticality * 2.0)
                safety = int(est_daily * self.safety_stock_days)

                if inv_pos < safety:
                    shortage_hospitals.append((h_id, safety - inv_pos))
                elif stock > int(safety * 1.5):
                    surplus_qty = stock - int(safety * 1.5)
                    surplus_hospitals.append((h_id, surplus_qty))

            # Match greedily by nearest distance
            for dest_id, need_qty in shortage_hospitals:
                if need_qty <= 0 or not surplus_hospitals:
                    continue
                dest_hosp = sim.hospitals[dest_id]

                # Sort surplus hospitals by distance to destination
                surplus_hospitals.sort(
                    key=lambda s: dest_hosp.distance_to(sim.hospitals[s[0]])
                )

                for idx, (src_id, avail_qty) in enumerate(surplus_hospitals):
                    if avail_qty <= 0:
                        continue
                    transfer_qty = min(need_qty, avail_qty)
                    actions.append({
                        "from_hospital": src_id,
                        "to_hospital": dest_id,
                        "medicine_id": m_id,
                        "quantity": transfer_qty,
                    })
                    need_qty -= transfer_qty
                    surplus_hospitals[idx] = (src_id, avail_qty - transfer_qty)
                    if need_qty <= 0:
                        break

        return actions


class ProportionalSharingPolicy:
    """
    Proportional sharing baseline:
    Pools network surplus and distributes proportionally to deficit severity.
    """

    def __call__(self, sim) -> List[Dict[str, Any]]:
        actions = []
        for m_id in sim.medicines:
            deficits = {}
            surpluses = {}
            for h_id, hosp in sim.hospitals.items():
                stock = sim.get_hospital_medicine_stock(h_id, m_id)
                bench = hosp.bed_count * 2
                if stock < bench:
                    deficits[h_id] = bench - stock
                elif stock > bench * 2:
                    surpluses[h_id] = stock - bench * 2

            total_deficit = sum(deficits.values())
            if total_deficit <= 0 or not surpluses:
                continue

            for s_id, s_qty in surpluses.items():
                if s_qty <= 0:
                    continue
                for d_id, d_qty in deficits.items():
                    share = min(s_qty, int(s_qty * (d_qty / total_deficit)))
                    if share > 0:
                        actions.append({
                            "from_hospital": s_id,
                            "to_hospital": d_id,
                            "medicine_id": m_id,
                            "quantity": share,
                        })
                        s_qty -= share
                        if s_qty <= 0:
                            break
        return actions
