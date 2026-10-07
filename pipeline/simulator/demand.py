"""Demand generation module with hospital size, seasonality, outbreak surges, and OPD leading indicators."""
import math
import numpy as np
from typing import Dict, List, Tuple
from pipeline.simulator.entities import Hospital, Medicine


class DemandGenerator:
    """Generates stochastic medicine demand and OPD leading indicators."""

    def __init__(self, demand_rng: np.random.Generator):
        self.rng = demand_rng

    def generate_opd_visits(
        self,
        hospital: Hospital,
        day: int,
        outbreak_multiplier: float = 1.0,
        is_monsoon: bool = False,
    ) -> Dict[str, int]:
        """
        Generate OPD (Out-Patient Department) visit counts for a hospital on a given day.
        Acts as an observable leading indicator for upcoming medicine demand.
        """
        # Base capacity scales with bed count
        base_patients = max(10, int(hospital.bed_count * 2.2))

        # Day of week effect (0=Mon ... 6=Sun)
        dow = day % 7
        dow_factor = 1.25 if dow in (0, 1) else (0.65 if dow == 6 else 1.0)

        # Seasonal effect (365-day cycle)
        day_of_year = day % 365
        annual_factor = 1.0 + 0.15 * math.sin(2 * math.pi * day_of_year / 365.0)

        # Environmental/Outbreak surges
        fever_factor = annual_factor * dow_factor * outbreak_multiplier
        if is_monsoon:
            fever_factor *= 1.4
            gastro_factor = 1.5 * dow_factor
        else:
            gastro_factor = 1.0 * dow_factor

        lam_fever = max(1.0, base_patients * 0.35 * fever_factor)
        lam_gastro = max(1.0, base_patients * 0.20 * gastro_factor)
        lam_general = max(2.0, base_patients * 0.45 * dow_factor * annual_factor)

        # Draw counts using Poisson
        fever_opd = int(self.rng.poisson(lam_fever))
        gastro_opd = int(self.rng.poisson(lam_gastro))
        general_opd = int(self.rng.poisson(lam_general))

        return {
            "fever_opd": fever_opd,
            "gastro_opd": gastro_opd,
            "general_opd": general_opd,
            "total_opd": fever_opd + gastro_opd + general_opd,
        }

    def generate_medicine_demand(
        self,
        hospital: Hospital,
        medicine: Medicine,
        day: int,
        opd_history: List[Dict[str, int]],
        scenario_multiplier: float = 1.0,
        is_monsoon: bool = False,
    ) -> Tuple[int, Dict[str, float]]:
        """
        Generate true latent medicine demand for a hospital and medicine.
        Uses OPD history as a leading indicator (lag 1-3 days).
        Returns:
            (true_demand_count, latent_metadata_for_logging)
        """
        # Base scale per hospital bed
        bed_factor = max(0.5, hospital.bed_count / 50.0)

        # Category-based base rate per bed
        category_bases = {
            "Analgesic": 12.0,
            "Antipyretic": 10.0,
            "Antibiotic": 7.0,
            "IV Fluid": 5.0,
            "Antimalarial": 3.0,
            "Antihistamine": 4.0,
            "Antidiabetic": 6.0,
            "Cardiovascular": 5.0,
            "Respiratory": 5.0,
            "Gastrointestinal": 6.0,
            "Dermatological": 3.0,
            "Oncology": 1.5,
        }
        base_rate = category_bases.get(medicine.category, 4.0)

        # Day of week effect
        dow = day % 7
        dow_factor = 1.2 if dow in (0, 1) else (0.7 if dow == 6 else 1.0)

        # Annual seasonality
        day_of_year = day % 365
        annual_factor = 1.0 + 0.2 * math.sin(2 * math.pi * (day_of_year - 45) / 365.0)

        # OPD leading indicator: inspect OPD visits from 1-2 days ago
        opd_effect = 1.0
        if len(opd_history) >= 1:
            lag1_opd = opd_history[-1]
            lag2_opd = opd_history[-2] if len(opd_history) >= 2 else lag1_opd

            if medicine.is_fever_related or medicine.category in ("Antipyretic", "Analgesic", "Antimalarial"):
                # Average recent fever OPD relative to hospital baseline
                recent_fever = (lag1_opd.get("fever_opd", 10) * 0.7 + lag2_opd.get("fever_opd", 10) * 0.3)
                baseline_fever = max(5.0, hospital.bed_count * 0.7)
                opd_effect = max(0.5, min(3.5, recent_fever / baseline_fever))
            elif medicine.is_gastro_related or medicine.category in ("Gastrointestinal", "IV Fluid"):
                recent_gastro = (lag1_opd.get("gastro_opd", 5) * 0.7 + lag2_opd.get("gastro_opd", 5) * 0.3)
                baseline_gastro = max(3.0, hospital.bed_count * 0.4)
                opd_effect = max(0.5, min(3.0, recent_gastro / baseline_gastro))

        # Monsoon effect on specific categories
        monsoon_factor = 1.0
        if is_monsoon and (medicine.is_fever_related or medicine.is_gastro_related):
            monsoon_factor = 1.35

        # Outbreak multiplier from scenario (e.g. dengue 2x, 4x)
        outbreak_effect = scenario_multiplier if medicine.is_fever_related else 1.0

        # Mean expected demand lambda
        expected_demand = (
            base_rate
            * bed_factor
            * dow_factor
            * annual_factor
            * opd_effect
            * monsoon_factor
            * outbreak_effect
        )

        # Count noise via Negative Binomial (overdispersion) or Poisson
        # Using Poisson with random gamma frailty (Negative Binomial equivalent)
        frailty = self.rng.gamma(shape=10.0, scale=0.1)  # mean 1.0, var 0.1
        noisy_lambda = max(0.1, expected_demand * frailty)
        true_demand = int(self.rng.poisson(noisy_lambda))

        metadata = {
            "expected_demand": float(expected_demand),
            "opd_effect": float(opd_effect),
            "outbreak_effect": float(outbreak_effect),
            "monsoon_factor": float(monsoon_factor),
            "noisy_lambda": float(noisy_lambda),
        }

        return true_demand, metadata
