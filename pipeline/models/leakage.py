"""Leakage protection suite and diagnostic tests verifying absence of data snooping or hidden state access."""
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable
from sklearn.metrics import r2_score, f1_score


class DataLeakageAuditor:
    """Audits dataframes and training pipelines to detect information leakage."""

    FORBIDDEN_COLUMNS = [
        "true_demand",
        "unmet_demand",
        "latent_expected_demand",
        "is_in_outbreak",
        "scenario",
        "injected_anomaly",
        "hidden_stock",
    ]

    @classmethod
    def check_observation_barrier(cls, df: pd.DataFrame) -> None:
        """Verify observation dataframe contains zero hidden simulator variables."""
        leaked = [col for col in cls.FORBIDDEN_COLUMNS if col in df.columns]
        if leaked:
            raise ValueError(f"CRITICAL LEAKAGE: Simulator hidden truth detected in features: {leaked}")

    @classmethod
    def check_feature_target_separation(cls, feature_cols: List[str], target_cols: List[str]) -> None:
        """Verify no target variable or future label exists within feature columns."""
        overlap = set(feature_cols).intersection(set(target_cols))
        if overlap:
            raise ValueError(f"CRITICAL LEAKAGE: Target variable(s) found in feature set: {overlap}")

    @classmethod
    def label_shuffle_leakage_test(
        cls,
        X: pd.DataFrame,
        y: np.ndarray,
        train_eval_fn: Callable[[pd.DataFrame, np.ndarray], float],
        expected_random_baseline: float = 0.5,
        threshold: float = 0.75,
    ) -> bool:
        """
        Label-shuffle leakage test:
        If labels are randomly shuffled, any genuine model's test performance must collapse
        towards the random baseline. If performance remains suspiciously high, features leak the label!
        """
        rng = np.random.default_rng(42)
        shuffled_y = rng.permutation(y)

        shuffled_score = train_eval_fn(X, shuffled_y)
        if shuffled_score > threshold:
            warnings.warn(
                f"POTENTIAL DATA LEAKAGE DETECTED! Shuffled-label score {shuffled_score:.3f} exceeds threshold {threshold:.3f}"
            )
            return False
        return True

    @classmethod
    def check_near_perfect_score(cls, score: float, metric_name: str = "R2", alert_threshold: float = 0.99) -> None:
        """Trigger diagnostic warning if a score is suspiciously near-perfect (classic sign of label leakage)."""
        if score >= alert_threshold:
            warnings.warn(
                f"SUSPICIOUS NEAR-PERFECT SCORE: {metric_name} = {score:.4f} >= {alert_threshold}. Verify lack of target leakage."
            )
