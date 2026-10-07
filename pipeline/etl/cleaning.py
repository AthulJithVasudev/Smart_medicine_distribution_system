"""Data cleaning module: missing value imputation, deduplication, and unit normalization."""
import numpy as np
import pandas as pd
from typing import Dict, Any


def clean_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform ETL cleaning on raw observations:
    - Deduplicate records on (report_day, hospital_id, medicine_id) prioritizing OFFICIAL > SIMULATED > DERIVED.
    - Impute missing numeric values using robust grouped medians.
    - Detect and correct unit conversion errors (e.g. 10x strip/box errors).
    """
    if df.empty:
        return df.copy()

    cleaned = df.copy()

    # 1. Deduplication
    provenance_priority = {"OFFICIAL": 3, "SIMULATED": 2, "DERIVED": 1}
    cleaned["_priority"] = cleaned["provenance"].map(lambda p: provenance_priority.get(str(p), 0))
    cleaned = cleaned.sort_values(
        by=["report_day", "hospital_id", "medicine_id", "_priority"],
        ascending=[True, True, True, False],
    )
    cleaned = cleaned.drop_duplicates(
        subset=["report_day", "hospital_id", "medicine_id"],
        keep="first",
    ).drop(columns=["_priority"])

    # 2. Missing-value imputation
    numeric_cols = [
        "observed_stock",
        "observed_consumption",
        "opd_visits_total",
        "opd_visits_fever",
        "opd_visits_gastro",
        "nearest_expiry_days",
    ]
    for col in numeric_cols:
        if col in cleaned.columns and cleaned[col].isnull().any():
            # Grouped median by hospital and medicine, or global median
            group_medians = cleaned.groupby(["hospital_id", "medicine_id"])[col].transform("median")
            cleaned[col] = cleaned[col].fillna(group_medians).fillna(cleaned[col].median()).fillna(0)

    # 3. Unit normalization
    # Compute median consumption per medicine across all hospitals
    med_stats = cleaned.groupby("medicine_id")["observed_consumption"].transform("median").replace(0, 1.0)
    ratio = cleaned["observed_consumption"] / med_stats

    # Flag and normalize 10x anomalies (e.g. reported 10 tablets instead of 1 strip)
    unit_error_high = (ratio >= 8.5) & (ratio <= 12.5) & (cleaned["observed_consumption"] >= 50)
    cleaned.loc[unit_error_high, "observed_consumption"] = (
        cleaned.loc[unit_error_high, "observed_consumption"] // 10
    )

    # 4. Standard types
    cleaned["observed_stock"] = cleaned["observed_stock"].astype(int)
    cleaned["observed_consumption"] = cleaned["observed_consumption"].astype(int)
    cleaned["day"] = cleaned["day"].astype(int)
    cleaned["report_day"] = cleaned["report_day"].astype(int)

    return cleaned
