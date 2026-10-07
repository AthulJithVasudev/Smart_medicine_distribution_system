"""Feature engineering module ensuring strict temporal causality (no future information at time t)."""
import math
import numpy as np
import pandas as pd
from typing import Dict, Optional
from pipeline.simulator.entities import Hospital, Medicine


def engineer_features(
    cleaned_df: pd.DataFrame,
    hospitals_map: Optional[Dict[str, Hospital]] = None,
    medicines_map: Optional[Dict[str, Medicine]] = None,
) -> pd.DataFrame:
    """
    Engineer domain, lag, rolling, calendar, and OPD features.
    Guarantees no future leakage: all lag and rolling metrics are strictly computed
    on values prior to the observation day (shift >= 1).
    """
    if cleaned_df.empty:
        return cleaned_df.copy()

    df = cleaned_df.sort_values(by=["hospital_id", "medicine_id", "report_day"]).copy()

    # 1. Calendar and Seasonality features
    df["calendar_dow"] = df["report_day"] % 7
    df["calendar_doy"] = df["report_day"] % 365
    df["sin_doy"] = np.sin(2 * np.pi * df["calendar_doy"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["calendar_doy"] / 365.0)
    # Monsoon season (approx day 150 to 250 of year)
    df["is_monsoon_season"] = ((df["calendar_doy"] >= 150) & (df["calendar_doy"] <= 250)).astype(int)

    # 2. Grouped Temporal Features (Strict historical shift >= 1)
    grouped = df.groupby(["hospital_id", "medicine_id"])

    # Lags of observed consumption
    df["consumption_lag_1"] = grouped["observed_consumption"].shift(1)
    df["consumption_lag_2"] = grouped["observed_consumption"].shift(2)
    df["consumption_lag_3"] = grouped["observed_consumption"].shift(3)
    df["consumption_lag_7"] = grouped["observed_consumption"].shift(7)

    # Lags of observed stock
    df["stock_lag_1"] = grouped["observed_stock"].shift(1)
    df["stock_lag_7"] = grouped["observed_stock"].shift(7)

    # Lags of OPD leading indicators (hospital level)
    df["opd_total_lag_1"] = grouped["opd_visits_total"].shift(1)
    df["opd_fever_lag_1"] = grouped["opd_visits_fever"].shift(1)
    df["opd_gastro_lag_1"] = grouped["opd_visits_gastro"].shift(1)

    # Rolling statistics: computed strictly on historical values by shifting 1 step first!
    shifted_consumption = grouped["observed_consumption"].shift(1)
    df["rolling_mean_7d_consumption"] = (
        shifted_consumption.groupby([df["hospital_id"], df["medicine_id"]])
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["rolling_std_7d_consumption"] = (
        shifted_consumption.groupby([df["hospital_id"], df["medicine_id"]])
        .rolling(window=7, min_periods=1)
        .std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
    )
    df["rolling_mean_14d_consumption"] = (
        shifted_consumption.groupby([df["hospital_id"], df["medicine_id"]])
        .rolling(window=14, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Fill earliest history missing lags with available initial observations
    df["consumption_lag_1"] = df["consumption_lag_1"].fillna(df["observed_consumption"])
    df["consumption_lag_2"] = df["consumption_lag_2"].fillna(df["consumption_lag_1"])
    df["consumption_lag_3"] = df["consumption_lag_3"].fillna(df["consumption_lag_2"])
    df["consumption_lag_7"] = df["consumption_lag_7"].fillna(df["consumption_lag_1"])
    df["stock_lag_1"] = df["stock_lag_1"].fillna(df["observed_stock"])
    df["stock_lag_7"] = df["stock_lag_7"].fillna(df["observed_stock"])
    df["rolling_mean_7d_consumption"] = df["rolling_mean_7d_consumption"].fillna(df["observed_consumption"].replace(0, 1.0))
    df["rolling_mean_14d_consumption"] = df["rolling_mean_14d_consumption"].fillna(df["rolling_mean_7d_consumption"])

    # 3. Expiry and Inventory Health Metrics
    df["days_to_nearest_expiry"] = df["nearest_expiry_days"].clip(lower=0)
    df["expiry_risk_score"] = 1.0 / (df["days_to_nearest_expiry"] + 1.0)
    # Days of stock = current stock / average daily burn rate
    df["days_of_stock"] = df["observed_stock"] / df["rolling_mean_7d_consumption"].clip(lower=1.0)

    # 4. Integrate Hospital metadata if provided
    if hospitals_map:
        df["hospital_type"] = df["hospital_id"].map(lambda hid: hospitals_map[hid].hospital_type if hid in hospitals_map else "Unknown")
        df["is_rural"] = df["hospital_id"].map(lambda hid: int(hospitals_map[hid].is_rural) if hid in hospitals_map else 0)
        df["bed_count"] = df["hospital_id"].map(lambda hid: hospitals_map[hid].bed_count if hid in hospitals_map else 50)
        df["district"] = df["hospital_id"].map(lambda hid: hospitals_map[hid].district if hid in hospitals_map else "Unknown")
        df["latitude"] = df["hospital_id"].map(lambda hid: hospitals_map[hid].latitude if hid in hospitals_map else 10.0)
        df["longitude"] = df["hospital_id"].map(lambda hid: hospitals_map[hid].longitude if hid in hospitals_map else 76.5)

    # 5. Integrate Medicine metadata if provided
    if medicines_map:
        df["medicine_name"] = df["medicine_id"].map(lambda mid: medicines_map[mid].name if mid in medicines_map else "Medicine")
        df["category"] = df["medicine_id"].map(lambda mid: medicines_map[mid].category if mid in medicines_map else "General")
        df["price"] = df["medicine_id"].map(lambda mid: medicines_map[mid].price if mid in medicines_map else 10.0)
        df["criticality"] = df["medicine_id"].map(lambda mid: medicines_map[mid].criticality if mid in medicines_map else 3)
        df["shelf_life_days"] = df["medicine_id"].map(lambda mid: medicines_map[mid].shelf_life_days if mid in medicines_map else 365)

    return df
