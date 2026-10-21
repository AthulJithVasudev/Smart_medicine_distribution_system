"""
Auto-generate docs/DATA_DICTIONARY.md from the serving/sample Parquet tables.
Ensures documentation is 100% synchronized with actual parquet schemas.
"""
from pathlib import Path
import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

COLUMN_DESCRIPTIONS = {
    # Common
    "day": "Simulation day index (1-based, typically 1 to 90)",
    "hospital_id": "Unique identifier of the government hospital (e.g. H01)",
    "hospital_name": "Official facility name of the government hospital",
    "district": "Geographic district / administrative health zone",
    "hospital_type": "Tier level: DH (District Hospital), SDH (Sub-District Hospital), CHC (Community Health Center), PHC (Primary Health Center)",
    "bed_capacity": "Total operational inpatient bed count",
    "latitude": "WGS84 decimal latitude coordinates for geospatial mapping",
    "longitude": "WGS84 decimal longitude coordinates for geospatial mapping",
    "cluster_id": "Cluster assignment from K-Means clustering on OPD volume & capacity",
    "medicine_id": "Unique medicine identifier (e.g. M01)",
    "medicine_name": "Generic medicine and formulation name",
    "category": "Therapeutic class / clinical category (e.g. Antibiotic, Antipyretic, IV Fluid)",
    "unit_cost": "Government procurement cost per unit in Indian Rupees (INR)",
    "shelf_life_days": "Total manufacturer shelf-life from batch production to expiry",
    "scenario": "Evaluation scenario key (e.g. normal, monsoon_surge, dengue_4x)",
    "policy": "Redistribution policy key (optimizer, greedy, no_transfer)",
    "seed": "Simulation random seed index (1, 2, 3)",
    "metric": "Name of the evaluation or performance metric",
    # Evaluation
    "wape": "Weighted Absolute Percentage Error (|Actual - Forecast| / Total Actual)",
    "mae": "Mean Absolute Error in demand units",
    "model_name": "Name of the predictive model (LightGBM, Prophet, Moving Average, etc.)",
    "precision": "Precision score of binary classification model",
    "recall": "Recall / sensitivity score for predicting critical stockout events",
    "f1_score": "Harmonic mean of precision and recall",
    "auc_roc": "Area Under Receiver Operating Characteristic Curve",
    "baseline_wape": "WAPE achieved by the 7-day rolling average baseline forecaster",
    "ml_wape": "WAPE achieved by the trained ML forecaster",
    "wape_improvement_pct": "Relative percentage error reduction of ML over rolling baseline",
    # KPIs
    "total_shortage_units": "Cumulative units of medicine requested by patients but unfulfilled",
    "total_wastage_units": "Cumulative units of expired medicine discarded",
    "total_transport_cost": "Total inter-facility logistics and vehicle costs incurred (INR)",
    "fill_rate": "Proportion of total patient demand successfully served without stockout",
    "stockout_days": "Total hospital-days where at least one critical medicine had zero inventory",
    "wastage_prevented_units": "Units saved from expiry compared to the 'Do Nothing' (no_transfer) policy",
    "wastage_prevented_inr": "Financial value of medicine saved from expiry (INR)",
    "stockouts_avoided": "Stockout events prevented compared to no_transfer",
    # State & Transfers
    "stock": "Current physical stock units on hand at the end of the day",
    "demand": "Observed patient prescription demand for the medicine on that day",
    "forecast": "Predicted demand for the upcoming decision horizon",
    "forecast_lower": "Lower bound of the 95% prediction interval",
    "forecast_upper": "Upper bound of the 95% prediction interval",
    "safety_stock": "Dynamic safety stock buffer computed from demand variance and lead time",
    "batch_id": "Unique manufacturing batch code",
    "expiry_day": "Simulation day on which the medicine batch expires",
    "days_to_expiry": "Number of days remaining until the batch reaches expiration",
    "is_at_risk": "Flag indicating whether batch will expire within threshold before local consumption",
    "from_hospital": "Dispatching hospital ID originating the transfer",
    "to_hospital": "Receiving hospital ID taking delivery of the transfer",
    "quantity": "Number of units transferred",
    "units": "Number of units",
    "transport_cost": "Cost incurred for the transfer based on distance and mode (INR)",
    "distance_km": "Euclidean / road distance between facilities in kilometers",
    "reason": "Explainability narrative justifying why the transfer was executed",
}


def get_data_dir() -> Path:
    serving = REPO_ROOT / "data" / "serving"
    sample = REPO_ROOT / "data" / "sample"
    if serving.exists() and any(serving.glob("*.parquet")):
        return serving
    return sample


def generate_data_dictionary():
    data_dir = get_data_dir()
    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DOCS_DIR / "DATA_DICTIONARY.md"

    lines = [
        "# Data Dictionary: Serving Parquet Tables",
        "",
        "> **Notice:** This document is automatically generated by `tools/make_data_dictionary.py` directly from the Parquet files in `"
        + str(data_dir.relative_to(REPO_ROOT)).replace("\\", "/")
        + "`. It reflects the exact schemas, column types, and data properties in the serving layer.",
        "",
        f"**Source Directory:** `{data_dir.relative_to(REPO_ROOT).as_posix()}`  ",
        f"**Table Count:** {len(parquet_files)}  ",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
    ]

    for p in parquet_files:
        tbl_name = p.stem
        lines.append(f"- [{tbl_name}](#table-{tbl_name.replace('_', '-')})")

    lines.append("")
    lines.append("---")
    lines.append("")

    con = duckdb.connect()

    for p in parquet_files:
        tbl_name = p.stem
        rel_path = p.relative_to(REPO_ROOT).as_posix()
        file_size_kb = p.stat().st_size / 1024

        df_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p.as_posix()}')").fetchdf()
        row_count = con.execute(f"SELECT count(*) FROM read_parquet('{p.as_posix()}')").fetchone()[0]

        lines.append(f"### Table: `{tbl_name}`")
        lines.append("")
        lines.append(f"- **Path:** `{rel_path}`")
        lines.append(f"- **Row Count:** {row_count:,}")
        lines.append(f"- **File Size:** {file_size_kb:.1f} KB")
        lines.append("")
        lines.append("| Column Name | Data Type | Nullable | Description |")
        lines.append("| :--- | :--- | :---: | :--- |")

        for _, row in df_info.iterrows():
            col_name = row["column_name"]
            col_type = row["column_type"]
            nullable = row.get("null", "YES")
            desc = COLUMN_DESCRIPTIONS.get(col_name, f"System attribute `{col_name}`")
            lines.append(f"| `{col_name}` | `{col_type}` | {nullable} | {desc} |")

        lines.append("")
        lines.append("---")
        lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {out_file} with {len(parquet_files)} tables.")


if __name__ == "__main__":
    generate_data_dictionary()
