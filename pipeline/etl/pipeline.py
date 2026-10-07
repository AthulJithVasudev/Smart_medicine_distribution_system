"""ETL pipeline execution script coordinating raw storage, cleaning, curation, and DuckDB querying."""
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from pipeline.etl.storage import ParquetStorage, query_duckdb
from pipeline.etl.cleaning import clean_observations
from pipeline.etl.features import engineer_features
from pipeline.etl.spark_etl import run_spark_transform
from pipeline.simulator.entities import Hospital, Medicine


def run_etl(
    base_dir: str = "data/dev",
    obs_df: Optional[pd.DataFrame] = None,
    hospitals_map: Optional[Dict[str, Hospital]] = None,
    medicines_map: Optional[Dict[str, Medicine]] = None,
    shuffle_partitions: int = 8,
) -> pd.DataFrame:
    """
    Execute full ETL pipeline:
    1. Read or save raw observations in partitioned Parquet format.
    2. Apply cleaning (dedup, imputation, unit correction).
    3. Run PySpark local-mode transform if Spark is available, otherwise DuckDB/Arrow engine.
    4. Compute temporal lag, rolling, and health features without future leakage.
    5. Save curated table to Parquet.
    6. Return curated DataFrame.
    """
    storage = ParquetStorage(base_dir=base_dir)

    if obs_df is not None and not obs_df.empty:
        storage.save_raw_observations(obs_df, partition_col="report_day")

    raw_data = storage.load_raw_observations()
    if raw_data.empty:
        if obs_df is not None and not obs_df.empty:
            raw_data = obs_df.copy()
        else:
            return pd.DataFrame()

    # Step 1: Cleaning
    cleaned_df = clean_observations(raw_data)

    # Step 2: Try Spark ETL if available, else engineer_features
    spark_res = run_spark_transform(
        cleaned_df,
        hospitals_map=hospitals_map,
        medicines_map=medicines_map,
        shuffle_partitions=shuffle_partitions,
    )
    if spark_res is not None and not spark_res.empty:
        # Finalize domain features
        curated_df = engineer_features(spark_res, hospitals_map=hospitals_map, medicines_map=medicines_map)
    else:
        # DuckDB / pandas feature engine
        curated_df = engineer_features(cleaned_df, hospitals_map=hospitals_map, medicines_map=medicines_map)

    # Step 3: Save curated Parquet
    storage.save_curated_data(curated_df, filename="curated_daily.parquet")

    return curated_df
