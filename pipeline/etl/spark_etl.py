"""PySpark local-mode ETL module for large-scale data transformation and feature extraction."""
import logging
from typing import Optional, Dict
import pandas as pd
from pipeline.simulator.entities import Hospital, Medicine

logger = logging.getLogger(__name__)


def try_get_spark_session(shuffle_partitions: int = 8):
    """
    Attempt to create a local-mode PySpark session with 8-16 shuffle partitions.
    Returns SparkSession if Java/PySpark environment is fully functional, else None.
    If ETL_BACKEND is set to 'duckdb', skips Spark JVM initialization entirely.
    """
    import os
    if os.environ.get("ETL_BACKEND", "spark").lower() == "duckdb":
        logger.info("ETL_BACKEND=duckdb configured. Using DuckDB/Arrow engine.")
        return None

    try:
        from pyspark.sql import SparkSession
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("MedicineRedistributionETL")
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
            .config("spark.default.parallelism", str(shuffle_partitions))
            .getOrCreate()
        )
        return spark
    except Exception as e:
        logger.warning(f"PySpark JVM initialization not available locally ({e}). Using optimized DuckDB/Arrow engine.")
        return None


def run_spark_transform(
    raw_df: pd.DataFrame,
    hospitals_map: Optional[Dict[str, Hospital]] = None,
    medicines_map: Optional[Dict[str, Medicine]] = None,
    shuffle_partitions: int = 8,
) -> Optional[pd.DataFrame]:
    """
    Execute ETL using PySpark local mode if Spark JVM is active.
    Transforms raw observations into curated dataset with windowed lags and rolling features.
    """
    spark = try_get_spark_session(shuffle_partitions=shuffle_partitions)
    if spark is None:
        return None

    try:
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        spark_df = spark.createDataFrame(raw_df)

        # 1. Deduplication using Window row_number
        dedup_window = Window.partitionBy("report_day", "hospital_id", "medicine_id").orderBy(
            F.when(F.col("provenance") == "OFFICIAL", 1)
            .when(F.col("provenance") == "SIMULATED", 2)
            .otherwise(3)
        )
        deduped = (
            spark_df.withColumn("_rn", F.row_number().over(dedup_window))
            .filter(F.col("_rn") == 1)
            .drop("_rn")
        )

        # 2. Imputation & Unit correction
        cleaned = (
            deduped.fillna(0, subset=["observed_stock", "observed_consumption", "nearest_expiry_days"])
            .withColumn("observed_stock", F.col("observed_stock").cast("int"))
            .withColumn("observed_consumption", F.col("observed_consumption").cast("int"))
        )

        # 3. Window Lags (strictly prior days)
        series_window = Window.partitionBy("hospital_id", "medicine_id").orderBy("report_day")
        curated_spark = (
            cleaned.withColumn("consumption_lag_1", F.lag("observed_consumption", 1).over(series_window))
            .withColumn("consumption_lag_2", F.lag("observed_consumption", 2).over(series_window))
            .withColumn("consumption_lag_3", F.lag("observed_consumption", 3).over(series_window))
            .withColumn("consumption_lag_7", F.lag("observed_consumption", 7).over(series_window))
            .withColumn("stock_lag_1", F.lag("observed_stock", 1).over(series_window))
            .withColumn("stock_lag_7", F.lag("observed_stock", 7).over(series_window))
            .withColumn("opd_total_lag_1", F.lag("opd_visits_total", 1).over(series_window))
            .withColumn("opd_fever_lag_1", F.lag("opd_visits_fever", 1).over(series_window))
            .withColumn("opd_gastro_lag_1", F.lag("opd_visits_gastro", 1).over(series_window))
        )

        result_pdf = curated_spark.toPandas()
        return result_pdf
    except Exception as e:
        logger.warning(f"Spark execution encountered: {e}. Falling back to DuckDB/Arrow engine.")
        return None
