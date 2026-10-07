"""Storage module for managing partitioned Parquet tables and DuckDB querying."""
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetStorage:
    """Manages raw and curated Parquet storage with partition management."""

    def __init__(self, base_dir: Union[str, Path] = "data/dev"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.curated_dir = self.base_dir / "curated"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.curated_dir.mkdir(parents=True, exist_ok=True)

    def save_raw_observations(self, obs_df: pd.DataFrame, partition_col: str = "report_day") -> Path:
        """
        Store raw observations into Parquet partitioned by partition_col (e.g. report_day).
        """
        if obs_df.empty:
            return self.raw_dir

        table = pa.Table.from_pandas(obs_df)
        pq.write_to_dataset(
            table,
            root_path=str(self.raw_dir),
            partition_cols=[partition_col],
            use_dictionary=True,
            compression="SNAPPY",
            existing_data_behavior="overwrite_or_ignore",
        )
        return self.raw_dir

    def load_raw_observations(self) -> pd.DataFrame:
        """Read all partitioned raw Parquet files."""
        if not self.raw_dir.exists() or not any(self.raw_dir.iterdir()):
            return pd.DataFrame()
        try:
            dataset = pq.ParquetDataset(str(self.raw_dir), use_legacy_dataset=False)
            table = dataset.read()
            return table.to_pandas()
        except Exception:
            # Fallback via DuckDB if direct pyarrow read has empty partition edge case
            query = f"SELECT * FROM read_parquet('{self.raw_dir}/**/*.parquet')"
            con = duckdb.connect()
            return con.execute(query).df()

    def save_curated_data(self, curated_df: pd.DataFrame, filename: str = "curated_daily.parquet") -> Path:
        """Save clean, feature-engineered curated table to Parquet."""
        out_path = self.curated_dir / filename
        table = pa.Table.from_pandas(curated_df)
        pq.write_table(table, str(out_path), compression="SNAPPY")
        return out_path

    def load_curated_data(self, filename: str = "curated_daily.parquet") -> pd.DataFrame:
        """Load curated Parquet table."""
        path = self.curated_dir / filename
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(str(path))


def query_duckdb(query: str, parquet_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Execute DuckDB SQL query. If parquet_path is provided, replaces 'PARQUET_TABLE'
    in the query with the parquet path formatted for read_parquet().
    """
    con = duckdb.connect(":memory:")
    formatted_query = query
    if parquet_path is not None:
        p_str = str(parquet_path).replace("\\", "/")
        if Path(parquet_path).is_dir():
            p_str = f"{p_str}/**/*.parquet"
        formatted_query = formatted_query.replace("PARQUET_TABLE", f"read_parquet('{p_str}')")
    return con.execute(formatted_query).df()
