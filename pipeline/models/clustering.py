"""Hospital clustering module using K-Means on throughput, size, location, and rurality."""
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class HospitalClusterer:
    """Clusters hospitals based on throughput, bed capacity, geographic coordinates, and rurality."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.cluster_names: Dict[int, str] = {}

    def fit_predict(self, curated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract hospital-level feature profiles and perform K-Means clustering.
        Returns DataFrame with [hospital_id, cluster_id, cluster_label, avg_daily_volume, bed_count, is_rural].
        """
        if curated_df.empty:
            return pd.DataFrame()

        # Group by hospital to build structural & throughput profile
        profiles = (
            curated_df.groupby("hospital_id")
            .agg(
                avg_daily_volume=("observed_consumption", "mean"),
                total_stock=("observed_stock", "mean"),
                bed_count=("bed_count", "first"),
                is_rural=("is_rural", "first"),
                latitude=("latitude", "first"),
                longitude=("longitude", "first"),
            )
            .reset_index()
        )

        n_samples = len(profiles)
        actual_k = min(self.n_clusters, n_samples)
        if actual_k <= 1:
            profiles["cluster_id"] = 0
            profiles["cluster_label"] = "General Facility"
            return profiles

        feature_cols = ["avg_daily_volume", "total_stock", "bed_count", "is_rural", "latitude", "longitude"]
        X = profiles[feature_cols].fillna(0.0)
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(n_clusters=actual_k, random_state=self.random_state, n_init=10)
        profiles["cluster_id"] = self.kmeans.fit_predict(X_scaled)

        # Assign descriptive cluster names based on bed count and volume
        cluster_means = profiles.groupby("cluster_id")["avg_daily_volume"].mean().sort_values()
        label_map = {}
        sorted_c_ids = list(cluster_means.index)
        if len(sorted_c_ids) >= 3:
            label_map[sorted_c_ids[0]] = "Peripheral PHC / Clinic"
            label_map[sorted_c_ids[1]] = "Sub-District / Taluk Hospital"
            label_map[sorted_c_ids[2]] = "Tertiary / District Medical Center"
        elif len(sorted_c_ids) == 2:
            label_map[sorted_c_ids[0]] = "Peripheral Facility"
            label_map[sorted_c_ids[1]] = "Regional Hub Hospital"
        else:
            label_map[sorted_c_ids[0]] = "General Facility"

        profiles["cluster_label"] = profiles["cluster_id"].map(label_map)
        return profiles
