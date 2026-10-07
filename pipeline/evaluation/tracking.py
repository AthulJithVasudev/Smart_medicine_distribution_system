"""MLflow experiment tracking and Champion/Challenger registry."""
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Manages MLflow tracking runs with SQLite backend and parent/child experiment hierarchy."""

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", experiment_name: str = "medicine_distribution"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def hash_config(self, config_dict: Dict[str, Any]) -> str:
        """Create deterministic MD5 hash of config dictionary."""
        s = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

    def log_experiment_run(
        self,
        run_name: str,
        model_type: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts_df: Optional[pd.DataFrame] = None,
        artifact_filename: str = "predictions.parquet",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Log a tracked experiment run with parameters, metrics, and Parquet artifacts."""
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Log tags
            mlflow.set_tag("model_type", model_type)
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, str(v))

            # Log params (flattened)
            for p_k, p_v in params.items():
                if isinstance(p_v, (dict, list)):
                    mlflow.log_param(p_k, json.dumps(p_v)[:250])
                else:
                    mlflow.log_param(p_k, str(p_v)[:250])

            # Log metrics
            for m_k, m_v in metrics.items():
                if isinstance(m_v, (int, float)):
                    mlflow.log_metric(m_k, float(m_v))

            # Save per-series results as Parquet artifact
            if artifacts_df is not None and not artifacts_df.empty:
                temp_path = Path("scratch")
                temp_path.mkdir(exist_ok=True)
                file_path = temp_path / artifact_filename
                artifacts_df.to_parquet(file_path, index=False)
                mlflow.log_artifact(str(file_path))
                if file_path.exists():
                    file_path.unlink()

            return run_id


class ChampionChallengerRegistry:
    """
    Manages champion/challenger model promotion.
    A new candidate model becomes Champion ONLY if it beats the incumbent
    on the primary backtest metric (e.g. lower WAPE or higher F1).
    """

    def __init__(self, registry_file: str = "data/champion_models.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def evaluate_candidate(
        self,
        task_name: str,
        candidate_model_type: str,
        candidate_metric_value: float,
        metric_name: str = "wape",
        lower_is_better: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate against champion.
        Returns:
            {"promoted": bool, "champion_type": str, "champion_metric": float}
        """
        current_champ = self.registry.get(task_name)

        if current_champ is None:
            # No current champion; automatic promotion
            promoted = True
        else:
            current_metric = current_champ["metric_value"]
            if lower_is_better:
                promoted = candidate_metric_value < current_metric
            else:
                promoted = candidate_metric_value > current_metric

        if promoted:
            self.registry[task_name] = {
                "model_type": candidate_model_type,
                "metric_name": metric_name,
                "metric_value": candidate_metric_value,
                "metadata": metadata or {},
            }
            self._save()

        champ = self.registry[task_name]
        return {
            "promoted": promoted,
            "champion_type": champ["model_type"],
            "champion_metric": champ["metric_value"],
        }
