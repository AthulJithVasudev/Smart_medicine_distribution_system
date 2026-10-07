"""Closed-loop scenario execution engine comparing policies across multiple seeds."""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import numpy as np
import pandas as pd

from pipeline.simulator.engine import Simulator
from pipeline.etl.pipeline import run_etl
from pipeline.etl.cleaning import clean_observations
from pipeline.etl.features import engineer_features
from pipeline.models.forecasting import HybridMedicineForecaster, train_forecasters
from pipeline.models.classification import RiskClassifier
from pipeline.models.clustering import HospitalClusterer
from pipeline.optimizer.solver import RedistributionOptimizer
from pipeline.optimizer.rolling import RollingHorizonOptimizer
from pipeline.evaluation.baselines import NoTransferPolicy, GreedyNearestSurplusPolicy
from pipeline.evaluation.metrics import calculate_optimization_metrics
from pipeline.utils.scenario_loader import load_all_scenarios, get_scenario_catalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def execute_closed_loop_run(
    config_path: str,
    scenario_cfg: Dict[str, Any],
    policy_name: str,
    seed: int,
    horizon_days: int,
    trained_forecaster: Optional[HybridMedicineForecaster] = None,
) -> Dict[str, Any]:
    """
    Execute a single closed-loop scenario run for a specific (scenario, policy, seed) tuple.
    Guarantees that actions taken today directly impact tomorrow's simulated inventory.
    """
    # 1. Fresh simulator from config + seed
    sim = Simulator(config=config_path, seed=seed)
    sim.set_scenario(scenario_cfg)

    # 2. Setup policy callable
    if policy_name == "optimizer":
        opt_solver = RedistributionOptimizer(
            fixed_transport_cost=sim.config.get("optimizer", {}).get("fixed_transport_cost", 100.0),
            cost_per_km=sim.config.get("optimizer", {}).get("cost_per_km", 2.5),
            horizon_days=sim.config.get("optimizer", {}).get("horizon_days", 14),
        )
        policy_fn = RollingHorizonOptimizer(optimizer=opt_solver)
    elif policy_name == "greedy":
        policy_fn = GreedyNearestSurplusPolicy()
    elif policy_name == "no_transfer":
        policy_fn = NoTransferPolicy()
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    replay_records: List[Dict[str, Any]] = []
    kpi_records: List[Dict[str, Any]] = []

    # 3. Closed-loop daily execution
    for current_day in range(horizon_days):
        # Current observations
        obs_df = sim.observations()

        # Run ETL on current observations
        forecast_map: Dict[Any, float] = {}
        pred_df = None
        if trained_forecaster is not None and not obs_df.empty:
            cleaned_daily = clean_observations(obs_df)
            curated_daily = engineer_features(
                cleaned_daily,
                hospitals_map=sim.hospitals,
                medicines_map=sim.medicines,
            )
            latest_curated = curated_daily[curated_daily["report_day"] == curated_daily["report_day"].max()].copy()
            if not latest_curated.empty:
                pred_df = trained_forecaster.predict(latest_curated)
                for _, pr in pred_df.iterrows():
                    forecast_map[(pr["hospital_id"], pr["medicine_id"])] = pr["predicted_demand"]

        # Run policy to obtain transfers for today
        actions = policy_fn(sim)

        # Record replay data before advancing step (forecast issued at current_day)
        # We will match actual demand after sim.step()
        day_forecasts = {}
        if pred_df is not None and not pred_df.empty:
            for _, pr in pred_df.iterrows():
                day_forecasts[(pr["hospital_id"], pr["medicine_id"])] = {
                    "forecast": pr["predicted_demand"],
                    "lower": pr["lower_bound"],
                    "upper": pr["upper_bound"],
                }

        # STEP SIMULATION (APPLY TRANSFERS & ADVANCE CLOCK)
        sim.step(actions)

        # Inspect resulting true ground truth for this day
        gt_day = sim.get_hidden_ground_truth(up_to_day=current_day)
        if not gt_day.empty:
            gt_latest = gt_day[gt_day["day"] == current_day]
            for _, gt_row in gt_latest.iterrows():
                h_id = gt_row["hospital_id"]
                m_id = gt_row["medicine_id"]
                fc_info = day_forecasts.get((h_id, m_id), {
                    "forecast": float(gt_row["true_demand"]),
                    "lower": float(gt_row["true_demand"] * 0.7),
                    "upper": float(gt_row["true_demand"] * 1.3),
                })
                replay_records.append({
                    "scenario": scenario_cfg["name"],
                    "policy": policy_name,
                    "seed": seed,
                    "day": current_day,
                    "hospital_id": h_id,
                    "medicine_id": m_id,
                    "forecast": fc_info["forecast"],
                    "lower_bound": fc_info["lower"],
                    "upper_bound": fc_info["upper"],
                    "actual_demand": gt_row["true_demand"],
                    "unmet_demand": gt_row["unmet_demand"],
                    "stock_remaining": gt_row["current_stock"],
                    "expired_units": gt_row["expired_units"],
                })

            # Calculate daily KPI snapshot
            day_metrics = calculate_optimization_metrics(
                gt_latest,
                prices_map={m.medicine_id: m.price for m in sim.medicines.values()}
            )
            day_metrics["scenario"] = scenario_cfg["name"]
            day_metrics["policy"] = policy_name
            day_metrics["seed"] = seed
            day_metrics["day"] = current_day
            kpi_records.append(day_metrics)

    # 4. Overall run KPIs
    full_gt = sim.get_hidden_ground_truth()
    full_transfers = sim.get_transfers_log()
    overall_kpis = calculate_optimization_metrics(
        full_gt,
        full_transfers,
        prices_map={m.medicine_id: m.price for m in sim.medicines.values()}
    )
    overall_kpis["scenario"] = scenario_cfg["name"]
    overall_kpis["policy"] = policy_name
    overall_kpis["seed"] = seed

    return {
        "scenario": scenario_cfg["name"],
        "policy": policy_name,
        "seed": seed,
        "overall_kpis": overall_kpis,
        "kpis_daily": pd.DataFrame(kpi_records),
        "replay_df": pd.DataFrame(replay_records),
        "transfers_df": full_transfers,
        "state_daily": full_gt,
    }


def run_scenario_matrix(
    profile: str = "dev",
    scenarios_to_run: Optional[List[str]] = None,
    policies_to_run: Optional[List[str]] = None,
    seeds: int = 3,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute full scenario matrix (Scenarios x Policies x Seeds).
    Computes mean and spread (standard deviation) across seeds.
    """
    config_path = f"configs/{profile}.yaml"
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    horizon_days = config.get("simulation", {}).get("scenario_days", 14)
    out_dir = Path(output_dir or config.get("data_dir", f"data/{profile}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scenarios = load_all_scenarios()
    target_scenarios = scenarios_to_run or list(all_scenarios.keys())
    target_policies = policies_to_run or ["optimizer", "greedy", "no_transfer"]

    # Pre-train baseline forecaster on burn-in historical data
    burnin_sim = Simulator(config=config_path, seed=42)
    obs_burnin, _ = burnin_sim.run(days=14)
    curated_burnin = run_etl(
        base_dir=str(out_dir / "burnin"),
        obs_df=obs_burnin,
        hospitals_map=burnin_sim.hospitals,
        medicines_map=burnin_sim.medicines,
    )
    forecaster = train_forecasters(curated_burnin, with_opd=True)

    all_runs_summary = []
    all_replay = []
    all_transfers = []
    all_daily_kpis = []
    all_daily_states = []

    for sc_name in target_scenarios:
        if sc_name not in all_scenarios:
            logger.warning(f"Scenario {sc_name} not found, skipping.")
            continue
        sc_cfg = all_scenarios[sc_name]

        for pol_name in target_policies:
            seed_kpis = []
            logger.info(f"Executing: Scenario={sc_name} | Policy={pol_name} | Seeds={seeds}")

            for s_idx in range(seeds):
                current_seed = 42 + s_idx * 10
                run_res = execute_closed_loop_run(
                    config_path=config_path,
                    scenario_cfg=sc_cfg,
                    policy_name=pol_name,
                    seed=current_seed,
                    horizon_days=horizon_days,
                    trained_forecaster=forecaster,
                )
                seed_kpis.append(run_res["overall_kpis"])
                all_replay.append(run_res["replay_df"])
                all_transfers.append(run_res["transfers_df"])
                all_daily_kpis.append(run_res["kpis_daily"])
                all_daily_states.append(run_res["state_daily"])

            # Compute mean and standard deviation across seeds
            kpi_df = pd.DataFrame(seed_kpis)
            summary_row = {
                "scenario": sc_name,
                "policy": pol_name,
                "seeds_count": seeds,
                "wastage_units_mean": round(float(kpi_df["wastage_units"].mean()), 2),
                "wastage_units_std": round(float(kpi_df["wastage_units"].std()), 2) if seeds > 1 else 0.0,
                "stockout_days_mean": round(float(kpi_df["stockout_days"].mean()), 2),
                "stockout_days_std": round(float(kpi_df["stockout_days"].std()), 2) if seeds > 1 else 0.0,
                "fill_rate_mean": round(float(kpi_df["fill_rate"].mean()), 4),
                "fill_rate_std": round(float(kpi_df["fill_rate"].std()), 4) if seeds > 1 else 0.0,
                "transport_cost_mean": round(float(kpi_df["transport_cost"].mean()), 2),
                "transport_cost_std": round(float(kpi_df["transport_cost"].std()), 2) if seeds > 1 else 0.0,
            }
            all_runs_summary.append(summary_row)

    summary_df = pd.DataFrame(all_runs_summary)
    replay_concat = pd.concat(all_replay, ignore_index=True) if all_replay else pd.DataFrame()
    transfers_concat = pd.concat(all_transfers, ignore_index=True) if all_transfers else pd.DataFrame()
    daily_kpis_concat = pd.concat(all_daily_kpis, ignore_index=True) if all_daily_kpis else pd.DataFrame()

    # Save outputs to Parquet
    matrix_dir = out_dir / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(matrix_dir / "summary.parquet", index=False)
    replay_concat.to_parquet(matrix_dir / "replay.parquet", index=False)
    transfers_concat.to_parquet(matrix_dir / "transfers.parquet", index=False)
    daily_kpis_concat.to_parquet(matrix_dir / "daily_kpis.parquet", index=False)

    logger.info("Scenario matrix completed successfully.")
    return {
        "summary": summary_df,
        "replay": replay_concat,
        "transfers": transfers_concat,
        "daily_kpis": daily_kpis_concat,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Smart Medicine Redistribution Scenario Engine")
    parser.add_argument("--profile", default="dev", choices=["dev", "demo", "full"], help="Profile to use")
    parser.add_argument("--scenario", default="normal", help="Scenario name or 'all'")
    parser.add_argument("--policy", default="optimizer", help="Policy name ('optimizer', 'greedy', 'no_transfer', or 'all')")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds to evaluate")
    parser.add_argument("--output-dir", default=None, help="Directory to save output Parquet tables")
    args = parser.parse_args()

    scenarios = None if args.scenario == "all" else [args.scenario]
    policies = None if args.policy == "all" else [args.policy]

    run_scenario_matrix(
        profile=args.profile,
        scenarios_to_run=scenarios,
        policies_to_run=policies,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
