"""Unified command-line interface for Smart Medicine Redistribution System."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from pipeline.doctor import run_doctor


def get_default_profile() -> str:
    """Return default profile from environment variable or 'dev'."""
    return os.environ.get("MEDSYS_PROFILE", "dev")


def run_demo_workflow(profile: Optional[str] = None, seeds: int = 1, verbose: bool = False) -> int:
    """
    Execute full end-to-end pipeline demo with step-by-step progress and timing:
    1. Prepare & ETL
    2. Train ML models & track in MLflow
    3. Run closed-loop scenario matrix
    4. Export serving tables
    """
    profile = profile or get_default_profile()
    start_total = time.time()

    print("=" * 65)
    print(f" SMART MEDICINE REDISTRIBUTION SYSTEM — DEMO PIPELINE")
    print(f" Profile: {profile.upper()} | Seeds per scenario: {seeds}")
    print("=" * 65)

    estimated_times = {
        "dev": "~4 minutes (10 hospitals, 30 medicines, 14 days)",
        "demo": "~15 minutes (30 hospitals, 60 medicines, 30 days)",
        "full": "~45 minutes (50 hospitals, 200 medicines, 60 days)",
    }
    print(f" Estimated completion time for '{profile}' profile: {estimated_times.get(profile, '3-10 minutes')}")
    print("-" * 65)

    stages = [
        ("Stage 1/4: Data Generation & ETL", _run_prepare),
        ("Stage 2/4: ML Forecasting & Baseline Training", _run_train),
        ("Stage 3/4: Closed-Loop Scenario Optimization", _run_scenarios),
        ("Stage 4/4: Export Published Serving Parquet", _run_export),
    ]

    for stage_name, stage_fn in stages:
        print(f"\n>> {stage_name}...")
        t0 = time.time()
        try:
            summary_info = stage_fn(profile, seeds)
            elapsed = time.time() - t0
            print(f"   [DONE] Completed in {elapsed:.1f}s. {summary_info}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"   [FAILED] Failed after {elapsed:.1f}s: {e}")
            print(f"\n   Troubleshooting Hint:")
            print(f"   - Run 'python -m pipeline doctor' to check your environment.")
            print(f"   - Ensure write permissions to the 'data/' directory.")
            print(f"   - If PySpark failed, set ETL_BACKEND=duckdb to use local fallback.")
            if verbose:
                print("\nFull traceback:")
                traceback.print_exc()
            else:
                print("   (Run with --verbose to view the complete error traceback)")
            return 1

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 65)
    print(f" [SUCCESS] End-to-end demo pipeline completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 65)
    print("\nTo view the interactive executive dashboard, run:")
    print("   streamlit run app/Home.py")
    print("or:")
    print("   python -m pipeline app")
    print()
    return 0


def _run_prepare(profile: str, seeds: int) -> str:
    from pipeline.prepare import prepare_profile_data
    prepare_profile_data(profile=profile)
    out_dir = Path(f"data/{profile}")
    curated_p = out_dir / "curated" / "curated_daily.parquet"
    if curated_p.exists():
        import pyarrow.parquet as pq
        meta = pq.read_metadata(curated_p)
        return f"{meta.num_rows:,} curated records written to {curated_p}."
    return f"Prepared data generated at {out_dir}."


def _run_train(profile: str, seeds: int) -> str:
    from pipeline.train import train_models_pipeline
    train_models_pipeline(profile=profile)
    models_dir = Path(f"data/{profile}/models")
    return f"Models, baselines, and evaluation reports saved at {models_dir}."


def _run_scenarios(profile: str, seeds: int) -> str:
    from pipeline.run_scenario import run_scenario_matrix
    res = run_scenario_matrix(profile=profile, seeds=seeds)
    runs_count = len(res.get("summary", []))
    return f"Evaluated {runs_count} scenario-policy configurations."


def _run_export(profile: str, seeds: int) -> str:
    from pipeline.export import export_serving_data
    export_serving_data(profile=profile, target_dir="data/serving")
    # Also keep data/sample synchronized if on dev profile
    if profile == "dev":
        export_serving_data(profile=profile, target_dir="data/sample")
    serving_files = list(Path("data/serving").glob("*.parquet"))
    return f"{len(serving_files)} serving Parquet tables published to data/serving/."


def run_app() -> int:
    """Launch the Streamlit executive dashboard."""
    app_path = Path("app/Home.py")
    if not app_path.exists():
        print(f"Error: {app_path} not found.")
        return 1
    print("Launching Streamlit dashboard (app/Home.py)...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        return 0


def main():
    default_profile = get_default_profile()
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Smart Medicine Redistribution System CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # demo subcommand
    demo_p = subparsers.add_parser("demo", help="Run full end-to-end pipeline demo")
    demo_p.add_argument("--profile", default=default_profile, choices=["dev", "demo", "full"], help=f"Execution profile (default: {default_profile})")
    demo_p.add_argument("--seeds", type=int, default=1, help="Number of random seeds per scenario (default: 1)")
    demo_p.add_argument("--verbose", action="store_true", help="Show full stack traces on error")

    # doctor subcommand
    doc_p = subparsers.add_parser("doctor", help="Run environment diagnostic checks")
    doc_p.add_argument("--verbose", action="store_true", help="Show detailed diagnostic information")

    # app subcommand
    subparsers.add_parser("app", help="Launch the Streamlit executive dashboard")

    # prepare subcommand
    prep_p = subparsers.add_parser("prepare", help="Generate raw data and run ETL")
    prep_p.add_argument("--profile", default=default_profile, choices=["dev", "demo", "full"])

    # train subcommand
    train_p = subparsers.add_parser("train", help="Train models and log to MLflow")
    train_p.add_argument("--profile", default=default_profile, choices=["dev", "demo", "full"])

    # run_scenario subcommand
    sc_p = subparsers.add_parser("run_scenario", help="Execute closed-loop scenarios")
    sc_p.add_argument("--profile", default=default_profile, choices=["dev", "demo", "full"])
    sc_p.add_argument("--scenario", default="all", help="Scenario name or 'all'")
    sc_p.add_argument("--policy", default="all", help="Policy name or 'all'")
    sc_p.add_argument("--seeds", type=int, default=1)

    # export subcommand
    exp_p = subparsers.add_parser("export", help="Publish serving Parquet tables")
    exp_p.add_argument("--profile", default=default_profile, choices=["dev", "demo", "full"])
    exp_p.add_argument("--target-dir", default="data/serving")

    args = parser.parse_args()

    if args.command == "demo":
        sys.exit(run_demo_workflow(profile=args.profile, seeds=args.seeds, verbose=args.verbose))
    elif args.command == "doctor":
        sys.exit(run_doctor(verbose=args.verbose))
    elif args.command == "app":
        sys.exit(run_app())
    elif args.command == "prepare":
        from pipeline.prepare import prepare_profile_data
        prepare_profile_data(profile=args.profile)
    elif args.command == "train":
        from pipeline.train import train_models_pipeline
        train_models_pipeline(profile=args.profile)
    elif args.command == "run_scenario":
        from pipeline.run_scenario import run_scenario_matrix
        scs = None if args.scenario == "all" else [args.scenario]
        pols = None if args.policy == "all" else [args.policy]
        run_scenario_matrix(profile=args.profile, scenarios_to_run=scs, policies_to_run=pols, seeds=args.seeds)
    elif args.command == "export":
        from pipeline.export import export_serving_data
        export_serving_data(profile=args.profile, target_dir=args.target_dir)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
