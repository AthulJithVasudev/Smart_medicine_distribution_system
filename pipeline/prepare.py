"""Data preparation script: generates historical simulator observations and runs ETL into curated Parquet."""
import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd

from pipeline.simulator.engine import Simulator
from pipeline.etl.pipeline import run_etl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_profile_data(profile: str = "dev") -> None:
    """Generate initial raw dataset and execute ETL into curated Parquet."""
    config_path = f"configs/{profile}.yaml"
    with open(config_path, "r") as fp:
        cfg = yaml.safe_load(fp)

    out_dir = Path(cfg.get("data_dir", f"data/{profile}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    history_days = cfg.get("simulation", {}).get("history_days", 30)

    logger.info(f"Preparing dataset for profile: {profile} ({history_days} history days)")
    sim = Simulator(config=config_path, seed=cfg.get("seed", 42))

    # Run historical simulation
    obs_df, gt_df = sim.run(days=history_days)
    logger.info(f"Simulated {len(obs_df)} observation records across {len(sim.hospitals)} hospitals")

    # Save network metadata
    hospitals_df = pd.DataFrame([
        {
            "hospital_id": h.hospital_id,
            "name": h.name,
            "hospital_type": h.hospital_type,
            "is_rural": h.is_rural,
            "bed_count": h.bed_count,
            "latitude": h.latitude,
            "longitude": h.longitude,
            "district": h.district,
        }
        for h in sim.hospitals.values()
    ])
    hospitals_df.to_parquet(out_dir / "hospitals.parquet", index=False)

    medicines_df = pd.DataFrame([
        {
            "medicine_id": m.medicine_id,
            "name": m.name,
            "category": m.category,
            "price": m.price,
            "criticality": m.criticality,
            "shelf_life_days": m.shelf_life_days,
            "is_fever_related": m.is_fever_related,
            "is_gastro_related": m.is_gastro_related,
        }
        for m in sim.medicines.values()
    ])
    medicines_df.to_parquet(out_dir / "medicines.parquet", index=False)

    # Run ETL
    curated_df = run_etl(
        base_dir=str(out_dir),
        obs_df=obs_df,
        hospitals_map=sim.hospitals,
        medicines_map=sim.medicines,
    )
    logger.info(f"ETL completed: {len(curated_df)} curated records created at {out_dir / 'curated'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare historical dataset and run ETL")
    parser.add_argument("--profile", default="dev", choices=["dev", "demo", "full"], help="Profile to run")
    args = parser.parse_args()
    prepare_profile_data(profile=args.profile)


if __name__ == "__main__":
    main()
