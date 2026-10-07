"""Scenario management, catalog loading, and seen/unseen distribution shift categorization."""
from pathlib import Path
from typing import Dict, List, Any
import yaml


def load_all_scenarios(scenarios_dir: str = "scenarios") -> Dict[str, Dict[str, Any]]:
    """Load all scenario YAML files from scenarios directory."""
    s_dir = Path(scenarios_dir)
    scenarios = {}
    for f in s_dir.glob("*.yaml"):
        with open(f, "r") as fp:
            data = yaml.safe_load(fp)
            scenarios[data["name"]] = data
    return scenarios


def get_scenario_catalog(scenarios_dir: str = "scenarios") -> List[Dict[str, Any]]:
    """Return catalog list with seen/unseen metadata."""
    scenarios = load_all_scenarios(scenarios_dir)
    catalog = []
    for name, data in scenarios.items():
        catalog.append({
            "name": name,
            "type": data.get("type", "seen"),
            "description": data.get("description", ""),
            "outbreak_multiplier": data.get("outbreak_multiplier", 1.0),
            "is_monsoon": data.get("is_monsoon", False),
            "supplier_delay_days": data.get("supplier_delay_days", 0),
            "initial_stock_ratio": data.get("initial_stock_ratio", 1.0),
            "rural_transit_multiplier": data.get("rural_transit_multiplier", 1.0),
        })
    return sorted(catalog, key=lambda x: (x["type"] != "seen", x["name"]))
