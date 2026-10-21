"""Single source of truth for all user-facing plain-language labels, glossaries, and color badges."""
from typing import Dict, Any, Tuple

# Policy Labels
POLICY_LABELS: Dict[str, str] = {
    "optimizer": "Smart plan (AI/MILP)",
    "greedy": "Nearest surplus (Heuristic)",
    "no_transfer": "Do nothing (Baseline)",
}

POLICY_DESCRIPTIONS: Dict[str, str] = {
    "optimizer": "Mathematical optimization balancing urgent deficit, wastage reduction, and transport distance.",
    "greedy": "Transfers stock from the geographically nearest hospital with stock, without future demand forecasting.",
    "no_transfer": "No redistribution transfers between facilities; hospital relies solely on periodic supplier orders.",
}

# Reverse lookup for selectboxes
LABEL_TO_POLICY: Dict[str, str] = {v: k for k, v in POLICY_LABELS.items()}


# Scenario Catalog & Tags
SCENARIO_INFO: Dict[str, Dict[str, str]] = {
    "normal": {
        "display_name": "Normal Operations",
        "tag": "Seen Before",
        "description": "Baseline hospital operations with typical outpatient demand and standard supply replenishment.",
        "badge_color": "#10b981",
    },
    "monsoon_surge": {
        "display_name": "Monsoon Seasonal Surge",
        "tag": "Seen Before",
        "description": "Seasonal elevation in fever and gastrointestinal cases across district taluk hospitals.",
        "badge_color": "#3b82f6",
    },
    "dengue_2x": {
        "display_name": "Moderate Dengue Outbreak (2x)",
        "tag": "Seen Before",
        "description": "Regional surge doubling antipyretic, analgesic, and IV fluid requirements.",
        "badge_color": "#f59e0b",
    },
    "dengue_4x": {
        "display_name": "Severe Dengue Epidemic (4x)",
        "tag": "New Situation",
        "description": "Extreme 4x demand surge representing an unprecedented distribution shift not seen in training history.",
        "badge_color": "#ef4444",
    },
    "supplier_delay": {
        "display_name": "Central Supplier Disruption (+10 Days)",
        "tag": "New Situation",
        "description": "Supply chain shock delaying bulk deliveries by 10 days across all facilities.",
        "badge_color": "#8b5cf6",
    },
    "low_stock": {
        "display_name": "Constrained Network Stock (30% Baseline)",
        "tag": "Seen Before",
        "description": "Severe network-wide starting inventory deficit testing emergency allocation under tight scarcity.",
        "badge_color": "#f97316",
    },
    "rural_access": {
        "display_name": "Rural Transit Disruption (2x Road Time)",
        "tag": "New Situation",
        "description": "Flooding and infrastructure damage doubling travel times to peripheral rural health centers.",
        "badge_color": "#ec4899",
    },
}

LABEL_TO_SCENARIO: Dict[str, str] = {
    v["display_name"]: k for k, v in SCENARIO_INFO.items()
}


# Status Badges: Text + Color (Never color alone)
STATUS_CONFIG: Dict[str, Dict[str, str]] = {
    "critical_shortage": {
        "label": "Critical Shortage",
        "color": "#ef4444",
        "badge": "🔴 Critical Shortage (≤ 2 days stock)",
    },
    "low_stock": {
        "label": "Low Inventory Alert",
        "color": "#f59e0b",
        "badge": "🟠 Low Inventory (3–5 days stock)",
    },
    "normal": {
        "label": "Normal Stock Level",
        "color": "#10b981",
        "badge": "🟢 Normal (6–14 days stock)",
    },
    "surplus": {
        "label": "Transferable Surplus",
        "color": "#06b6d4",
        "badge": "🔵 Surplus (> 14 days stock)",
    },
    "expiry_risk": {
        "label": "Expiry Risk",
        "color": "#e11d48",
        "badge": "⚠️ Expiry Risk (≤ 30 days life)",
    },
}


# Project Glossary with Tooltips
GLOSSARY: Dict[str, str] = {
    "Smart plan": "The AI/MILP optimization algorithm that simultaneously minimizes stockouts, prevents medicine expiry, and limits transport costs.",
    "Nearest surplus": "Heuristic baseline that ships from the closest hospital with extra stock without considering shelf-life or future forecast.",
    "Do nothing": "Operational baseline where no transfers take place between hospitals; stockouts persist until the next regular supplier order.",
    "FEFO": "First-Expired, First-Out: dispensing and transferring medicines nearing expiration before newer batches.",
    "Receiving Cap": "A strict mathematical limit ensuring a hospital is never sent more stock than its forecast need, preventing dumping.",
    "Safety Stock": "A reserve buffer (typically 5 days of demand) maintained to absorb unexpected patient surges.",
    "WAPE": "Weighted Absolute Percentage Error: an industry standard accuracy measure for demand forecasting.",
    "Fill Rate": "Percentage of total patient medicine demand fulfilled immediately without deficit.",
    "Censored Demand": "Recorded hospital dispensing that only reflects available stock, hiding true unfulfilled patient need during stockouts.",
    "Oracle Bound": "Theoretical perfect performance calculated with 100% omniscient future knowledge; defines the mathematical upper limit.",
}


def get_scenario_display(scenario_key: str) -> str:
    """Return plain-language scenario display name."""
    info = SCENARIO_INFO.get(scenario_key)
    return info["display_name"] if info else scenario_key.replace("_", " ").title()


def get_scenario_tag(scenario_key: str) -> str:
    """Return 'Seen Before' or 'New Situation' tag."""
    info = SCENARIO_INFO.get(scenario_key)
    return info["tag"] if info else "Scenario"


def get_policy_display(policy_key: str) -> str:
    """Return plain-language policy name."""
    return POLICY_LABELS.get(policy_key, policy_key.replace("_", " ").title())
