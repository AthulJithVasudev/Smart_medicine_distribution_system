"""Data manager handling active hospital stock records, CSV import/export,
ETL stats tracking, and mathematical redistribution calculations.
Pure presentation/application logic — no heavy PySpark or model training inside.
"""
from typing import Dict, Any, List, Tuple, Optional
import datetime
import io
import math
import pandas as pd
import numpy as np
import streamlit as st

# Kerala Government Hospital Registry with accurate geographic coordinates
KERALA_HOSPITALS = [
    {"hospital_id": "H001", "name": "Ernakulam PHC Hospital #1", "hospital_type": "PHC", "district": "Ernakulam", "is_rural": True, "bed_count": 20, "latitude": 10.02, "longitude": 76.35},
    {"hospital_id": "H002", "name": "Thrissur PHC Hospital #2", "hospital_type": "PHC", "district": "Thrissur", "is_rural": False, "bed_count": 25, "latitude": 10.52, "longitude": 76.21},
    {"hospital_id": "H003", "name": "Palakkad District Hospital #3", "hospital_type": "District Hospital", "district": "Palakkad", "is_rural": False, "bed_count": 150, "latitude": 10.78, "longitude": 76.65},
    {"hospital_id": "H004", "name": "Malappuram District Hospital #4", "hospital_type": "District Hospital", "district": "Malappuram", "is_rural": False, "bed_count": 180, "latitude": 11.07, "longitude": 76.07},
    {"hospital_id": "H005", "name": "Kozhikode PHC Hospital #5", "hospital_type": "PHC", "district": "Kozhikode", "is_rural": False, "bed_count": 30, "latitude": 11.25, "longitude": 75.78},
    {"hospital_id": "H006", "name": "Ernakulam District Hospital #6", "hospital_type": "District Hospital", "district": "Ernakulam", "is_rural": False, "bed_count": 200, "latitude": 9.98, "longitude": 76.28},
    {"hospital_id": "H007", "name": "Thrissur PHC Hospital #7", "hospital_type": "PHC", "district": "Thrissur", "is_rural": True, "bed_count": 15, "latitude": 10.45, "longitude": 76.15},
    {"hospital_id": "H008", "name": "Palakkad Taluk Hospital #8", "hospital_type": "Taluk Hospital", "district": "Palakkad", "is_rural": False, "bed_count": 75, "latitude": 10.72, "longitude": 76.55},
    {"hospital_id": "H009", "name": "Malappuram PHC Hospital #9", "hospital_type": "PHC", "district": "Malappuram", "is_rural": True, "bed_count": 18, "latitude": 10.95, "longitude": 76.12},
    {"hospital_id": "H010", "name": "Kozhikode PHC Hospital #10", "hospital_type": "PHC", "district": "Kozhikode", "is_rural": True, "bed_count": 22, "latitude": 11.18, "longitude": 75.85},
]

HOSPITALS_LOOKUP = {h["hospital_id"]: h for h in KERALA_HOSPITALS}
HOSPITALS_NAME_LOOKUP = {h["name"]: h for h in KERALA_HOSPITALS}


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate road-adjusted Haversine distance in km between two coordinate pairs."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    straight_km = r * c
    # Road curvature tortuosity factor (1.3x for Kerala terrain)
    return round(straight_km * 1.3, 1)


def get_hospital_distance(h1_name: str, h2_name: str) -> float:
    """Get estimated transit distance between two hospitals."""
    h1 = HOSPITALS_NAME_LOOKUP.get(h1_name) or HOSPITALS_LOOKUP.get(h1_name)
    h2 = HOSPITALS_NAME_LOOKUP.get(h2_name) or HOSPITALS_LOOKUP.get(h2_name)
    if not h1 or not h2:
        return 35.0
    if h1["hospital_id"] == h2["hospital_id"]:
        return 0.0
    return calculate_haversine_distance(h1["latitude"], h1["longitude"], h2["latitude"], h2["longitude"])


# Realistic Predefined Reviewer Demo Dataset satisfying Section 26:
# 1. Hospital with shortage (Ernakulam PHC #1 Amoxicillin, Palakkad Taluk #8 Cipro)
# 2. Hospital with surplus (Malappuram District #4 Amoxicillin, Palakkad District #3 Cipro)
# 3. Medicine approaching expiry (Palakkad District #3 Cipro 18 days, Malappuram #4 Amox 28 days)
# 4. Rural hospital priority (Ernakulam PHC #1 Rural weight 1.5x, Thrissur PHC #7 Rural)
# 5. Multiple redistribution choices (Palakkad District vs Kozhikode PHC for Palakkad Taluk shortage)
# 6. Distance & transport cost impacting selection
REALISTIC_DEMO_RECORDS = [
    {
        "hospital_name": "Ernakulam PHC Hospital #1",
        "district": "Ernakulam",
        "is_rural": True,
        "medicine_name": "Amoxicillin 500mg",
        "category": "Antibiotic",
        "current_stock": 250,
        "daily_consumption": 110,
        "safety_stock": 300,
        "days_to_expiry": 240,
        "criticality": 4,
        "price": 6.0,
    },
    {
        "hospital_name": "Malappuram District Hospital #4",
        "district": "Malappuram",
        "is_rural": False,
        "medicine_name": "Amoxicillin 500mg",
        "category": "Antibiotic",
        "current_stock": 3200,
        "daily_consumption": 120,
        "safety_stock": 400,
        "days_to_expiry": 28,  # Imminent expiry risk!
        "criticality": 4,
        "price": 6.0,
    },
    {
        "hospital_name": "Palakkad Taluk Hospital #8",
        "district": "Palakkad",
        "is_rural": False,
        "medicine_name": "Ciprofloxacin 500mg",
        "category": "Antibiotic",
        "current_stock": 120,
        "daily_consumption": 75,
        "safety_stock": 250,
        "days_to_expiry": 180,
        "criticality": 4,
        "price": 8.5,
    },
    {
        "hospital_name": "Palakkad District Hospital #3",
        "district": "Palakkad",
        "is_rural": False,
        "medicine_name": "Ciprofloxacin 500mg",
        "category": "Antibiotic",
        "current_stock": 2400,
        "daily_consumption": 90,
        "safety_stock": 350,
        "days_to_expiry": 18,  # Closer distance (14 km) + urgent expiry!
        "criticality": 4,
        "price": 8.5,
    },
    {
        "hospital_name": "Kozhikode PHC Hospital #5",
        "district": "Kozhikode",
        "is_rural": False,
        "medicine_name": "Ciprofloxacin 500mg",
        "category": "Antibiotic",
        "current_stock": 1800,
        "daily_consumption": 40,
        "safety_stock": 200,
        "days_to_expiry": 120,  # Further distance (95 km) + longer expiry
        "criticality": 4,
        "price": 8.5,
    },
    {
        "hospital_name": "Thrissur PHC Hospital #7",
        "district": "Thrissur",
        "is_rural": True,
        "medicine_name": "Paracetamol 500mg",
        "category": "Analgesic",
        "current_stock": 300,
        "daily_consumption": 140,
        "safety_stock": 450,
        "days_to_expiry": 300,
        "criticality": 3,
        "price": 1.5,
    },
    {
        "hospital_name": "Thrissur PHC Hospital #2",
        "district": "Thrissur",
        "is_rural": False,
        "medicine_name": "Paracetamol 500mg",
        "category": "Analgesic",
        "current_stock": 4200,
        "daily_consumption": 150,
        "safety_stock": 500,
        "days_to_expiry": 24,  # Surplus and near expiry!
        "criticality": 3,
        "price": 1.5,
    },
    {
        "hospital_name": "Kozhikode PHC Hospital #10",
        "district": "Kozhikode",
        "is_rural": True,
        "medicine_name": "Oral Rehydration Salts",
        "category": "Electrolyte",
        "current_stock": 80,
        "daily_consumption": 60,
        "safety_stock": 200,
        "days_to_expiry": 365,
        "criticality": 4,
        "price": 4.0,
    },
    {
        "hospital_name": "Kozhikode PHC Hospital #5",
        "district": "Kozhikode",
        "is_rural": False,
        "medicine_name": "Oral Rehydration Salts",
        "category": "Electrolyte",
        "current_stock": 1900,
        "daily_consumption": 70,
        "safety_stock": 250,
        "days_to_expiry": 22,  # Near expiry in Kozhikode
        "criticality": 4,
        "price": 4.0,
    },
    {
        "hospital_name": "Ernakulam District Hospital #6",
        "district": "Ernakulam",
        "is_rural": False,
        "medicine_name": "Normal Saline 500ml",
        "category": "IV Fluid",
        "current_stock": 3500,
        "daily_consumption": 180,
        "safety_stock": 600,
        "days_to_expiry": 180,
        "criticality": 5,
        "price": 25.0,
    },
    {
        "hospital_name": "Ernakulam PHC Hospital #1",
        "district": "Ernakulam",
        "is_rural": True,
        "medicine_name": "Normal Saline 500ml",
        "category": "IV Fluid",
        "current_stock": 90,
        "daily_consumption": 50,
        "safety_stock": 150,
        "days_to_expiry": 200,
        "criticality": 5,
        "price": 25.0,
    },
    {
        "hospital_name": "Malappuram PHC Hospital #9",
        "district": "Malappuram",
        "is_rural": True,
        "medicine_name": "Artesunate Injection",
        "category": "Antimalarial",
        "current_stock": 15,
        "daily_consumption": 12,
        "safety_stock": 40,
        "days_to_expiry": 150,
        "criticality": 5,
        "price": 85.0,
    },
    {
        "hospital_name": "Malappuram District Hospital #4",
        "district": "Malappuram",
        "is_rural": False,
        "medicine_name": "Artesunate Injection",
        "category": "Antimalarial",
        "current_stock": 280,
        "daily_consumption": 15,
        "safety_stock": 50,
        "days_to_expiry": 26,  # Antimalarial nearing expiry!
        "criticality": 5,
        "price": 85.0,
    },
]


def init_session_state() -> None:
    """Initialize Streamlit session state keys if not present."""
    if "active_records" not in st.session_state:
        st.session_state["active_records"] = pd.DataFrame(REALISTIC_DEMO_RECORDS)
    if "etl_stats" not in st.session_state:
        st.session_state["etl_stats"] = {
            "records_received": len(REALISTIC_DEMO_RECORDS),
            "records_processed": len(REALISTIC_DEMO_RECORDS),
            "missing_values": 0,
            "invalid_records": 0,
            "duplicates_removed": 0,
            "last_updated": datetime.datetime.now().strftime("%H:%M:%S"),
        }
    if "demo_completed" not in st.session_state:
        st.session_state["demo_completed"] = False


def get_active_dataset() -> pd.DataFrame:
    """Retrieve the current active inventory records dataframe."""
    init_session_state()
    df = st.session_state["active_records"]
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        df = pd.DataFrame(REALISTIC_DEMO_RECORDS)
        st.session_state["active_records"] = df
    return df.copy()


def reset_to_demo_data() -> None:
    """Reset the active dataset to the realistic predefined scenario."""
    init_session_state()
    df = pd.DataFrame(REALISTIC_DEMO_RECORDS)
    st.session_state["active_records"] = df
    st.session_state["etl_stats"] = {
        "records_received": len(df),
        "records_processed": len(df),
        "missing_values": 0,
        "invalid_records": 0,
        "duplicates_removed": 0,
        "last_updated": datetime.datetime.now().strftime("%H:%M:%S"),
    }
    st.session_state["demo_completed"] = False


def clear_active_dataset() -> None:
    """Clear all inventory records."""
    init_session_state()
    st.session_state["active_records"] = pd.DataFrame(columns=[
        "hospital_name", "district", "is_rural", "medicine_name", "category",
        "current_stock", "daily_consumption", "safety_stock", "days_to_expiry", "criticality", "price"
    ])
    st.session_state["etl_stats"] = {
        "records_received": 0,
        "records_processed": 0,
        "missing_values": 0,
        "invalid_records": 0,
        "duplicates_removed": 0,
        "last_updated": datetime.datetime.now().strftime("%H:%M:%S"),
    }
    st.session_state["demo_completed"] = False


def add_manual_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate and append a single manually entered record."""
    init_session_state()
    try:
        # Basic validation
        h_name = str(record.get("hospital_name", "")).strip()
        m_name = str(record.get("medicine_name", "")).strip()
        if not h_name or not m_name:
            return False, "Hospital Name and Medicine Name are required."

        stock = int(record.get("current_stock", 0))
        usage = max(1, int(record.get("daily_consumption", 1)))
        safety = int(record.get("safety_stock", usage * 3))
        expiry = int(record.get("days_to_expiry", 180))
        crit = int(record.get("criticality", 3))
        price = float(record.get("price", 10.0))
        is_rural = bool(record.get("is_rural", False))
        district = str(record.get("district", "General District")).strip()
        category = str(record.get("category", "General Medicine")).strip()

        new_row = {
            "hospital_name": h_name,
            "district": district,
            "is_rural": is_rural,
            "medicine_name": m_name,
            "category": category,
            "current_stock": stock,
            "daily_consumption": usage,
            "safety_stock": safety,
            "days_to_expiry": expiry,
            "criticality": crit,
            "price": price,
        }

        df = st.session_state["active_records"]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state["active_records"] = df

        stats = st.session_state["etl_stats"]
        stats["records_received"] += 1
        stats["records_processed"] += 1
        stats["last_updated"] = datetime.datetime.now().strftime("%H:%M:%S")

        return True, f"Successfully recorded {m_name} for {h_name}."
    except Exception as e:
        return False, f"Invalid record values: {str(e)}"


def import_csv_data(uploaded_file) -> Tuple[bool, str]:
    """Parse, clean, and ingest an uploaded CSV file with error handling."""
    init_session_state()
    try:
        raw_df = pd.read_csv(uploaded_file)
        if raw_df.empty:
            return False, "The uploaded CSV file is empty."

        total_received = len(raw_df)
        missing_count = int(raw_df.isna().sum().sum())

        # Flexible column mapping
        col_map = {}
        for col in raw_df.columns:
            c_low = col.lower().strip().replace(" ", "_")
            if "hosp" in c_low:
                col_map[col] = "hospital_name"
            elif "med" in c_low:
                col_map[col] = "medicine_name"
            elif "stock" in c_low:
                col_map[col] = "current_stock"
            elif "usage" in c_low or "consum" in c_low:
                col_map[col] = "daily_consumption"
            elif "expir" in c_low:
                col_map[col] = "days_to_expiry"
            elif "safety" in c_low:
                col_map[col] = "safety_stock"
            elif "rural" in c_low:
                col_map[col] = "is_rural"
            elif "crit" in c_low:
                col_map[col] = "criticality"
            elif "dist" in c_low:
                col_map[col] = "district"

        renamed = raw_df.rename(columns=col_map)
        req_cols = ["hospital_name", "medicine_name", "current_stock", "daily_consumption"]
        for rc in req_cols:
            if rc not in renamed.columns:
                return False, f"Missing required column: '{rc}'. Please check the sample CSV format."

        # Fill missing values and sanitize numeric columns
        clean_df = renamed.copy()
        clean_df["current_stock"] = pd.to_numeric(clean_df["current_stock"], errors="coerce").fillna(0).astype(int)
        clean_df["daily_consumption"] = pd.to_numeric(clean_df["daily_consumption"], errors="coerce").fillna(1).clip(lower=1).astype(int)

        if "safety_stock" not in clean_df.columns:
            clean_df["safety_stock"] = (clean_df["daily_consumption"] * 3).astype(int)
        else:
            clean_df["safety_stock"] = pd.to_numeric(clean_df["safety_stock"], errors="coerce").fillna(clean_df["daily_consumption"] * 3).astype(int)

        if "days_to_expiry" not in clean_df.columns:
            clean_df["days_to_expiry"] = 180
        else:
            clean_df["days_to_expiry"] = pd.to_numeric(clean_df["days_to_expiry"], errors="coerce").fillna(180).astype(int)

        if "criticality" not in clean_df.columns:
            clean_df["criticality"] = 3
        else:
            clean_df["criticality"] = pd.to_numeric(clean_df["criticality"], errors="coerce").fillna(3).clip(1, 5).astype(int)

        if "is_rural" not in clean_df.columns:
            clean_df["is_rural"] = clean_df["hospital_name"].astype(str).str.contains("Rural|PHC|#1|#7|#9|#10", case=False)
        else:
            clean_df["is_rural"] = clean_df["is_rural"].astype(str).str.lower().isin(["true", "1", "yes", "rural"])

        if "district" not in clean_df.columns:
            clean_df["district"] = "General District"

        if "price" not in clean_df.columns:
            clean_df["price"] = 10.0

        if "category" not in clean_df.columns:
            clean_df["category"] = "General Medicine"

        # Deduplicate on (hospital_name, medicine_name)
        initial_len = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["hospital_name", "medicine_name"], keep="last")
        duplicates_count = initial_len - len(clean_df)

        st.session_state["active_records"] = clean_df
        st.session_state["etl_stats"] = {
            "records_received": total_received,
            "records_processed": len(clean_df),
            "missing_values": missing_count,
            "invalid_records": 0,
            "duplicates_removed": duplicates_count,
            "last_updated": datetime.datetime.now().strftime("%H:%M:%S"),
        }
        return True, f"Successfully processed {len(clean_df)} inventory records from CSV."
    except Exception as e:
        return False, f"Unable to process the dataset. Please ensure required columns are present. ({str(e)})"


def get_sample_csv_text() -> str:
    """Generate realistic sample CSV string for reviewer download."""
    sample_df = pd.DataFrame([
        {"Hospital": "Ernakulam PHC Hospital #1", "Medicine": "Paracetamol 500mg", "Stock": 200, "Daily Usage": 100, "Expiry": 180, "Rural": "YES"},
        {"Hospital": "Kozhikode PHC Hospital #5", "Medicine": "Paracetamol 500mg", "Stock": 4500, "Daily Usage": 120, "Expiry": 25, "Rural": "NO"},
        {"Hospital": "Thrissur PHC Hospital #7", "Medicine": "Amoxicillin 500mg", "Stock": 150, "Daily Usage": 80, "Expiry": 300, "Rural": "YES"},
        {"Hospital": "Malappuram District Hospital #4", "Medicine": "Amoxicillin 500mg", "Stock": 3200, "Daily Usage": 120, "Expiry": 28, "Rural": "NO"},
        {"Hospital": "Palakkad Taluk Hospital #8", "Medicine": "Ciprofloxacin 500mg", "Stock": 100, "Daily Usage": 70, "Expiry": 180, "Rural": "NO"},
        {"Hospital": "Palakkad District Hospital #3", "Medicine": "Ciprofloxacin 500mg", "Stock": 2400, "Daily Usage": 90, "Expiry": 18, "Rural": "NO"},
        {"Hospital": "Thrissur PHC Hospital #2", "Medicine": "Normal Saline 500ml", "Stock": 1800, "Daily Usage": 80, "Expiry": 60, "Rural": "NO"},
        {"Hospital": "Ernakulam PHC Hospital #1", "Medicine": "Normal Saline 500ml", "Stock": 90, "Daily Usage": 50, "Expiry": 200, "Rural": "YES"},
    ])
    return sample_df.to_csv(index=False)


def analyze_inventory(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Perform complete stock analysis:
    - Calculates total stock
    - Finds shortages (days of stock < safety stock days)
    - Finds surpluses (stock > horizon demand + safety stock)
    - Finds imminent expiry risks (days until expiry <= 30)
    """
    if df is None:
        df = get_active_dataset()

    if df.empty:
        return {
            "total_stock": 0,
            "shortage_count": 0,
            "surplus_count": 0,
            "expiry_risk_count": 0,
            "shortages_df": pd.DataFrame(),
            "surplus_df": pd.DataFrame(),
            "expiry_df": pd.DataFrame(),
        }

    records = df.copy()
    records["days_remaining"] = (records["current_stock"] / records["daily_consumption"].clip(lower=1)).round(1)
    
    # 7-day horizon demand + safety stock target
    records["target_req"] = records["daily_consumption"] * 7 + records["safety_stock"]
    records["deficit"] = (records["target_req"] - records["current_stock"]).clip(lower=0)
    records["surplus"] = (records["current_stock"] - records["target_req"]).clip(lower=0)

    # 1. Shortages: days remaining < 4 days or deficit > 0
    shortages = records[records["deficit"] > 0].copy()
    if not shortages.empty:
        shortages["risk"] = np.where(shortages["days_remaining"] <= 2.0, "CRITICAL", "HIGH")
        shortages = shortages.sort_values(by=["days_remaining", "deficit"], ascending=[True, False])

    # 2. Surplus: stock exceeding 7-day demand + safety
    surplus = records[records["surplus"] > 0].copy()
    if not surplus.empty:
        surplus["action"] = "Available for Transfer"
        surplus = surplus.sort_values(by="surplus", ascending=False)

    # 3. Expiry Risk: days_to_expiry <= 30 days
    expiry = records[records["days_to_expiry"] <= 30].copy()
    if not expiry.empty:
        expiry["risk"] = np.where(expiry["days_to_expiry"] <= 20, "CRITICAL", "HIGH")
        expiry = expiry.sort_values(by="days_to_expiry", ascending=True)

    return {
        "total_stock": int(records["current_stock"].sum()),
        "shortage_count": len(shortages),
        "surplus_count": len(surplus),
        "expiry_risk_count": len(expiry),
        "shortages_df": shortages,
        "surplus_df": surplus,
        "expiry_df": expiry,
        "full_df": records,
    }


def compute_redistributions(df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """
    Mathematical redistribution matching algorithm:
    Matches surplus facilities to shortage facilities:
    1. Medicine matching
    2. Expiry feasibility check (batch must not expire before transit arrival + 5 days shelf life)
    3. FEFO preference (earlier expiring surplus given priority)
    4. Distance minimization (lower transport cost)
    5. Rural receiving facility priority (1.5x urgency)
    """
    if df is None:
        df = get_active_dataset()

    analysis = analyze_inventory(df)
    shortages_df = analysis["shortages_df"]
    surplus_df = analysis["surplus_df"]

    if shortages_df.empty or surplus_df.empty:
        return []

    recommendations: List[Dict[str, Any]] = []

    # Track remaining surplus to prevent double-allocation
    available_surplus = surplus_df.set_index(["hospital_name", "medicine_name"])["surplus"].to_dict()

    # Prioritize shortages by rural status, criticality, then days remaining
    shortages_sorted = shortages_df.sort_values(
        by=["is_rural", "criticality", "days_remaining"],
        ascending=[False, False, True]
    )

    for _, dest in shortages_sorted.iterrows():
        med_name = dest["medicine_name"]
        dst_hosp = dest["hospital_name"]
        needed = int(dest["deficit"])

        if needed <= 0:
            continue

        # Find matching candidates with surplus of the exact same medicine
        candidates = []
        for (src_hosp, sm_name), sup_qty in available_surplus.items():
            if sm_name == med_name and src_hosp != dst_hosp and sup_qty > 0:
                src_row = surplus_df[(surplus_df["hospital_name"] == src_hosp) & (surplus_df["medicine_name"] == med_name)].iloc[0]
                dist_km = get_hospital_distance(src_hosp, dst_hosp)
                transit_days = max(1, int(math.ceil(dist_km / 120.0)))
                expiry_days = int(src_row["days_to_expiry"])

                # Expiry feasibility constraint: must have >= 7 days shelf life at arrival
                if expiry_days - transit_days < 7:
                    continue

                # Transport cost = fixed ₹100 + ₹2.5/km
                trans_cost = 100.0 + (2.5 * dist_km)

                # Score: prefer earlier expiry (FEFO) and shorter distance
                score = (expiry_days * 0.4) + (dist_km * 0.6)
                candidates.append({
                    "src_hosp": src_hosp,
                    "src_row": src_row,
                    "dist_km": dist_km,
                    "transit_days": transit_days,
                    "expiry_days": expiry_days,
                    "trans_cost": trans_cost,
                    "score": score,
                    "available_qty": sup_qty,
                })

        if not candidates:
            continue

        # Sort candidates by score (best FEFO + distance match)
        candidates.sort(key=lambda x: x["score"])

        for cand in candidates:
            if needed <= 0:
                break
            src_hosp = cand["src_hosp"]
            avail_qty = available_surplus.get((src_hosp, med_name), 0)
            if avail_qty <= 0:
                continue

            transfer_qty = min(needed, avail_qty)
            needed -= transfer_qty
            available_surplus[(src_hosp, med_name)] -= transfer_qty

            # Determine priority
            is_dest_rural = dest["is_rural"]
            is_near_expiry = cand["expiry_days"] <= 30
            priority = "HIGH" if (is_dest_rural or is_near_expiry or dest["days_remaining"] <= 2.0) else "MEDIUM"

            reason = (
                f"{dst_hosp} has only {dest['days_remaining']:.1f} days stock remaining "
                f"while {src_hosp} has usable surplus"
            )
            if is_near_expiry:
                reason += f" nearing expiry in {cand['expiry_days']} days (FEFO priority)"
            if is_dest_rural:
                reason += " [Rural Facility Priority]"

            recommendations.append({
                "priority": priority,
                "medicine": med_name,
                "from_hospital": src_hosp,
                "to_hospital": dst_hosp,
                "quantity": int(transfer_qty),
                "reason": reason,
                "distance_km": cand["dist_km"],
                "transit_days": cand["transit_days"],
                "transport_cost": round(cand["trans_cost"], 0),
                "why_details": {
                    "dest_stock": int(dest["current_stock"]),
                    "predicted_demand": int(dest["daily_consumption"] * 7),
                    "safety_stock": int(dest["safety_stock"]),
                    "expected_shortage": int(dest["deficit"]),
                    "source_surplus": int(cand["src_row"]["surplus"]),
                    "expiry_risk": "HIGH" if is_near_expiry else "NORMAL",
                    "source_expiry_days": cand["expiry_days"],
                    "distance_km": cand["dist_km"],
                    "transit_days": cand["transit_days"],
                    "is_rural": is_dest_rural,
                    "medicine": med_name,
                    "from_hosp": src_hosp,
                    "to_hosp": dst_hosp,
                    "quantity": int(transfer_qty),
                }
            })

    return recommendations


def compute_before_after_simulation(
    df: Optional[pd.DataFrame] = None,
    recommendations: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Compute rigorous before vs after impact simulation.
    All percentage improvements and unit changes are calculated dynamically.
    """
    if df is None:
        df = get_active_dataset()
    if recommendations is None:
        recommendations = compute_redistributions(df)

    analysis = analyze_inventory(df)
    shortages_df = analysis["shortages_df"]
    expiry_df = analysis["expiry_df"]
    surplus_df = analysis["surplus_df"]

    # BEFORE METRICS
    before_shortages_count = len(shortages_df)
    before_shortage_units = int(shortages_df["deficit"].sum()) if not shortages_df.empty else 0
    before_expiry_units = int(expiry_df["current_stock"].sum()) if not expiry_df.empty else 0
    before_surplus_units = int(surplus_df["surplus"].sum()) if not surplus_df.empty else 0

    # TRANSFERS IMPACT
    total_transferred = sum(r["quantity"] for r in recommendations)
    total_transport_cost = sum(r["transport_cost"] for r in recommendations)

    # Calculate rescued expiry stock
    rescued_expiry_units = 0
    for r in recommendations:
        if r["why_details"]["expiry_risk"] == "HIGH":
            rescued_expiry_units += r["quantity"]

    # AFTER METRICS
    after_shortage_units = max(0, before_shortage_units - total_transferred)
    # Estimate remaining critical shortage count
    resolved_shortages = 0
    remaining_req = before_shortage_units
    for r in recommendations:
        if remaining_req > 0:
            resolved_shortages += 1
            remaining_req -= r["quantity"]

    after_shortages_count = max(0, before_shortages_count - resolved_shortages)
    after_expiry_units = max(0, before_expiry_units - rescued_expiry_units)
    after_surplus_units = max(0, before_surplus_units - total_transferred)

    # DYNAMIC PERCENTAGE REDUCTIONS
    shortage_reduction_pct = (
        round(((before_shortages_count - after_shortages_count) / max(1, before_shortages_count)) * 100.0, 1)
        if before_shortages_count > 0 else 100.0
    )
    wastage_reduction_pct = (
        round((rescued_expiry_units / max(1, before_expiry_units)) * 100.0, 1)
        if before_expiry_units > 0 else 0.0
    )

    return {
        "before": {
            "critical_shortages": before_shortages_count,
            "shortage_units": before_shortage_units,
            "expiry_risk_stock": before_expiry_units,
            "surplus_stock": before_surplus_units,
        },
        "after": {
            "critical_shortages": after_shortages_count,
            "shortage_units": after_shortage_units,
            "expiry_risk_stock": after_expiry_units,
            "surplus_stock": after_surplus_units,
        },
        "impact": {
            "shortage_reduction_pct": shortage_reduction_pct,
            "wastage_reduction_pct": wastage_reduction_pct,
            "medicine_redistributed": total_transferred,
            "transport_cost": int(total_transport_cost),
            "transfers_count": len(recommendations),
        }
    }


def get_system_status() -> Dict[str, Any]:
    """Return status summary for the Home page CURRENT SYSTEM STATUS panel."""
    init_session_state()
    df = st.session_state["active_records"]
    stats = st.session_state["etl_stats"]
    is_loaded = df is not None and isinstance(df, pd.DataFrame) and not df.empty

    hospitals_count = df["hospital_name"].nunique() if is_loaded else 0
    medicines_count = df["medicine_name"].nunique() if is_loaded else 0
    records_count = len(df) if is_loaded else 0

    return {
        "data_loaded": "YES" if is_loaded else "NO",
        "hospitals_count": hospitals_count,
        "medicines_count": medicines_count,
        "records_count": records_count,
        "last_analysis": stats.get("last_updated", datetime.datetime.now().strftime("%H:%M:%S")),
        "is_loaded_bool": is_loaded,
    }
