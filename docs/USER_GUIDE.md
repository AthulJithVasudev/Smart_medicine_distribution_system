# User Guide: Smart Medicine Redistribution System

Welcome to the **Smart Medicine Redistribution System** operational dashboard. This guide explains how to navigate, interpret, and act upon the information presented across the platform.

---

## 1. Quick Navigation & Shared Controls

All pages share a unified sidebar filter block located on the left side of the screen:
- **Scenario:** Select from 7 evaluation scenarios (e.g. *Normal Operations*, *Monsoon Surge*, *Severe Dengue Outbreak*).
- **Policy:** Switch between **Smart Plan** (MILP optimization), **Nearest Surplus** (Greedy heuristic), and **Do Nothing** (Status quo / No transfer).
- **Simulation Day:** Slide across the 90-day simulation timeline. When switching scenarios, the system automatically focuses on the day with highest network shortages.
- **District / Facility Tier:** Filter by administrative district or hospital level (District Hospital, CHC, PHC).
- **URL Synchronization:** Your filter settings are automatically reflected in your browser URL (`?scenario=...&policy=...&day=...`) so you can bookmark or share any view.

### Guided Tour
If you are presenting or exploring for the first time, expand the **"Quick Guided Tour"** in the sidebar. Click any of the 5 numbered steps to immediately configure the dashboard for that scenario:
1. *See normal operations*
2. *Trigger a monsoon surge*
3. *Compare Smart Plan with Do Nothing*
4. *Check forecast accuracy in Replay*
5. *Review Model Evaluation verdicts*

---

## 2. Page-by-Page Overview

### 2.1 Home Page (`app/Home.py`)
Designed for **Government Health Officers** and executive decision-makers with an **Action-First** layout:
- **Role Selector:** Switch between *Government Health Officer*, *Hospital Administrator*, and *Researcher / Data Scientist*.
- **Block 1: What Needs Attention Now**
  - High-visibility metric cards: Network stockout rate, days of inventory, critical shortage count, and expiring batch count.
  - Urgent alerts: Top 5 hospitals facing imminent stockouts and top 5 batches expiring within 15 days.
- **Block 2: What Should Be Done**
  - Recommended inter-hospital transfer manifests showing source facility, destination facility, batch ID, units, and vehicle cost.
  - Plain-language justification for each transfer (e.g., *"Shortage predicted in 3 days; batch expires in 18 days; source maintains safety stock buffer"*).
  - One-click **"Download Transfer Orders (CSV)"** button for dispatch logistics.
- **Block 3: What It Achieves**
  - Quantitative impact vs. Status Quo: Units of wastage prevented, financial savings in INR, stockouts avoided, and fill rate comparison.
  - Interactive OpenStreetMap showing active transfer routes and hospital inventory status.

### 2.2 Hospital Administrator (`app/pages/1_Hospital_Administrator.py`)
Designed for individual facility superintendents and chief pharmacists:
- **Facility Selector:** Select any hospital (e.g. *H01 - Central District Hospital*).
- **Local Inventory Health:** Real-time stock levels, current days of supply, and replenishment status for every drug.
- **Demand Forecast with Uncertainty Band:** Visualizes historical consumption alongside predicted patient demand with 95% confidence intervals.
- **Local Dispatch & Receipt Manifests:** Table of incoming deliveries (with estimated transit arrival) and outgoing shipments.
- **Download Facility Orders:** Export facility-specific transfer orders for warehouse staff.

### 2.3 Scenario Comparison (`app/pages/2_Scenario_Comparison.py`)
Evaluates system resilience across baseline and stress conditions:
- **Categorized Tabs:**
  - *Standard Conditions:* Normal operations, Monsoon seasonal surge, Supply delay disruptions.
  - *Novel / Stress Situations:* Dengue 4x epidemic, Double demand shock, Extended supply disruption.
- **Automated Takeaway:** Plain-English summary computed dynamically from the run data (e.g. *"Smart Plan reduced stockout days by 85.0% compared with Do Nothing"*).
- **Multi-Seed Variance:** Displays mean performance and error bars across distinct random seeds to prove algorithmic stability.

### 2.4 Replay & Historical Accuracy (`app/pages/3_Replay.py`)
Audits model trustworthiness on historical dates:
- Select any past date to inspect what the machine learning model forecasted versus what patient demand actually occurred.
- Visualizes point forecasts, actual OPD dispenses, and 95% prediction intervals.
- Reports Mean Absolute Error (MAE) and forecast bias.

### 2.5 Model Evaluation (`app/pages/4_Model_Evaluation.py`)
Presents transparent, auditable validation for researchers and examiners:
- **Verdict Cards:**
  - Forecast error reduction of ML over standard 7-day moving average.
  - Stockout risk classification recall on critical deficit events.
  - Unsupervised OPD volume anomaly detection rate.
- **Feature Importance & Ablation:** Proves the incremental value of integrating outpatient OPD lead indicators into the demand forecaster.
- **Theoretical Oracle Bound:** Explains the mathematical upper limit achievable with perfect future information.

### 2.6 About & Methodology (`app/pages/5_About.py`)
- Complete **2-Minute Executive Overview**.
- **Interactive Glossary:** Searchable terminology guide defining terms like FEFO, MILP, WAPE, Safety Stock, and Fill Rate.
- **Data Provenance:** Clarifies the simulated nature of the data and outlines real-world integration pathways for HMIS/e-Aushadhi systems.

---

## 3. Status Badges & Color Reference

Status indicators always combine **clear text with color** to ensure accessibility:

| Status Badge | Color | Operational Meaning |
| :--- | :---: | :--- |
| **Critical Shortage** | 🔴 Red | Current inventory is 0 or less than 1 day of projected demand. Imminent stockout. |
| **Low Stock** | 🟠 Amber | Stock is between 1 and 3 days of supply (below dynamic safety stock threshold). |
| **Normal** | 🟢 Green | Inventory is well-balanced (between 3 and 14 days of projected demand). |
| **Surplus** | 🔵 Blue | Inventory exceeds 14 days of demand; eligible for redistribution to facilities in deficit. |
| **Expiring Soon** | 🔴 Red / 🟠 Amber | Batch expires within 30 days and exceeds projected local consumption. Urgent transfer recommended. |
| **Active Transfer** | 🟢 Green | Transfer manifest dispatched and in-transit within viable lead-time. |
