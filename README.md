# Smart Medicine Redistribution System for Government Hospitals

> **System Architecture v4: Cloud-Hosted, Scenario-Driven Multi-Facility Inventory Optimization**  
> *Academic Data Mining, Data Analysis, and Big Data Research Implementation*

---

## ⚠️ Academic & Synthetic Data Disclaimer

All hospital facilities, patient demand records, inventory levels, Out-Patient Department (OPD) visit counts, and batch identifiers in this system are **SYNTHETIC DATA** generated reproducibly from fixed random seeds by the daily time-stepped simulation engine. These values are produced strictly for academic simulation, mathematical optimization benchmarks, and research evaluation. They do **not** represent real, official government hospital statistics. Data provenance is tracked across all records (`SIMULATED`, `DERIVED`, `OFFICIAL`).

---

## 1. Project Overview & Business Problem

Public healthcare networks across geographical regions (Primary Health Centres, Taluk Hospitals, District Medical Colleges) frequently suffer from severe inventory misalignments:

- **Facility A** holds excess stock of essential medicines (e.g. antibiotics, antipyretics, IV fluids) approaching expiration dates, incurring substantial financial write-offs.
- **Facility B** (frequently a peripheral or rural health clinic) experiences unanticipated demand surges (e.g. during monsoon seasons or dengue epidemics) and faces imminent stockouts.

The **Smart Medicine Redistribution System** solves this operational challenge by operating a closed-loop data intelligence pipeline:
$$\text{Observe} \longrightarrow \text{ETL} \longrightarrow \text{Forecast} \longrightarrow \text{Optimize} \longrightarrow \text{Apply Transfers} \longrightarrow \text{Next Day}$$
The system decides **whether, when, which batch, how much, and from where to transfer** medicines to eliminate stockouts and minimize wastage while respecting transit times and shelf-life feasibility.

---

## 2. Logical Architecture Diagram

```
                        ┌─────────────────────────────────────────┐
                        │       STATEFUL DAILY TIME-STEPPED       │
                        │             SIMULATOR                   │
                        └────────────────────┬────────────────────┘
                                             │ Raw Observations (Censored & Noisy)
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │          RAW PARQUET TABLES             │
                        │       (Partitioned by report_day)       │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │            PYSPARK ETL                  │
                        │  (Deduplication, Window Lags, Rolling)  │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │          CURATED PARQUET TABLES         │
                        │    (Strict Historical Information at t) │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │          DUCKDB SQL ANALYTICS           │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │          ML & FORECASTING               │
                        │  - SARIMAX Dedicated Models (High-Vol)  │
                        │  - Global Gradient Boosting (Long-Tail) │
                        │  - Random Forest Risk Classifier        │
                        │  - Unsupervised Isolation Forest        │
                        │  - Hospital K-Means Clustering          │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │     EVALUATION & MLFLOW TRACKING        │
                        │  - Walk-Forward Cross Validation        │
                        │  - Champion / Challenger Registry       │
                        │  - Oracle Ground-Truth Bounds           │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │      OR-TOOLS ILP REDISTRIBUTION        │
                        │  - Mandatory Receiving Cap              │
                        │  - Batch Stock Limits                   │
                        │  - Expiry & Timing Feasibility          │
                        │  - Soft FEFO Tie-Breaker ($\epsilon$)   │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │        PUBLISHED SERVING PARQUET        │
                        │       (12 Compact Serving Tables)       │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │      STREAMLIT EXECUTIVE DASHBOARD      │
                        │    (Zero-Compute Serving / UI Layer)    │
                        └─────────────────────────────────────────┘
```

---

## 3. Technology Stack

- **Language & Runtime**: Python 3.12 / 3.14
- **Distributed ETL**: PySpark local-mode (`spark.sql.shuffle.partitions` = 8)
- **Analytical Storage**: Apache Parquet (Snappy-compressed), DuckDB
- **Forecasting & ML**: statsmodels (SARIMAX), Scikit-Learn (HistGradientBoosting, RandomForest, KMeans, IsolationForest)
- **Mathematical Optimization**: Google OR-Tools (`pywraplp` Mixed Integer Linear Programming)
- **Experiment Tracking**: Local MLflow with SQLite backend
- **Serving & UI**: Streamlit, Plotly Express & Graph Objects, Mapbox Geospatial
- **Workflow Automation**: GitHub Actions CI/CD pipelines (Ubuntu runners, Java 17, Python 3.12)

---

## 4. Repository Structure

```
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Automated testing on PR and push
│       └── pipeline.yml           # Scheduled weekly pipeline execution
├── app/
│   ├── Home.py                    # State Health Department Command Center
│   ├── data_loader.py             # Cached Parquet serving reader
│   └── pages/
│       ├── 1_Hospital_Administrator.py   # Facility inventory & transfer orders
│       ├── 2_Scenario_Comparison.py      # Cross-policy benchmark & stress testing
│       ├── 3_Replay.py                   # Forecast verification vs realized demand
│       ├── 4_Model_Evaluation.py         # Metrics, baselines, and ablation suite
│       └── 5_About.py                    # Architecture, methodology, disclaimer
├── configs/
│   ├── dev.yaml                   # 10 hospitals, 30 medicines, 14-day horizon
│   ├── demo.yaml                  # 30 hospitals, 60 medicines, 30-day horizon
│   └── full.yaml                  # 50 hospitals, 200 medicines, 3-year history
├── pipeline/
│   ├── simulator/                 # Stateful discrete-event engine & entities
│   │   ├── entities.py
│   │   ├── demand.py
│   │   ├── supply.py
│   │   ├── observation.py
│   │   └── engine.py
│   ├── etl/                       # PySpark & DuckDB ETL pipeline
│   │   ├── storage.py
│   │   ├── cleaning.py
│   │   ├── features.py
│   │   ├── spark_etl.py
│   │   └── pipeline.py
│   ├── models/                    # ML forecasting, classification, clustering
│   │   ├── forecasting.py
│   │   ├── clustering.py
│   │   ├── classification.py
│   │   ├── anomaly.py
│   │   └── leakage.py
│   ├── optimizer/                 # OR-Tools MILP optimizer
│   │   ├── solver.py
│   │   └── rolling.py
│   ├── evaluation/                # Metrics, baselines, walk-forward, MLflow
│   │   ├── metrics.py
│   │   ├── baselines.py
│   │   ├── validation.py
│   │   └── tracking.py
│   ├── utils/
│   │   └── scenario_loader.py
│   ├── prepare.py                 # Historical data generation & ETL CLI
│   ├── train.py                   # Model training & baseline evaluation CLI
│   ├── run_scenario.py            # Closed-loop scenario matrix CLI
│   └── export.py                  # Serving Parquet publisher CLI
├── scenarios/
│   ├── normal.yaml                # Baseline conditions
│   ├── monsoon_surge.yaml         # Seasonal fever & gastro surge
│   ├── dengue_2x.yaml             # 2x fever outbreak
│   ├── dengue_4x.yaml             # 4x severe epidemic (unseen distribution shift)
│   ├── supplier_delay.yaml        # +10 day supply chain disruption (unseen)
│   ├── low_stock.yaml             # 30% initial inventory constraint
│   └── rural_access.yaml          # 2x transit duration on rural roads (unseen)
├── tests/
│   ├── test_simulator.py
│   ├── test_etl.py
│   ├── test_evaluation.py
│   ├── test_models.py
│   ├── test_leakage.py
│   ├── test_optimizer.py
│   ├── test_scenarios.py
│   ├── test_pipeline_and_publishing.py
│   ├── test_streamlit_app.py
│   └── test_integration.py
├── requirements.txt
├── requirements-pipeline.txt
└── pyproject.toml
```

---

## 5. Development Profiles

| Profile | Hospitals | Medicines | History Window | Scenario Horizon | Primary Purpose |
|---|---|---|---|---|---|
| **DEV** | 10 | 30 | 30 days | 14 days | Rapid local development & test suite execution |
| **DEMO** | 30 | 60 | 180 days | 30 days | High-fidelity university demonstration |
| **FULL** | 50 | 200 | 1,095 days (3 yrs) | 60 days | Large-scale empirical research experiments |

---

## 6. Execution Commands

### 1. Installation
```powershell
pip install -r requirements-pipeline.txt
```

### 2. Run Test Suite
```powershell
pytest -ra
```

### 3. Stage 1: Data Preparation & ETL
```powershell
python -m pipeline.prepare --profile dev
```

### 4. Stage 2: Train ML Models & Log to MLflow
```powershell
python -m pipeline.train --profile dev
```

### 5. Stage 3: Closed-Loop Scenario Matrix
```powershell
python -m pipeline.run_scenario --scenario normal --policy optimizer --profile dev --seeds 3
```

To run all scenarios across all policies:
```powershell
python -m pipeline.run_scenario --scenario all --policy all --profile dev --seeds 3
```

### 6. Stage 4: Publish Serving Tables
```powershell
python -m pipeline.export --profile dev --target-dir data/serving
```

### 7. Launch Streamlit Executive Dashboard
```powershell
streamlit run app/Home.py
```

---

## 7. Mathematical Optimization Formulation

The redistribution optimizer is formulated as a Mixed-Integer Linear Program (MILP) solved using Google OR-Tools (`pywraplp` CBC/SCIP):

$$\min \sum_{i,j,m,b} c^{\text{trans}}_{ij} x_{ijmb} + \sum_{j,m} c^{\text{short}}_{jm} u_{jm} + \sum_{i,m,b} \gamma V_m w_{imb} + \sum_{i,j,m,b} \epsilon (\text{exp}_b - t) x_{ijmb}$$

Subject to:
1. **Source Transferable Surplus Limit**:
   $$\sum_{j,b} x_{ijmb} \le \text{avail}_{im} \quad \forall i, m$$
2. **Individual Batch Stock Limit**:
   $$\sum_{j} x_{ijmb} \le \text{stock}_{imb} \quad \forall i, m, b$$
3. **Shortage Demand Balance**:
   $$\sum_{i,b} x_{ijmb} + u_{jm} \ge \text{need}_{jm} \quad \forall j, m$$
4. **Mandatory Destination Receiving Cap**:
   $$\sum_{i,b} x_{ijmb} \le \text{need}_{jm} \quad \forall j, m$$
   *(inflow is capped at the forecast need, so surplus is not pushed onto receiving hospitals (forecast error can still leave some excess)).*
5. **Expiry Wastage Reduction Penalty**:
   $$w_{imb} + \sum_{j} x_{ijmb} \ge \text{at\_risk}_{imb} \quad \forall i, m, b \quad (w_{imb} \ge 0)$$
   where $\text{at\_risk}_{imb}$ is computed pre-solve via FEFO allocation of forecast demand.
6. **Shelf-Life & Transit Timing Feasibility**:
   $$x_{ijmb} = 0 \quad \text{if } \text{exp}_b - (t + \tau_{ij}) < \text{min\_shelf\_life} \quad \text{or} \quad \tau_{ij} > \text{horizon}$$
7. **Soft FEFO Tie-Breaker**:
   $$\epsilon \cdot (\text{exp}_b - t)$$ breaks ties toward earlier-expiring batches (soft FEFO) when transport costs are identical.

---

## 8. Verification & Test Suite

[![CI Status](https://github.com/AthulJithVasudev/Smart_medicine_distribution_system/actions/workflows/ci.yml/badge.svg)](https://github.com/AthulJithVasudev/Smart_medicine_distribution_system/actions)

All 71 automated tests across all test suites run cleanly:
- `test_cli_and_doctor.py`: System Doctor diagnostics, environment checks, and CLI subcommands.
- `test_simulator.py`: FEFO consumption, reproducible random streams, batch lifecycles, and demand censoring.
- `test_etl.py`: Partitioned Parquet, deduplication, missing values, unit corrections, DuckDB queries, temporal lag causality, and ETL_BACKEND fallback.
- `test_evaluation.py`: Naive baselines, walk-forward splits, Oracle evaluation, and MLflow logging.
- `test_models.py`: Dedicated SARIMAX, global tree regressors, hospital K-Means clustering, and forward-outcome Random Forest risk classification.
- `test_leakage.py`: 7-point leakage audit, label-shuffle tests, and unsupervised Isolation Forest verification.
- `test_optimizer.py`: Receiving cap adherence, source limits, expiry feasibility, wastage reduction, and FEFO tie-break selection.
- `test_scenarios.py`: Closed-loop feedback, seen vs unseen stress tests, replay data schemas, and multi-seed spread.
- `test_pipeline_and_publishing.py`: End-to-end execution of prepare, train, scenario matrix, and serving export.
- `test_streamlit_app.py`: Serving Parquet readability, zero-compute presentation rules, bundled sample fallback, and DuckDB projection.
- `test_integration.py`: End-to-end clean pipeline execution.

