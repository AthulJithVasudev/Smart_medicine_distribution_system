# Developer & Contributor Guide

This document describes the engineering structure, CLI tools, configuration profiles, testing workflows, and deployment processes for the **Smart Medicine Redistribution System**.

---

## 1. Repository Directory Structure

```text
├── .github/workflows/          # GitHub Actions CI & scheduled data pipeline workflows
├── app/                        # Zero-compute Streamlit interactive presentation web app
│   ├── Home.py                 # Main entry point (Action-First overview & KPI dashboard)
│   ├── data_loader.py          # Dual-layer data loader (serving -> sample) via DuckDB
│   ├── labels.py               # Single source of truth for display names, status, glossary
│   ├── state.py                # Session state management, URL query sync, Guided Tour
│   └── pages/                  # Multipage navigation views
│       ├── 1_Hospital_Administrator.py
│       ├── 2_Scenario_Comparison.py
│       ├── 3_Replay.py
│       ├── 4_Model_Evaluation.py
│       └── 5_About.py
├── configs/                    # Simulation and scenario YAML configurations
│   ├── profiles.yaml           # Dev / Full / Evaluation environment scales
│   └── scenarios.yaml          # Parameters for 7 distinct scenario stress-tests
├── data/                       # Parquet data storage
│   ├── raw/                    # Raw simulated transaction logs
│   ├── curated/                # PySpark / DuckDB transformed parquet datasets
│   ├── serving/                # Precomputed analytical serving tables (12 tables)
│   └── sample/                 # Bundled lightweight git-committed sample snapshot (~1 MB)
├── docs/                       # Architecture, user guides, viva scripts, and data dictionary
├── pipeline/                   # Backend computational pipelines
│   ├── __main__.py             # CLI entry: python -m pipeline <subcommand>
│   ├── cli.py                  # Unified CLI implementation (argparse)
│   ├── doctor.py               # Environment diagnostics and prerequisite checker
│   ├── etl/                    # PySpark ETL with automatic zero-JVM DuckDB fallback
│   ├── models/                 # Forecasters (LightGBM/Prophet), Classifiers, Clustering
│   ├── optimizer/              # Google OR-Tools MILP redistribution solver
│   ├── simulator/              # Daily time-stepped multi-facility hospital simulator
│   ├── prepare.py              # Stage 1: Data generation & ETL
│   ├── train.py                # Stage 2: Machine learning model training
│   ├── run_scenario.py         # Stage 3: Scenario simulation & policy evaluation
│   └── export.py               # Stage 4: Serving parquet export
├── tools/                      # Engineering automation tools (data dictionary generator)
├── tests/                      # Pytest automated test suite (74+ tests)
├── pyproject.toml              # Build metadata, CLI console scripts, pytest configuration
├── requirements.txt            # Minimal serving dependencies for Streamlit app
├── requirements-pipeline.txt   # Full data engineering, ML, and optimization dependencies
├── run_demo.ps1                # One-command automated demo runner for Windows (PowerShell)
└── run_demo.sh                 # One-command automated demo runner for Unix / macOS / Linux
```

---

## 2. Execution Profiles

The system uses profiles defined in `configs/profiles.yaml` to balance rapid iteration with production realism:

| Profile | Hospitals | Medicines | Horizon | Monte Carlo Seeds | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`dev`** *(default)* | 10 | 30 | 90 days | 1 - 3 | Rapid local development, unit tests, fast demo runs (~5 minutes). |
| **`full`** | 50 | 100 | 180 days | 3 - 5 | Full regional network simulation for research papers and benchmarks. |
| **`eval`** | 20 | 50 | 120 days | 5 | Cross-validation and model ablation benchmarking. |

You can specify the active profile via the CLI flag `--profile <profile>` or by setting the environment variable:
```bash
# Windows PowerShell
$env:MEDSYS_PROFILE = "dev"

# Linux / macOS
export MEDSYS_PROFILE="dev"
```

---

## 3. Unified CLI Reference

All pipeline operations can be invoked through `python -m pipeline <command>`:

### 3.1 Environment Health Check
```powershell
python -m pipeline doctor
```
Verifies Python version ($\ge 3.12$), Java/JVM presence, required Python libraries, writable directory permissions, and active `ETL_BACKEND`.

### 3.2 End-to-End Demo Execution
```powershell
python -m pipeline demo --profile dev --seeds 1
```
Executes all four pipeline stages consecutively:
1. `prepare` (Simulator + ETL)
2. `train` (Demand forecasting, risk classification, clustering)
3. `run_scenario` (7 scenarios $\times$ 3 policies)
4. `export` (Export 12 precomputed Parquet serving tables)

### 3.3 Individual Pipeline Stages
```powershell
# Stage 1: Generate simulated transactions and run ETL
python -m pipeline prepare --profile dev

# Stage 2: Train ML models and baselines
python -m pipeline train --profile dev

# Stage 3: Run specific scenario and policy
python -m pipeline run_scenario --scenario normal --policy optimizer --profile dev --seeds 1

# Stage 4: Export analytical tables to serving directory
python -m pipeline export --profile dev --target-dir data/serving
```

### 3.4 Launch Dashboard
```powershell
python -m pipeline app
# Equivalent to: streamlit run app/Home.py
```

---

## 4. How to Add a New Scenario

To introduce a new stress test or public health scenario:
1. Open `configs/scenarios.yaml`.
2. Add a new scenario entry under `scenarios:`:
   ```yaml
   epidemic_cholera:
     name: "Cholera Epidemic Surge"
     demand_multiplier: 3.5
     affected_categories: ["Antibiotic", "IV Fluid", "Electrolyte"]
     supply_delay_days: 4
     probability: 0.15
   ```
3. Open `app/labels.py` and register the display name and badge:
   ```python
   SCENARIOS["epidemic_cholera"] = "Cholera Epidemic Surge"
   SCENARIO_TAGS["epidemic_cholera"] = ("New Situation", "novel")
   ```
4. Execute the pipeline:
   ```powershell
   python -m pipeline run_scenario --scenario epidemic_cholera --policy optimizer --profile dev
   python -m pipeline export --profile dev
   ```

---

## 5. Running the Automated Test Suite

Run all unit, integration, and architecture tests with:
```powershell
pytest -ra
```
To run specific subsystems:
```powershell
pytest tests/test_optimizer.py       # MILP solver mathematical correctness
pytest tests/test_streamlit_app.py   # Streamlit UI AppTest rendering & zero-compute audit
pytest tests/test_etl.py             # PySpark & DuckDB ETL schema parity
```

---

## 6. Streamlit Community Cloud Deployment

The repository is pre-configured for **instant 1-click deployment** on Streamlit Community Cloud:

1. Push your code to GitHub (`main` branch).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
3. Select your repository, branch (`main`), and set the Main file path to:
   ```text
   app/Home.py
   ```
4. Under **Advanced settings**, set Python version to **3.12**.
5. Click **"Deploy!"**.
6. The application will start immediately using the bundled `data/sample/` dataset. No heavy dependencies (Spark, LightGBM, OR-Tools) are installed in the cloud instance.
