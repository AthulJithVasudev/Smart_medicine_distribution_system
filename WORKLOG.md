# Work Log & Execution Audit

## Environment Information
- **Operating System:** Windows 11 (AMD64 / win32)
- **Shell:** PowerShell
- **Python Version:** 3.14.7 (Virtual environment `.venv` active)
- **Java Version:** None detected (`java` not recognized in PATH; JDK not installed)
- **Installed Key Packages:**
  - `pandas`: 3.0.6
  - `duckdb`: 1.5.6
  - `pyarrow`: 25.0.1
  - `streamlit`: 1.66.0
  - `ortools`: 9.15.6755
  - `statsmodels`: 0.15.0
  - `scikit-learn`: 1.9.1
  - `mlflow`: 3.17.2
  - `plotly`: 7.1.0
  - `pydeck`: 0.9.3
  - `pytest`: 9.1.1
  - `pyspark`: Not installed / Python 3.14 + no JVM available. Fallback DuckDB/Arrow ETL active.
- **ETL Backend:** PySpark local-mode with DuckDB/Arrow fallback engine.

---

## Phase 0: Diagnose and get everything running (P0)

### Initial Diagnostic & Verification
1. **Pytest Run:** Initial run failed due to missing virtual environment dependencies (`pytest`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `ortools`, `streamlit`, `plotly`, `pydeck`, `mlflow`, `pytest-cov`).
2. **Dependency Installation:** Installed all required pipeline and app packages into `.venv`.
3. **Test Suite Execution:** Ran `pytest -ra`. Result: **58 passed, 15 warnings** across all test modules.
4. **Pipeline Stage Verifications:**
   - `python -m pipeline.prepare --profile dev`: PASSED (Simulated 25,926 records, 22,013 curated records).
   - `python -m pipeline.train --profile dev`: PASSED (Forecasting WAPE with OPD: 0.616 vs without OPD: 0.6356; baselines evaluated; models saved).
   - `python -m pipeline.run_scenario --scenario normal --policy optimizer --profile dev --seeds 1`: PASSED (Closed loop simulation & MILP optimization completed).
   - `python -m pipeline.export --profile dev --target-dir data/serving`: PASSED (12 compact serving Parquet tables exported).
   - `streamlit run app/Home.py`: PASSED (Loaded and rendered successfully with serving data).
5. **PySpark Contingency & ETL Backend:**
   - Implemented `ETL_BACKEND` support (`spark` default, `duckdb` fallback).
   - Added `test_etl_backend_fallback_duckdb` to `tests/test_etl.py` to ensure feature equivalence.
6. **Package Specification Fixes:**
   - Kept minimal serving requirements in `requirements.txt` (Streamlit, pandas, duckdb, plotly, pydeck, pyarrow).
   - Pinned `requires-python = ">=3.12"` in `pyproject.toml`.
   - Referenced `requirements.txt` inside `requirements-pipeline.txt`.

**Phase 0 Status: COMPLETE AND VERIFIED.**

---

## Phase 1: Dashboard works with zero pipeline steps (P0)

### Actions Completed:
1. **Sample Snapshot Generation:** Generated a compact sample snapshot (10 hospitals, 30 medicines, 7 scenarios, 3 policies) and published it to `data/sample/`. Total snapshot size is ~1.0 MB (compressed Parquet), comfortably under the 20 MB budget.
2. **Data Loader Fallback Hierarchy:** Updated `app/data_loader.py` to prioritize `data/serving/` and automatically fall back to `data/sample/`.
3. **Sample Banner Notification:** Added `render_sample_banner()` in `app/data_loader.py` to display a clear notification when sample data is shown or if data directories need to be generated via the demo command.
4. **DuckDB Querying & Caching:** Added DuckDB-based selective column querying with SQL `WHERE` clauses and 300s TTL cache in `app/data_loader.py`, while handling Windows POSIX path formatting.
5. **Test Suite Expansion:** Added tests in `tests/test_streamlit_app.py` for sample fallback (`test_sample_fallback_when_serving_missing`) and DuckDB projection (`test_duckdb_projection_and_filtering`). All 6 tests in `test_streamlit_app.py` pass.

**Phase 1 Status: COMPLETE AND VERIFIED.**

---

## Phase 2: One-command run (P0)

### Actions Completed:
1. **Unified CLI Subcommands (`pipeline/cli.py` & `pipeline/__main__.py`):**
   - Added subcommands: `demo`, `doctor`, `app`, `prepare`, `train`, `run_scenario`, `export`.
   - Enabled invocation via `python -m pipeline <cmd>` and registered `pipeline` and `medsys` script entrypoints in `pyproject.toml`.
2. **Environment Variable Default Profile (`MEDSYS_PROFILE`):**
   - Profile defaults to `os.environ.get("MEDSYS_PROFILE", "dev")` across all entry points, making `--profile dev` optional while preserving manual override.
3. **Automated Shell Runners:**
   - Created `run_demo.ps1` for Windows PowerShell (handles venv creation, pip upgrade/install, system doctor check, and demo pipeline execution).
   - Created `run_demo.sh` for Linux/macOS/Bash environments.
4. **Diagnostic Doctor (`pipeline/doctor.py`):**
   - Added diagnostic suite checking Python (>=3.12), Java/PySpark status with fallback notice, 14 required packages, writable directories, and `ETL_BACKEND`.
5. **Measured Timings & Progress Tracking:**
   - Added step-by-step progress logging with elapsed times and row counts per stage.
   - Benchmark on Windows development laptop (DEV profile, 1 seed):
     - Stage 1 (Prepare & ETL): 28.5s (22,013 curated rows)
     - Stage 2 (Train Models): 42.1s (baselines + hybrid forecasters + clustering + risk classification + anomaly)
     - Stage 3 (Scenarios Matrix): 227.9s (21 scenario-policy runs)
     - Stage 4 (Export Parquet): 1.0s (12 serving tables)
     - Total runtime: 299.5s (~5.0 minutes)
6. **Testing Suite:**
   - Created `tests/test_cli_and_doctor.py` covering doctor diagnostics, default profile resolution, and CLI subcommands. All 7 tests pass.
7. **Acceptance Verification:**
   - Executed `powershell -ExecutionPolicy Bypass -File .\run_demo.ps1 -SkipInstall`. Completed all stages cleanly and printed dashboard launch instructions.

**Phase 2 Status: COMPLETE AND VERIFIED.**

---

## Phase 3: Correctness checks on the optimizer and claims (P1)

### Code vs README Mathematical Formulation Analysis:
- **Wastage Term ($w_{imb}$):** The previous objective had only transport cost, shortage penalty, and FEFO tie-break epsilon. We enhanced `pipeline/optimizer/solver.py` with:
  - Decision variables $w_{imb} \ge 0$ for units at risk of expiring unused.
  - Constraint: $w_{imb} + \sum_j x_{ijmb} \ge \text{at\_risk}_{imb}$ where $\text{at\_risk}_{imb}$ is calculated pre-solve via FEFO allocation of forecast demand.
  - Objective penalty term: $\gamma \cdot V_m \cdot w_{imb}$ with parameter `wastage_penalty_weight = 5.0`.
- **Timing Constraint:** Verified that $x_{ijmb} = 0$ when transit exceeds planning horizon (`transit_days > self.horizon_days`) and when shelf-life on arrival is insufficient (`remaining_days_at_arrival < self.min_shelf_life`). Added unit test `test_timing_constraint_transit_exceeds_horizon`.
- **Safety Stock Computation:** Confirmed that $\text{avail}_{im} = \max(0, \text{stock} - \text{horizon\_demand} - \text{safety\_stock})$ is computed as non-negative data before solve. Proved that $x = 0$ is always feasible. Added unit test `test_hospital_already_in_shortage_never_source`.
- **Receiving Cap:** Verified that $\sum_i x_{ijmb} \le \text{need}_{jm}$ strictly caps inflow to destination demand. Verified by `test_mandatory_receiving_cap`.
- **Local Consumption Feasibility:** Added test `test_shipping_not_forced_when_consumed_locally` showing that when stock is consumed locally, no transfers are forced.
- **Wording Updates:**
  - "strictly prioritizes earlier-expiring batches" $\to$ "breaks ties toward earlier-expiring batches (soft FEFO)"
  - "NEVER dumped" $\to$ "inflow is capped at the forecast need, so surplus is not pushed onto receiving hospitals (forecast error can still leave some excess)"
  - "Discrete-event simulation" $\to$ "daily time-stepped simulation" across code and README.
  - "generated deterministically" $\to$ "generated reproducibly from fixed random seeds".
  - Replaced hardcoded "56+ tests" with live CI status badge.

**Phase 3 Status: COMPLETE AND VERIFIED (71/71 tests passing).**

---

## Phase 4: Make the dashboard easy to navigate and use (P1)

### Tasks Completed:
1. **Shared UI Layer (`app/labels.py`):**
   - Single source of truth for all human-readable policy names (`Smart plan`, `Nearest surplus`, `Do nothing`).
   - Plain-language scenario names with categorized "seen before" vs "new situation" tags.
   - Status badges combining text and accessible color indicators (`Critical shortage`, `Low stock`, `Normal`, `Surplus`, `Expiring soon`, `Active transfer`).
   - Comprehensive system glossary dictionary with inline tooltips (`help=` args).
2. **State Management & URL Synchronization (`app/state.py`):**
   - Centralized `st.session_state` management for scenario, policy, simulation day, and district filters.
   - Automatically initializes default day to the day with the highest network shortages.
   - Synchronizes filter selections bidirectionally with `st.query_params` for shareable URL links.
   - Built-in 5-step **"Guided Demo Tour"** in the sidebar.
3. **Action-First Home Page (`app/Home.py`):**
   - Implemented 3-role selector: *Government Health Officer*, *Hospital Administrator*, *Researcher / Data Scientist*.
   - **Block 1: What needs attention now** (Top 5 critical shortages, top 5 expiring batches, high-visibility KPI metric cards).
   - **Block 2: What should be done** (Action cards with plain-language justification, vehicle logistics cost, and "Download Transfer Orders (CSV)" button).
   - **Block 3: What it achieves** (Wastage prevented in units and INR, stockouts avoided, fill rate comparison vs Do Nothing, Plotly trend chart, and interactive OpenStreetMap visualization).
   - Persistent banner for bundled sample data and synthetic data disclaimer.
4. **Canonical Pages Overhaul:**
   - `app/pages/1_Hospital_Administrator.py`: Facility selector, days of supply, demand forecast with 95% confidence bands, incoming/outgoing manifests, CSV download.
   - `app/pages/2_Scenario_Comparison.py`: Standard vs novel scenario tabs, multi-seed spread, dynamic takeaway summary computed from real run data.
   - `app/pages/3_Replay.py`: Historical forecast auditing against realized patient demand with MAE and bias metrics.
   - `app/pages/4_Model_Evaluation.py`: Verdict cards (ML vs 7-day average WAPE, risk classifier recall, anomaly detection), OPD feature ablation expander, theoretical Oracle bound.
   - `app/pages/5_About.py`: 2-minute quickstart, interactive glossary, architecture breakdown, and limitations.
   - Removed obsolete legacy pages (`1_DATA.py`, `2_ANALYSIS.py`, `3_RECOMMENDATIONS.py`, `4_MODEL.py`, `retro_theme.py`).
5. **Testing & Quality Audit:**
   - Extended `tests/test_streamlit_app.py` with AST zero-compute import verification, sample data loader fallback test, and Streamlit `AppTest` automated rendering across all 6 pages.
   - Verified that all 9 Streamlit tests pass cleanly.

**Phase 4 Status: COMPLETE AND VERIFIED.**

---

## Phase 5: README and Documentation (P1)

### Tasks Completed:
1. **Restructured `README.md`:**
   - Immediate title, one-line summary, CI status badge, and `<LIVE_APP_URL>` placeholder.
   - **Quick start in three commands** (Option A: Instant dashboard with bundled sample data; Option B: One-command full pipeline run via `run_demo.ps1`).
   - 5 core dashboard capabilities.
   - Clean Mermaid architecture flowchart.
   - Technology stack matrix and clear system limitations.
   - Comprehensive links to `docs/`.
2. **Detailed Documentation Suite in `docs/`:**
   - `docs/ARCHITECTURE.md`: Complete v4 architecture, component interactions, and verified OR-Tools MILP formulation.
   - `docs/USER_GUIDE.md`: 2-minute page tour, role explanations, status badge color conventions, and guided tour steps.
   - `docs/DEVELOPER.md`: Repo structure, profiles (`dev`, `demo`, `full`), CLI subcommands, adding scenarios, and Streamlit Cloud deployment.
   - `docs/TROUBLESHOOTING.md`: Guidance for Java/JVM missing, Windows winutils, port conflicts, sample fallback, and Python versions.
   - `docs/DATA_DICTIONARY.md`: Generated via `tools/make_data_dictionary.py` directly from the 12 serving Parquet tables.
   - `docs/DEMO_SCRIPT.md`: 5-minute viva presentation walkthrough with 6 rigorous examiner questions and answers.
   - `docs/CLAIMS_TO_EVIDENCE.md`: Table of every claim mapped to empirical metrics, unit tests, and dashboard pages (zero fabricated numbers).
3. **Automated Schema Tool:**
   - Created `tools/make_data_dictionary.py` using DuckDB to auto-document column types, descriptions, row counts, and sample schemas.

**Phase 5 Status: COMPLETE AND VERIFIED.**

---

## Phase 6: Cloud Readiness & Polish (P2)

### Tasks Completed:
1. **GitHub Actions Workflows:**
   - Verified `.github/workflows/ci.yml` (Ubuntu, Java 17, Python 3.12, Pytest suite).
   - Enhanced `.github/workflows/pipeline.yml`: Added `permissions: contents: write`, `actions/setup-java@v4` (Java 17), `actions/setup-python@v5` (Python 3.12), artifact upload, and automated orphan branch publication (`serving-data`) for cloud serving data syncing.
2. **Streamlit Community Cloud Readiness:**
   - Pinned root `requirements.txt` to strictly serving dependencies (`streamlit`, `pandas`, `duckdb`, `plotly`, `pydeck`, `pyarrow`).
   - Entry point is `app/Home.py`.
   - Dashboard opens instantly from `data/sample/` with zero pipeline execution required.
   - Documented step-by-step deployment instructions in `docs/DEVELOPER.md`.
3. **Devcontainer:**
   - Added `.devcontainer/devcontainer.json` featuring Python 3.12, Java 17, port forwarding for 8501, VS Code Python extensions, and postCreateCommand.
4. **System Doctor Verification:**
   - `python -m pipeline doctor` passes all checks cleanly with expected Java fallback notice.

**Phase 6 Status: COMPLETE AND VERIFIED.**

---

## Final Verification Checklist Results

| Check | Requirement | Result | Evidence |
| :-: | :--- | :---: | :--- |
| **1** | Fresh clone with minimal `requirements.txt` launches dashboard on sample data | **PASS** | `data/sample/*.parquet` bundled; `AppTest` verifies all 6 pages render cleanly. |
| **2** | `python -m pipeline doctor` passes | **PASS** | Python, packages, directories, backend checked and reported green. |
| **3** | `.\run_demo.ps1` runs end-to-end on `dev` profile | **PASS** | Measured ~5.0 minutes execution time across all 4 stages. |
| **4** | Full test suite `pytest -ra` is green | **PASS** | **74 passed, 0 failed** across 11 test files in 7m 51s. |
| **5** | Zero-compute rule: `app/` contains no ML/solver imports | **PASS** | Verified by AST analysis in `test_app_contains_no_training_or_optimization_imports`. |
| **6** | README & `ARCHITECTURE.md` formulation matches `solver.py` | **PASS** | Anti-dumping receiving cap, soft FEFO, timing constraint, and explicit wastage verified. |
| **7** | No fabricated numbers: all figures come from data | **PASS** | All metrics in `CLAIMS_TO_EVIDENCE.md` extracted via DuckDB from actual run parquet files. |
| **8** | `WORKLOG.md` complete and up-to-date | **PASS** | Full audit trail maintained across Phases 0 through 6. |

---

---

## Phase 9: Scenario Audit, Simulation Logic, and Dashboard Reliability Overhaul

### Problem Statement & Investigation
An end-to-end trace was performed to address reports of scenarios showing no meaningful policy separation, unexpected negative optimizer performance, and dashboard comparison discrepancies.
The trace revealed 6 foundational root causes:
1. **Pipeline Inventory Blindness / Bullwhip Churning**: `solver.py` and `GreedyNearestSurplusPolicy` ignored `pending_transfers`. Orders placed on day $t$ take 1–3 days to arrive. Without in-transit awareness, both policies re-ordered duplicate transfers daily, churned excessive stock, and drained source hospitals of shelf inventory before shipments arrived.
2. **Reporting Delay Dropping Hospitals**: `pipeline/run_scenario.py` filtered `curated_daily[curated_daily["report_day"] == max()]`. Hospitals subject to observation reporting delays (e.g., rural PHCs) were excluded from feature prediction, falling back to a default low demand rate (12 units/day). This caused the optimizer to misidentify them as having massive surplus, stripping their stock immediately prior to patient demand surges.
3. **Initial Expiry Horizon Inactivity**: `Simulator._init_inventory()` seeded initial batches with expiry in $[25, 90]$ days. Across a 14-day evaluation horizon, 0 units ever reached expiration, rendering FEFO redistribution inert. Staggering batch expiries activated FEFO wastage prevention.
4. **Missing Export Metadata**: `transfers.parquet` and `state_daily.parquet` lacked `scenario`, `policy`, and `seed` columns, causing cross-scenario and cross-policy record contamination in dashboard filters.
5. **Dashboard Self-Comparison**: `app/Home.py` compared `selected_policy` vs `no_transfer`, causing a zero-difference display whenever `no_transfer` was selected in the sidebar. Furthermore, `day_state` summed metrics across multiple random seeds without filtering by primary seed.
6. **Parquet Caching Latency**: `app/data_loader.py` cached tables without checking underlying file `st_mtime`, causing regenerated simulation data to remain stale until manual server reboot.

### Verified Fixes Implemented:
1. **In-Transit Pipeline Tracking**: Added `in_transit_incoming` to `RedistributionOptimizer` and `GreedyNearestSurplusPolicy`. In-transit units are credited toward receiving need, eliminating duplicate ordering while preserving local safety buffers.
2. **Delayed Reporting Recovery**: In `run_scenario.py`, selected the latest available report per hospital via `.groupby(["hospital_id", "medicine_id"]).last()`, preventing rural facility dropouts.
3. **Batched Expiry Calibration**: Staggered initial batch expiries so ~20% of batches reach expiry in days 5–14, activating FEFO redistribution.
4. **Export Schema Harmonization**: Stamped `scenario`, `policy`, and `seed` across all transfer and daily state records; saved `state_daily.parquet` directly to matrix and serving outputs.
5. **Comparison & Metric Clarity**: In `app/Home.py`, decoupled the baseline benchmark comparison (Smart Plan vs Baseline) from the operational day view filter; added primary seed filtering and Daily vs. Cumulative unmet demand toggle.
6. **File mtime Cache Invalidation**: Added `mtime` to `@st.cache_data` in `app/data_loader.py` and added a manual cache refresh button in `app/state.py`.
7. **Automated Regression Suite**: Added 11 automated regression tests in `tests/test_scenario_regression.py` covering all scenarios, policies, accounting consistency, safety buffers, and cache behavior.

### Verification Results:
- `pytest tests/test_scenario_regression.py`: **11 passed, 0 failed**
- `pytest tests/test_streamlit_app.py`: **11 passed, 0 failed**
- `pytest tests/test_scenarios.py`: **7 passed, 0 failed**
- `pytest tests/test_optimizer.py`: **10 passed, 0 failed**
- `pytest tests/test_evaluation.py`: **10 passed, 0 failed**
- `pytest tests/test_pipeline_and_publishing.py`: **2 passed, 0 failed**
- Total core test suite: **51 passed, 0 failed in 3m 04s**.





