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


