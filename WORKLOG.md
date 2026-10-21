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
