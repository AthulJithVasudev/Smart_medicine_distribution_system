# Troubleshooting Guide

This guide addresses common setup, runtime, and environmental issues encountered when executing the **Smart Medicine Redistribution System**.

---

## 1. Java & PySpark Issues

### Issue: `JAVA_HOME is not set` or `java: command not found`
**Symptom:** Running PySpark raises `JavaNotFoundException` or `doctor` reports Java FAIL.
**Root Cause:** Apache Spark requires a Java Virtual Machine (JVM), ideally OpenJDK 17 or 11.
**Solutions:**
1. **Zero-JVM Local Fallback (Recommended on Windows):**
   The system includes a native DuckDB ETL engine that generates identical Parquet tables without requiring Java.
   Set the environment variable:
   ```powershell
   $env:ETL_BACKEND = "duckdb"
   ```
2. **Install Java 17:**
   - Download and install [Eclipse Temurin JDK 17](https://adoptium.net/).
   - Set environment variables:
     ```powershell
     [System.Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Eclipse Adoptium\jdk-17.x.x-hotspot", "User")
     $env:Path += ";$env:JAVA_HOME\bin"
     ```

### Issue: `HADOOP_HOME` or `winutils.exe` missing on Windows
**Symptom:** Spark logs warnings or errors about `winutils.exe` when writing Parquet partitions on Windows.
**Solution:**
Use the DuckDB backend: `$env:ETL_BACKEND = "duckdb"`. DuckDB writes Parquet natively via C++ without depending on Hadoop filesystem shims.

---

## 2. Streamlit Dashboard Issues

### Issue: Port 8501 is already in use
**Symptom:** `OSError: [WinError 10048] Only one usage of each socket address is normally permitted`
**Solution:**
Specify an alternative port when launching:
```powershell
streamlit run app/Home.py --server.port 8502
```
Or kill the existing process on Windows:
```powershell
Get-Process -Name python | Where-Object { $_.CommandLine -like "*streamlit*" } | Stop-Process
```

### Issue: App displays "Showing bundled sample data"
**Explanation:** This is an intentional feature. When `data/serving/` has not yet been computed or is removed, the dashboard automatically falls back to `data/sample/` so examiners can evaluate the interface without running a 5-minute pipeline first.
**Solution to display fresh local run:**
Run the demo command:
```powershell
python -m pipeline demo --profile dev
```
Once `data/serving/` is populated, the sample banner disappears automatically upon page refresh.

### Issue: Blank screen or browser connection timeout
**Solution:**
Ensure you access `http://localhost:8501` rather than the network IP if behind a corporate VPN or strict firewall.

---

## 3. Python Version Compatibility

### Issue: Package installation failures on Python 3.13 / 3.14
**Symptom:** `pip install` fails building wheels for `pyarrow`, `ortools`, or `lightgbm`.
**Recommendation:**
The project officially targets **Python 3.12** (pinned in `pyproject.toml` and GitHub Actions). If your system runs Python 3.14, binary wheels may not yet be available for all scientific C-extensions without Visual Studio build tools.
**Solution:**
Install Python 3.12 via `pyenv-win`, winget, or the official python.org installer:
```powershell
winget install Python.Python.3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-pipeline.txt
```

---

## 4. Performance & Execution Speed

### Long run times on full profile
- **Cause:** Running 50 hospitals over 180 days with multiple seeds solves thousands of MILP instances sequentially.
- **Guidance for Presentation / Viva:**
  Always use the **`dev` profile** (`--profile dev --seeds 1`). The DEV profile executes all 4 stages in ~5 minutes on standard laptop hardware while exercising every pipeline step and producing realistic analytical serving tables.
