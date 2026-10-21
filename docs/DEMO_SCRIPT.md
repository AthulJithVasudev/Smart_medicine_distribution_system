# Viva Presentation & Demo Script (5-Minute Walkthrough)

This script is designed for a 5-minute practical demonstration during your project viva examination. It leverages the built-in **Guided Tour** in the dashboard to present a compelling, structured narrative.

---

## 1. Five-Minute Presentation Walkthrough

### Minute 0:00 - 1:00 | The Problem & System Architecture
- **Action:** Open `http://localhost:8501`. Expand the sidebar and click **"1. See normal operations"**.
- **Talking Points:**
  - *"Public hospital networks suffer from a twin paradox: acute stockouts of lifesaving drugs at rural clinics occurring simultaneously with lakhs of rupees of unused medicine expiring at district hospitals."*
  - *"Our system addresses this through Architecture v4: a stateful daily simulator, dual PySpark/DuckDB ETL, ML forecasting, and a Google OR-Tools MILP optimizer."*
  - *"Crucially, our Streamlit dashboard enforces a strict **Zero-Compute Rule**: it never runs heavy models on the fly; it queries precomputed columnar Parquet tables via DuckDB in under 100 milliseconds."*

### Minute 1:00 - 2:00 | Action-First Executive Dashboard (Government Officer View)
- **Action:** Point out Block 1, Block 2, and Block 3 on the Home page.
- **Talking Points:**
  - *"Rather than overwhelming an officer with raw data, our UI is **Action-First**."*
  - *"Block 1 highlights immediate risks: top facilities with imminent deficits and batches within 15 days of expiration."*
  - *"Block 2 generates executable transfer manifests. Notice the plain-language justification for every transfer: it proves why the origin hospital can spare the stock without violating its own safety stock."*
  - *"Click **'Download Transfer Orders (CSV)'** to show how orders integrate into hospital logistics."*

### Minute 2:00 - 3:00 | Stress-Testing Public Health Scenarios
- **Action:** Click sidebar tour button **"2. Trigger a monsoon surge"** or **"3. Compare Smart Plan with Do Nothing"**.
- **Talking Points:**
  - *"The real test of healthcare logistics is resilience during epidemics. We evaluated 7 distinct stress scenarios across multiple random seeds."*
  - *"In Scenario Comparison, notice the automated takeaway computed directly from our runs: **Smart Plan reduced stockout days significantly compared to Do Nothing** while cutting wastage."*
  - *"The chart separates standard seasonal patterns from novel, unprecedented shocks like a 4x dengue epidemic."*

### Minute 3:00 - 4:00 | Facility Level & Historical Auditing
- **Action:** Navigate to **Hospital Administrator** and then click **"4. Check forecast accuracy in Replay"**.
- **Talking Points:**
  - *"A hospital superintendent can inspect their specific facility: stock levels, 14-day demand forecast with 95% uncertainty bands, and upcoming truck deliveries."*
  - *"On the **Replay** page, we provide full historical transparency: examiners can select past dates to inspect what our ML model predicted versus realized patient demand."*

### Minute 4:00 - 5:00 | Empirical Validation & Model Verdicts
- **Action:** Click sidebar tour button **"5. Review Model Evaluation verdicts"**.
- **Talking Points:**
  - *"On the Model Evaluation page, we open with three transparent verdict cards:"*
    - *Forecast accuracy: Our ML model achieves substantially lower WAPE compared to the standard 7-day moving average.*
    - *Safety recall: High recall on identifying impending critical stockout events.*
    - *OPD lead feature ablation: Proves that outpatient registration volume acts as a leading indicator for pharmacy demand.*
  - *"We conclude with the theoretical Oracle bound, showing how close our constrained MILP gets to perfect future hindsight."*

---

## 2. Six Likely Examiner Questions & Data-Backed Answers

### Q1: "Why did you use Mixed-Integer Linear Programming (MILP) instead of Reinforcement Learning or a Deep Neural Network?"
**Answer:**
> *"In public healthcare logistics, decisions must be **strictly feasible, bounded, and explainable**. A deep neural network or RL agent can output hallucinated or infeasible transfers (e.g. dispatching stock a hospital doesn't have, or exceeding vehicle capacities) and offers zero mathematical optimality guarantees. OR-Tools MILP guarantees that safety stock constraints, transit feasibility, and anti-dumping caps are strictly respected ($100\%$ hard constraint satisfaction) while solving in seconds."*

---

### Q2: "How do you prevent 'dumping'—where a large hospital offloads near-expiry drugs onto a small rural clinic that can't use them?"
**Answer:**
> *"We implemented a hard **Destination Need Cap** constraint in our MILP formulation:*
> $$\sum_{i, b} x_{ijmb} \le \text{need}_{jm}$$
> *Inflow to receiving hospital $j$ is strictly bounded by its predicted patient deficit over the planning horizon. A hospital can never receive more units than its patient population is forecasted to consume before expiration. Furthermore, batches with remaining shelf-life shorter than transit time plus buffer are strictly forbidden ($x=0$)."*

---

### Q3: "What happens if your Machine Learning demand forecast is inaccurate or biased?"
**Answer:**
> *"The MILP optimizer does not consume raw point forecasts naively. First, the availability constraint reserves an explicit **safety stock buffer**:*
> $$\text{avail}_{imb} = \max(0, \text{stock}_{imb} - \text{forecast\_demand}_{im} - \text{safety\_stock}_{im})$$
> *Safety stock is dynamically scaled based on forecast variance and replenishment lead time. Even if demand spikes above expectation, the origin hospital remains protected. Second, we evaluate the system under a wide range of demand shock scenarios (up to 4x surge) to measure graceful degradation."*

---

### Q4: "Why does the Streamlit app strictly read precomputed Parquet files rather than solving on the fly?"
**Answer:**
> *"This is a deliberate architectural principle: the **Zero-Compute Serving Rule**. In a state government setting, senior health directors need sub-second response times on mobile or web dashboards without spinning up heavy GPU/JVM cluster compute. Running PySpark and MILP solvers on every page refresh causes timeouts, race conditions, and massive server costs. Instead, batch pipelines run overnight or hourly, materializing immutable Parquet serving tables that DuckDB queries in-memory in under 100 milliseconds."*

---

### Q5: "Is your data from real government hospitals or is it simulated?"
**Answer:**
> *"To ensure scientific rigor without violating patient privacy or NDAs, we built a **stateful daily time-stepped simulator** calibrated to real-world Indian public health tier structures (PHCs, CHCs, SDHs, and District Hospitals from HMIS data patterns). We clearly flag all data with provenance labels: `SIMULATED`, `DERIVED`, and `OFFICIAL`. The entire data ingestion layer is designed so that when connecting to live state systems like e-Aushadhi or DVDMS, only the ETL connector changes; the optimizer and models remain identical."*

---

### Q6: "How does this system scale to hundreds of hospitals across a state?"
**Answer:**
> *"We scale across three tiers:*
> 1. *ETL runs on **Apache Spark**, enabling horizontal distribution across Hadoop/Databricks clusters for millions of daily transaction logs.*
> 2. *Our K-Means clustering partitions large state networks into localized geographic clusters (districts/zones). Rather than solving one massive $1000 \times 1000$ MILP matrix, the optimizer decomposes into regional sub-problems solved in parallel.*
> 3. *The serving layer uses columnar Parquet with dictionary encoding and snappy/zstd compression, allowing DuckDB to query millions of rows in memory without performance degradation."*
