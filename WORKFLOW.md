# Development Workflow

Step-by-step development plan for the India Crime Intelligence Platform — a PySpark big data pipeline that ingests 15 NCRB datasets, performs multi-dimensional analytics, and generates an interactive 6-view dashboard.

---

## Phase 0: Setup & Prep

> Get the project organized before touching any code.

- [ ] Create folder structure: `src/`, `output/dashboard_data/`, `dashboard/`, `dashboard/assets/`
- [ ] Upload ALL 15 CSVs to HDFS
  ```bash
  hdfs dfs -mkdir -p /crime/input
  hdfs dfs -mkdir -p /crime/output
  hdfs dfs -put data/*.csv /crime/input/
  ```
- [ ] Verify `india_states.geojson` has all 36 states/UTs — note exact `NAME_1` values (this is what the choropleth keys off of)
- [ ] Build a **single authoritative state name mapping** dict that covers every variation across all 15 CSVs + the GeoJSON `NAME_1` values → one canonical form. This is the #1 source of bugs

**Deliverable:** Clean project structure, all data on HDFS, one master state name map.

---

## Phase 1: Data Preparation

> One script (`data_preparation.py`) that ingests all 15 CSVs and outputs 2 clean master tables.

### 1.1 — District-Level Master Table

**Input:** 6 district-level CSVs (3 IPC + 3 Women)

- [ ] **Union IPC files** — Load `01_*_2001_2012`, `01_*_2013`, `01_*_2014`. Normalize column names (the 2014 file has different headers like `"Rape"` vs `"RAPE"`). Union into single DataFrame
- [ ] **Union Women files** — Same for `42_District_wise_*`. Normalize `"Dowry Deaths"` vs `"DOWRY DEATHS"` etc.
- [ ] **Clean state/district names** — Apply the master state name map. Uppercase + trim districts. Filter out `"TOTAL"` aggregate rows
- [ ] **Join IPC + Women** — Left join on `(state, district, year)`. Log any mismatched districts
- [ ] **Compute derived columns:**
  - `total_violent = murder + attempt_to_murder + culpable_homicide + rape + kidnapping + dacoity + robbery`
  - `total_property = burglary + theft + auto_theft + cheating + criminal_breach_of_trust`
  - `total_women = rape_women + dowry_deaths + kidnapping_women + domestic_cruelty + assault_women + insult_modesty`
- [ ] **Save** → `HDFS: /crime/output/district_master` (Parquet)

**Output schema:** `state | district | year | murder | attempt_murder | rape | kidnapping | dacoity | robbery | burglary | theft | auto_theft | riots | cheating | arson | dowry_deaths | hurt | other_ipc | total_ipc | rape_women | kidnapping_women | dowry_deaths_women | assault_women | insult_modesty | domestic_cruelty | total_women | total_violent | total_property` (~10,500 rows × ~28 columns)

### 1.2 — State-Level Master Table

**Input:** District master (aggregated up) + 9 supplementary state-level CSVs

- [ ] **Aggregate district → state** — GroupBy `(state, year)`, sum all crime columns from the district master
- [ ] **Load Property data (10)** — Group by `(state, year)`, sum `value_stolen`, `value_recovered`. Compute `recovery_rate = recovered/stolen × 100`. Break down by `Group_Name` (Burglary, Robbery, Theft, Dacoity)
- [ ] **Load Auto Theft (30)** — Group by `(state, year)`. Keep `stolen` and `recovered` totals + breakdown by vehicle type (`Sub_Group_Name`: motorcycles, cars, buses, trucks, other)
- [ ] **Load Murder Demographics (32)** — Pivot on `Sub_Group_Name` (Male/Female). Keep age brackets: `upto_10`, `10_15`, `15_18`, `18_30`, `30_50`, `above_50`. Output: `murder_victims_male`, `murder_victims_female`, `murder_victims_young` (under 18), `murder_victims_old` (above 50)
- [ ] **Load Culpable Homicide (33)** — Same structure as murder. Pivot male/female, keep age brackets
- [ ] **Load Firearms (34)** — Keep `murder_by_firearms`, `murder_by_licensed_arms`. Compute `unlicensed_rate = (total - licensed) / total`
- [ ] **Load Kidnapping Purpose (39)** — Pivot on `Sub_Group_Name` to get columns: `kidnap_adoption`, `kidnap_begging`, `kidnap_illicit_intercourse`, `kidnap_marriage`, `kidnap_prostitution`, `kidnap_ransom`, `kidnap_revenge`, `kidnap_sale`, `kidnap_slavery`, `kidnap_unlawful`, `kidnap_other`, `kidnap_total`. Keep male/female breakdowns
- [ ] **Load Serious Fraud (31)** — Keep fraud brackets: `fraud_1_10cr`, `fraud_10_25cr`, `fraud_25_50cr`, `fraud_50_100cr`, `fraud_above_100cr`, `total_fraud`
- [ ] **Load Crime by Place (17)** — Union the 3 files (2001-12, 2013, 2014 — the 2014 file has completely different column names). Aggregate into: `crimes_residential`, `crimes_highway`, `crimes_railway`, `crimes_bank`, `crimes_commercial`, `crimes_other`
- [ ] **Load Women Cases Pipeline (42_Cases)** — Group by `(state, year)`. Pivot on `Sub_Group_Name`. Keep: `cases_reported`, `cases_chargesheeted`, `cases_sent_trial`, `cases_convicted`, `cases_acquitted`, `cases_pending_trial`, `cases_pending_investigation`. Compute: `chargesheet_rate`, `conviction_rate`, `pendency_rate`
- [ ] **Master Join** — Left join everything on `(state, year)`. IPC aggregated is the base table
- [ ] **Save** → `HDFS: /crime/output/state_master` (Parquet)

**Output:** ~500 rows × ~70+ columns (35 states × 14 years, with gaps in supplementary data that only goes to 2010)

### 1.3 — Validation

- [ ] No duplicate `(state, district, year)` in district master — `groupBy().count().filter(count > 1)`
- [ ] No "TOTAL" rows leaked through
- [ ] State names match GeoJSON — cross-reference with `NAME_1` values
- [ ] Null rate per column — flag any column with >20% nulls
- [ ] Year coverage — every state should have 2001–2014 for IPC. Supplementary data (murder, kidnapping, firearms, fraud) only goes to 2010 — that's expected

**Deliverable:** Two clean Parquet files on HDFS. A printed validation report.

---

## Phase 2: Analytics

> One script (`analytics.py`) that computes all analysis results and exports JSON files for the dashboard.

### 2.1 — District-Level Analytics

- [ ] **Crime Severity Score** — Weighted score: `murder×10 + rape×8 + kidnapping×6 + dacoity×5 + robbery×5 + burglary×2 + theft×1 + riots×3 + arson×4 + dowry_deaths×8`. Normalize to 0–100 (min-max per year)
- [ ] **Women Safety Index** — `rape_women + dowry_deaths×2 + domestic_cruelty + assault_women + kidnapping_women`. Normalize 0–100, **invert** (100 = safest). Per district per year
- [ ] **YoY Growth Rate** — For each district: `(crime_this_year - crime_last_year) / crime_last_year × 100`. Use `Window` with `lag()`
- [ ] **District Hotspot Ranking** — For 2014: rank districts by severity score. Also rank by avg YoY growth (2010–2014) to find **rising** hotspots
- [ ] **District Crime Profile** — Compute `pct_violent`, `pct_property`, `pct_women`. Classify each district by dominant type

**Save:** `output/dashboard_data/district_analysis.json`

### 2.2 — State-Level Clustering

- [ ] **Feature selection** — Use 8–10 features: `avg_ipc_crimes`, `ipc_variability` (stddev), `avg_crimes_women`, `avg_murder`, `avg_rape`, `avg_robbery`, `avg_recovery_rate`, `avg_auto_theft`, `avg_fraud`, `avg_firearms_murder`
- [ ] **StandardScaler** — Normalize features (mean=0, std=1)
- [ ] **K selection** — Silhouette score for k=2 to k=8, pick highest
- [ ] **KMeans** — Spark MLlib KMeans on scaled features
- [ ] **Auto-label clusters** — Based on centroid values: e.g., "High Crime, Low Recovery" / "Moderate Crime, High Women Safety" / "Low Crime, Efficient Justice"

**Save:** `output/dashboard_data/clusters.json`

### 2.3 — Crime Composition Profiles

- [ ] **Per-state composition** — `pct_murder`, `pct_rape`, `pct_kidnapping`, `pct_robbery`, `pct_burglary`, `pct_theft`, `pct_riots`, `pct_cheating`, `pct_arson`, `pct_dowry_deaths` (each as % of total IPC)
- [ ] **Radar chart axes** — Group into 5 dimensions: `Violent` (murder+rape+kidnapping+dacoity), `Property` (burglary+theft+auto_theft+cheating), `Women` (dowry+assault+cruelty), `Public Order` (riots+arson), `White Collar` (cheating+fraud+breach_of_trust)

**Save:** `output/dashboard_data/crime_profiles.json`

### 2.4 — Women Safety Deep-Dive

- [ ] **State-level women safety index** — Same formula as district, aggregated to state
- [ ] **Justice pipeline** — From `42_Cases`: funnel per state: `reported → chargesheeted → sent_to_trial → convicted`. Compute drop-off rates at each stage
- [ ] **Crime type breakdown** — Per state: rape vs dowry vs kidnapping vs cruelty vs assault as % of total women crimes
- [ ] **National trend** — Total women crimes per year, broken down by type

**Save:** `output/dashboard_data/women_safety.json`

### 2.5 — Supplementary Analyses

- [ ] **Property recovery** — Per state: total stolen, recovered, recovery rate. Per crime type recovery rates. Trend over years
- [ ] **Kidnapping motives** — Per state: % for each purpose (marriage, ransom, prostitution, etc.). National trend: how motives shifted 2001→2010
- [ ] **Murder demographics** — Per state: % male vs female victims, % by age bracket. National trend
- [ ] **Firearms usage** — Per state: % murders using firearms, % licensed vs unlicensed. National trend
- [ ] **Crime geography** — Per state: % residential vs highway vs railway vs bank vs commercial. National trend
- [ ] **Auto theft** — Per state: stolen vs recovered by vehicle type. Recovery rates

**Save:** `output/dashboard_data/supplementary.json`

### 2.6 — Forecasting (2015–2020)

- [ ] **Data** — State-level `total_ipc_crimes` and `total_women_crimes` per year (2001–2014)
- [ ] **Model** — Ridge regression with polynomial features (degree 2 and 3). Pick via TimeSeriesSplit CV (3 splits). Drop GBR (overfits on 14 data points)
- [ ] **Handle interpolated data** — Mark interpolated rows, don't train on them
- [ ] **Forecast** — Predict 2015–2020 per state. Clip negatives to 0
- [ ] **Tag rows** — Each row gets `type: "actual"` or `type: "predicted"` (dashboard uses this for solid vs dashed lines)

**Save:** `output/dashboard_data/forecasts.json`

### 2.7 — National Trends

- [ ] Total IPC crimes per year (national sum)
- [ ] Total women crimes per year
- [ ] Crime rate change 2001→2014 (% change)
- [ ] Top 5 fastest growing crime types
- [ ] Top 5 declining crime types

**Save:** `output/dashboard_data/national_trends.json`

**Deliverable:** 7 JSON files in `output/dashboard_data/`, ready for the dashboard.

---

## Phase 3: Dashboard

> A single interactive `index.html` with 6 tabbed views. Pure HTML + CSS + JS (Chart.js + Leaflet). No server needed.

### 3.1 — Skeleton & Navigation

- [ ] HTML structure — single page with top nav bar (6 tabs). Each tab is a `<div>` that shows/hides
- [ ] CSS — clean theme, CSS Grid layouts, responsive
- [ ] Tab switching — pure JS click handlers
- [ ] Data loading — all JSON data embedded inline as `<script>var data = {...}</script>` blocks (generated by `visualization.py`)

### 3.2 — Tab 1: India Overview Map

- [ ] **Choropleth map** — Leaflet + GeoJSON. States colored by selected metric
- [ ] **Year slider** — range input (2001–2014), map updates on slide
- [ ] **Metric dropdown** — switch between: Total IPC, Violent Crimes, Property Crimes, Women Crimes, Severity Score
- [ ] **Dynamic color scale** — based on data range for that metric/year. Sequential palette (yellow → orange → red → dark red)
- [ ] **Legend** — bottom-right, shows color → value range. Updates when metric changes
- [ ] **Tooltips** — hover state → name + value + national rank
- [ ] **Click interaction** — click state → mini detail card below map with key stats

### 3.3 — Tab 2: District Deep-Dive

- [ ] **State selector dropdown**
- [ ] **Year selector dropdown** (2001–2014)
- [ ] **Horizontal bar chart** — all districts in selected state, sorted by severity score. Color-coded by crime profile (red = violent, blue = property, purple = women)
- [ ] **Hotspot cards** — top 5 most dangerous districts nationally + top 5 fastest rising
- [ ] **District detail on click** — crime breakdown donut + trend sparkline

### 3.4 — Tab 3: Women Safety

- [ ] **India map** — colored by Women Safety Index (green = safe, red = dangerous)
- [ ] **Crime type stacked bar** — per state: rape, dowry, kidnapping, cruelty, assault breakdown
- [ ] **Justice funnel** — for selected state: Reported → Chargesheeted → Trial → Convicted. Show drop-off % at each stage
- [ ] **National trend lines** — total women crimes over time, broken down by type

### 3.5 — Tab 4: Trends & Forecasting

- [ ] **State multi-select** — pick 1–3 states to compare
- [ ] **Line chart** — solid for actual (2001–2014), dashed for predicted (2015–2020). Different color per state
- [ ] **Metric toggle** — Total IPC vs Women Crimes
- [ ] **National overlay** — checkbox to show/hide national average
- [ ] **Model info** — small text below: model used + CV MAE for selected state

### 3.6 — Tab 5: Crime Profiles

- [ ] **Radar chart** — 5 axes (Violent, Property, Women, Public Order, White Collar). Compare 1–2 states
- [ ] **Kidnapping motives** — donut chart per state (marriage, ransom, prostitution, etc.)
- [ ] **Murder demographics** — stacked bar (male/female × age brackets)
- [ ] **Crime geography** — donut (residential / highway / railway / bank / commercial)
- [ ] **Property recovery** — gauge/progress bar showing recovery rate %

### 3.7 — Tab 6: Clusters & Key Findings

- [ ] **Scatter plot** — X = avg crime rate, Y = YoY growth. Dots colored by cluster, sized by total crimes. Hover shows state name
- [ ] **Cluster cards** — name, description, member states, key characteristics
- [ ] **Key findings** — 6–8 auto-generated insights from analytics (computed in Phase 2, passed as JSON)

### 3.8 — Polish

- [ ] Consistent color palette across all charts
- [ ] Number formatting (commas)
- [ ] Loading states / spinners
- [ ] Empty state handling ("No data available")
- [ ] Tablet-friendly layout

---

## Phase 4: Visualization Script

> `visualization.py` — reads analytics outputs and generates the final dashboard HTML.

- [ ] Read all 7 JSON files from `output/dashboard_data/`
- [ ] Read `india_states.geojson`
- [ ] Template the HTML — inject all data as JS variables into the HTML template
- [ ] Write self-contained `dashboard/index.html` (all CSS/JS/data inline)

---

## Phase 5: Documentation & Cleanup

- [ ] **README.md** — project title, description, architecture diagram, how to run, dataset descriptions, dashboard screenshot, tech stack
- [ ] **Inline comments** — section headers and brief comments in all scripts
- [ ] **Clean up orphan files** — remove `crime_hotspots_map.html`, `analysis.log`, `derby.log`, `metastore_db/`, `spark-warehouse/`
- [ ] **`.gitignore`** — add `metastore_db/`, `spark-warehouse/`, `derby.log`, `*.log`, `output/`

---

## Final Project Structure

```
hadoop-crime-project/
├── README.md
├── WORKFLOW.md
├── CLAUDE.md
├── data/
│   ├── 01_District_wise_crimes_committed_IPC_2001_2012.csv
│   ├── 01_District_wise_crimes_committed_IPC_2013.csv
│   ├── 01_District_wise_crimes_committed_IPC_2014.csv
│   ├── 10_Property_stolen_and_recovered.csv
│   ├── 17_Crime_by_place_of_occurrence_2001_2012.csv
│   ├── 17_Crime_by_place_of_occurrence_2013.csv
│   ├── 17_Crime_by_place_of_occurrence_2014.csv
│   ├── 30_Auto_theft.csv
│   ├── 31_Serious_fraud.csv
│   ├── 32_Murder_victim_age_sex.csv
│   ├── 33_CH_not_amounting_to_murder_victim_age_sex.csv
│   ├── 34_Use_of_firearms_in_murder_cases.csv
│   ├── 39_Specific_purpose_of_kidnapping_and_abduction.csv
│   ├── 42_Cases_under_crime_against_women.csv
│   ├── 42_District_wise_crimes_committed_against_women_2001_2012.csv
│   ├── 42_District_wise_crimes_committed_against_women_2013.csv
│   ├── 42_District_wise_crimes_committed_against_women_2014.csv
│   └── india_states.geojson
├── src/
│   ├── data_preparation.py
│   ├── analytics.py
│   └── visualization.py
├── output/
│   ├── district_master/          (Parquet)
│   ├── state_master/             (Parquet)
│   ├── analytics_results/        (Parquet)
│   └── dashboard_data/           (JSON)
│       ├── district_analysis.json
│       ├── clusters.json
│       ├── crime_profiles.json
│       ├── women_safety.json
│       ├── supplementary.json
│       ├── forecasts.json
│       └── national_trends.json
├── dashboard/
│   ├── index.html
│   └── india_states.geojson
└── .gitignore
```

---

## Data Flow

```
data/*.csv → HDFS /crime/input/
    → data_preparation.py
        → HDFS /crime/output/district_master (Parquet)
        → HDFS /crime/output/state_master (Parquet)
    → analytics.py
        → HDFS /crime/output/analytics_results/ (Parquet)
        → output/dashboard_data/*.json (7 files)
    → visualization.py
        → dashboard/index.html (self-contained interactive dashboard)
```

---

## Datasets Reference

| # | File | Granularity | Years | What It Provides |
|---|------|-------------|-------|------------------|
| 01 | IPC Crimes (×3 files) | District | 2001–2014 | 30+ crime types per district per year — **backbone** |
| 42 | Crimes Against Women (×3 files) | District | 2001–2014 | Rape, kidnapping, dowry deaths, cruelty — district level |
| 42 | Cases Against Women (disposal) | State | 2001–2014 | Conviction rates, pending cases — **justice pipeline** |
| 10 | Property Stolen & Recovered | State | 2001–2014 | ₹ value stolen vs recovered by crime type |
| 17 | Crime by Place of Occurrence (×3) | State | 2001–2014 | Where crimes happen: highways, railways, banks, homes |
| 30 | Auto Theft | State | 2001–2014 | Stolen vs recovered by vehicle type |
| 31 | Serious Fraud | State | 2001–2010 | Financial crime by loss bracket |
| 32 | Murder Victims (age/sex) | State | 2001–2010 | Victim demographics: male/female × age brackets |
| 33 | Culpable Homicide Victims | State | 2001–2010 | Same as above for non-murder homicide |
| 34 | Firearms in Murder | State | 2001–2010 | Licensed vs unlicensed arms usage |
| 39 | Kidnapping by Purpose | State | 2001–2010 | Why people are kidnapped: ransom, marriage, prostitution |
