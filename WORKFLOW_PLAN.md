# Hadoop Crime Project — Improvement Workflow Plan

> Generated: 2026-04-18 | Reference doc for all planned changes

---

## Phase 1: Configuration Layer (Do First — Everything Depends on This)

### 1.1 Create `config.yaml`
- **File:** `src/config.yaml` (new)
- **What:** Single source of truth for all settings
- **Contents:**
  ```yaml
  hdfs:
    base_url: "hdfs://localhost:9000"
    input_path: "/crime/input"
    output_path: "/crime/output"

  pipeline:
    forecast_horizon: 6          # predict N years ahead
    hotspot_window: 5            # last N years for rising hotspot
    kmeans:
      k_max: 8
      seed: 42
    ridge:
      alphas: [10.0, 50.0]
      poly_degrees: [2, 3]
    time_series:
      cv_splits: 3

  data:
    auto_discover: true          # scan data/ dir for CSVs
    file_patterns:               # fallback if auto_discover fails
      - "*.csv"
    state_mapping_file: "state_mapping.yaml"  # external mappings
  ```

### 1.2 Create `src/config_loader.py` (new)
- Reads `config.yaml` + allows env var overrides
- Every other module imports from here instead of hardcoding values
- **Priority:** 🔴 Critical — do this before touching any other file

---

## Phase 2: Remove Hardcoded Values (The Big Cleanup)

### 2.1 `data_preparation.py` — Year Filters
- **Lines:** ~L192, ~L264
- **Current:** `df.filter(F.col("year").between(2001, 2014))`
- **Change to:**
  ```python
  min_year, max_year = df.agg(F.min("year"), F.max("year")).collect()[0]
  df.filter(F.col("year").between(min_year, max_year))
  ```
- Or just **remove the filter entirely** — let all years through

### 2.2 `data_preparation.py` — HDFS Paths
- **Lines:** ~L42-43
- **Current:** `HDFS_INPUT = "hdfs://localhost:9000/crime/input"`
- **Change to:** `HDFS_INPUT = config["hdfs"]["base_url"] + config["hdfs"]["input_path"]`

### 2.3 `data_preparation.py` — Hardcoded CSV Filenames
- **Lines:** ~L269-276, ~L382-709
- **Current:** 17 filenames listed as strings
- **Change to:** Auto-discover from `data/` directory OR read from `config.yaml`
- **Logic:**
  ```python
  if config["data"]["auto_discover"]:
      csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
  else:
      csv_files = config["data"]["file_patterns"]
  ```

### 2.4 `analytics.py` — HDFS Paths
- **Lines:** ~L41-43
- **Same fix as 2.2** — read from config

### 2.5 `analytics.py` — Forecast Years
- **Lines:** ~L962-963
- **Current:** `future_years = np.arange(2015, 2021)`
- **Change to:**
  ```python
  max_actual = int(df.agg(F.max("year")).collect()[0][0])
  horizon = config["pipeline"]["forecast_horizon"]
  future_years = np.arange(max_actual + 1, max_actual + 1 + horizon)
  ```

### 2.6 `analytics.py` — Hotspot Year
- **Lines:** ~L248-249
- **Current:** `severity_df.filter(F.col("year") == 2014)`
- **Change to:** `severity_df.filter(F.col("year") == max_year)`

### 2.7 `analytics.py` — Rising Hotspot Window
- **Lines:** ~L256
- **Current:** `.filter((F.col("year") >= 2010) & (F.col("year") <= 2014))`
- **Change to:**
  ```python
  window = config["pipeline"]["hotspot_window"]
  .filter((F.col("year") >= max_year - window + 1) & (F.col("year") <= max_year))
  ```

### 2.8 `analytics.py` — National Trend Comparison
- **Lines:** ~L1022-1043
- **Current:** `if 2001 in map_year and 2014 in map_year`
- **Change to:** `if min_year in map_year and max_year in map_year`

---

## Phase 3: Smart Data Ingestion (The Big Feature)

### 3.1 Auto-detect and validate new CSVs
- **New file:** `src/data_validator.py`
- **What it does:**
  1. Scans `data/` for all CSVs
  2. Checks each against expected NCRB schemas (column patterns)
  3. Warns if unexpected columns found
  4. Reports which years & categories are present
  5. Returns a manifest: `{filename, category, year_range, status}`

### 3.2 Make state mapping extensible
- **Current:** `state_mapping.py` — Python dict, requires code edit
- **Change:** Also support `state_mapping.yaml` as external file
  ```yaml
  # Users add new mappings here without touching Python code
  "Telangana TS": "Telangana"
  "Ladakh UT": "Ladakh"
  ```
- `state_mapping.py` loads from YAML first, falls back to built-in dict

### 3.3 Ingest new year data alongside existing
- When user drops in `crimes_2015.csv`, `crimes_2021.csv`:
  1. Auto-detected by 3.1
  2. State names normalized by 3.2
  3. Merged with existing cleaned data
  4. Year range auto-expands (Phase 2 changes handle this)

---

## Phase 4: Gap-Aware Forecasting (Your Vision)

### 4.1 Tag data points as actual vs predicted
- **Already partially exists** — analytics outputs have this
- **Enhance:** Add explicit `"source": "actual"` or `"source": "predicted"` to every data point in output JSON

### 4.2 Handle year gaps intelligently
- **Scenario:** User has 2001-2014 actual + 2021 actual
- **Pipeline should:**
  1. Detect the gap (2015-2020 missing)
  2. Train model on ALL actual years (2001-2014 + 2021)
  3. Fill gap years (2015-2020) as `"predicted"`
  4. Forecast beyond last actual (2022-2026) as `"predicted"`
  5. Output timeline: `actual → predicted → actual → predicted`

### 4.3 Forecast output format
```json
{
  "state": "Maharashtra",
  "timeline": [
    {"year": 2013, "value": 245678, "source": "actual"},
    {"year": 2014, "value": 251234, "source": "actual"},
    {"year": 2015, "value": 258100, "source": "predicted"},
    ...
    {"year": 2021, "value": 289456, "source": "actual"},
    {"year": 2022, "value": 295000, "source": "predicted"},
    {"year": 2023, "value": 301200, "source": "predicted"}
  ]
}
```

---

## Phase 5: Dashboard Dynamic Updates

### 5.1 Year dropdown reads from data
- **File:** `dashboard/index.html`
- **Current:** Likely has hardcoded year options or range
- **Change:** Read available years from the JSON data files
  ```javascript
  const years = [...new Set(data.map(d => d.year))].sort();
  // populate dropdown dynamically
  ```

### 5.2 Visual distinction for predicted vs actual
- Actual years: **solid line / solid fill**
- Predicted years: **dashed line / hatched fill**
- Gap years: **dotted line** with lighter color
- This makes the actual/predicted/actual/predicted pattern visually clear

### 5.3 Auto-refresh on new data
- Dashboard reads from `output/` JSON files
- When pipeline re-runs with new data, dashboard auto-reflects changes
- No code changes needed in dashboard — just re-run pipeline

---

## Phase 6: Metadata & Documentation

### 6.1 Output metadata
- Add to every output JSON:
  ```json
  {
    "meta": {
      "pipeline_version": "1.3.0",
      "data_years": [2001, 2002, ..., 2021],
      "actual_years": [2001, ..., 2014, 2021],
      "predicted_years": [2015, ..., 2020, 2022, ..., 2026],
      "generated_at": "2026-04-18T17:50:00Z",
      "config": { ... }
    }
  }
  ```

### 6.2 CLI arguments
- Add argparse to main scripts:
  ```bash
  spark-submit analytics.py --config config.yaml --years 2001-2021 --forecast-horizon 5
  ```
- CLI args override config.yaml, which overrides defaults

### 6.3 Update README
- Document the new config system
- Add "Adding New Data" section with step-by-step
- Add "Configuration Reference" for all config.yaml options

### 6.4 Update paper
- After implementing, update the paper's claims:
  - "16 state mappings" → actual new count if more added
  - "17 datasets" → "N datasets" or mention extensibility
  - Version bump to 1.3.0

---

## Execution Order (Dependency Graph)

```
Phase 1 (config.yaml + loader)
    ↓
Phase 2 (remove hardcoded values) — depends on Phase 1
    ↓
Phase 3 (auto-ingestion) — depends on Phase 2
    ↓
Phase 4 (gap-aware forecasting) — depends on Phase 2 + 3
    ↓
Phase 5 (dashboard updates) — depends on Phase 4
    ↓
Phase 6 (metadata + docs) — depends on all above
```

**Estimated effort:**
- Phase 1: ~1 hour
- Phase 2: ~2-3 hours (most tedious, lots of find-replace)
- Phase 3: ~2 hours
- Phase 4: ~3-4 hours (most complex logic)
- Phase 5: ~1-2 hours
- Phase 6: ~1 hour

**Total: ~10-13 hours of work**

---

## Things NOT Worth Changing

| Item | Why it's fine |
|------|--------------|
| `seed=42` in KMeans | Standard reproducibility practice |
| `k_max = min(8, ...)` | Already adaptive to data size |
| `TimeSeriesSplit(n_splits=3)` | Sensible for short time series |
| `Ridge(alpha=10.0)` | Reasonable default, model selection handles it |
| Docker setup | Already clean and working |
| Test structure | 51 tests, well organized |

