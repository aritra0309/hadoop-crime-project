# Version 1.3 Development Plan — India Crime Intelligence Platform

> **Goal:** Transform the platform from a static 2001–2014 pipeline into a dynamic, extensible
> system where anyone can drop in new NCRB data and get updated dashboards, forecasts, and
> clustering — with zero code changes.

---

## Phase 1: Central Configuration Layer

### Problem
Settings are scattered across multiple Python files as hardcoded constants — HDFS paths in
`data_preparation.py` line 42–43, year ranges in multiple filter calls, forecast horizons in
`analytics.py`, KMeans parameters, Ridge alpha values. If someone wants to change anything,
they have to hunt through 1000+ lines of code.

### What Needs to Happen
Create a single `config.yaml` file at the project root that holds every configurable value
in the entire pipeline. Then create a small `config_loader.py` in `src/` that reads this YAML
and exposes the values to all other modules.

### What Goes in config.yaml
- **HDFS settings:** base URL (`hdfs://localhost:9000`), input path, output path, cleaned
  data path — currently hardcoded in both `data_preparation.py` (L42–43) and `analytics.py`
  (L41–43)
- **Year range:** currently hardcoded as `2001, 2014` in filter calls at `data_preparation.py`
  L192 and L264. Should be set to `auto` by default (detect from data) with option to override
- **Forecast settings:** horizon (how many years ahead to predict, currently hardcoded as
  2015–2020 in `analytics.py` L962), baseline year (hardcoded as 2001 at L963), polynomial
  degree, Ridge alpha values
- **KMeans settings:** k range to search (currently 2–8), random seed (currently 42),
  max iterations
- **File paths:** GeoJSON boundary file location, output dashboard directory
- **Spark settings:** app name, log level, memory allocation

### Why This Matters
Every subsequent phase depends on this. Once config exists, removing hardcoded values is
just "read from config instead." Without it, every fix is ad hoc.

### Files Affected
- **New:** `config.yaml`, `src/config_loader.py`
- **Modified:** `data_preparation.py`, `analytics.py`, `utils.py` (import and use config)

---

## Phase 2: Remove All Hardcoded Values

### Problem
There are 8 specific hardcoded values that prevent the pipeline from working with any data
outside 2001–2014. These are the exact bottlenecks.

### Change 1: Year Range Filters in Data Preparation
**Where:** `data_preparation.py` lines 192 and 264
**What it does now:** Filters data to only keep rows between 2001 and 2014, throwing away
anything outside that range.
**What it should do:** Either use the config value, or if set to `auto`, detect the min and
max year from the actual data. This way if someone adds 2015 data, it flows through
automatically.

### Change 2: Forecast Year Array
**Where:** `analytics.py` line 962
**What it does now:** Creates a fixed array `[2015, 2016, 2017, 2018, 2019, 2020]` as the
years to predict.
**What it should do:** Compute `max_year_in_data + 1` through `max_year_in_data + horizon`
where horizon comes from config (default 6 years).

### Change 3: Forecast Baseline Year
**Where:** `analytics.py` line 963
**What it does now:** Subtracts 2001.0 from all years to create relative features. Assumes
data always starts at 2001.
**What it should do:** Subtract `min_year_in_data` instead. This way the polynomial features
are always relative to whenever the data starts.

### Change 4: Hotspot Year Filter
**Where:** `analytics.py` line 248–249
**What it does now:** Filters severity data to `year == 2014` to find current hotspots.
**What it should do:** Use `max(year)` from the data — whatever the latest year is, that's
the hotspot year.

### Change 5: Rising Hotspot Window
**Where:** `analytics.py` line 256
**What it does now:** Filters year-over-year data to `2010 <= year <= 2014` to detect rising
crime trends.
**What it should do:** Use `max_year - 4` to `max_year` — always the last 5 years of
whatever data is available.

### Change 6: National Trend Comparison
**Where:** `analytics.py` lines 1022–1043
**What it does now:** Explicitly compares year 2001 vs 2014 for each state to compute
percentage change.
**What it should do:** Compare `min_year` vs `max_year` from the data.

### Change 7: HDFS Path Constants
**Where:** `data_preparation.py` L42–43, `analytics.py` L41–43
**What it does now:** Hardcodes `hdfs://localhost:9000/crime/input` etc.
**What it should do:** Read from config. This also helps when deploying to a real cluster
where the namenode URL is different.

### Change 8: Dataset File List
**Where:** `data_preparation.py` lines 269–276 and the entire processing section (L382–709)
**What it does now:** Every CSV filename is a hardcoded string. Each dataset has its own
dedicated processing function with the filename baked in.
**What it should do:** Read from a dataset registry (see Phase 3). The processing functions
can stay dataset-specific, but the filenames and paths should come from config.

### Files Affected
- **Modified:** `data_preparation.py`, `analytics.py`
- **Depends on:** Phase 1 (config layer)

---

## Phase 3: Smart Data Ingestion

### Problem
Adding new data currently requires editing Python code — you need to know which function
handles which CSV, add the filename, and potentially write new processing logic. A researcher
who just wants to add 2015 data shouldn't need to touch code.

### Idea 1: Dataset Registry File
Create a `datasets.yaml` file that lists every dataset the pipeline knows about. Each entry
has: the filename pattern (or glob), which processing function handles it, what crime
category it belongs to, and what columns to expect. When the pipeline starts, it reads this
registry and processes whatever files match.

New datasets that follow the same schema as existing ones (e.g., a 2015 version of the IPC
crimes file) get auto-detected if they match the filename pattern. Truly new dataset types
would require adding an entry to the registry and potentially a new processing function —
but that's a much smaller ask than editing the main pipeline code.

### Idea 2: Extensible State Mappings
Currently `state_mapping.py` has a Python dictionary with 16 mappings hardcoded. This should
become a `state_mappings.yaml` file that anyone can edit without touching Python. The loader
reads the YAML at startup. If a new state name variant appears in 2015+ data (e.g., "Ladakh"
which became a UT in 2019), users just add a line to the YAML.

### Idea 3: Schema Validation on Ingestion
When a new CSV is loaded, the pipeline should check: Does it have the expected columns? Are
the data types correct? Are there new state names not in the mapping? If validation fails,
it should warn the user with a clear message ("Found unknown state name 'Ladakh' in row 45
— please add it to state_mappings.yaml") rather than silently producing wrong results or
crashing.

### Idea 4: Data Directory Convention
Establish a convention: raw data goes in `data/raw/`, cleaned data in `data/cleaned/`. Users
drop new CSVs into `data/raw/` and run the pipeline. The pipeline scans the directory,
matches files against the registry, and processes them. No code changes needed.

### Files Affected
- **New:** `datasets.yaml`, `state_mappings.yaml`, `src/schema_validator.py`
- **Modified:** `data_preparation.py` (use registry instead of hardcoded names),
  `state_mapping.py` (load from YAML)
- **Depends on:** Phase 1

---

## Phase 4: Gap-Aware Forecasting

### Problem
The current forecasting assumes a clean, continuous block of data (2001–2014) and predicts
the next 6 years. But what if someone has actual data for 2001–2014 AND 2021? The system
should be smart enough to show:
- 2001–2014: actual data
- 2015–2020: predicted (gap fill)
- 2021: actual data
- 2022–2027: predicted (future forecast)

### How It Should Work

**Step 1: Detect what years have actual data.** Scan the cleaned dataset and build a set of
years that have real observations. For example: `{2001, 2002, ..., 2014, 2021}`.

**Step 2: Identify gaps.** Any year between `min_year` and `max_year` that's missing is a
gap. In the example above, 2015–2020 are gaps.

**Step 3: Train on ALL actual data.** Don't just train on the first continuous block. Use
every actual data point available. If you have 2001–2014 and 2021, train on all 15 points.
This gives the model a much longer baseline and the 2021 point helps anchor the forecast.

**Step 4: Generate predictions for gaps AND future.** Predict values for 2015–2020 (gap fill)
and 2022–2027 (future forecast). The forecast horizon comes from config.

**Step 5: Tag every data point.** Each year in the output gets a label: `"actual"` or
`"predicted"`. This is critical for the dashboard to render them differently (solid vs dashed
lines).

**Step 6: Confidence intervals.** For predicted years, compute confidence intervals based on
the model's cross-validation error. Points further from actual data get wider intervals.
This gives users a visual sense of uncertainty.

### Edge Cases to Handle
- **All years present, no gaps:** Just forecast the future. Normal behavior.
- **Only one year of data:** Can't do polynomial regression. Fall back to showing just the
  actual data with a warning.
- **Large gaps (>5 years):** Warn the user that gap-fill predictions may be unreliable.
- **Non-contiguous sparse data (e.g., 2001, 2005, 2010, 2021):** Train on what's available
  but flag low confidence.

### Files Affected
- **Modified:** `analytics.py` (forecast functions ~L900–1000)
- **Depends on:** Phase 1 and 2 (dynamic year detection)

---

## Phase 5: Temporal KMeans Clustering

### Problem
The current KMeans clusters states by their **average** crime profile across all years. This
throws away all temporal information. A state that improved dramatically over 14 years gets
the same cluster as one that stayed constant — as long as their averages match.

### Idea 1: Trajectory Clustering
Instead of clustering on averages, cluster on **how crime changed over time**. For each state,
compute the year-over-year percentage change for each crime category. This gives you a
trajectory vector. States with similar trajectories (e.g., "crime dropping steadily") end up
in the same cluster, even if their absolute levels are very different.

This answers a much more useful question: "Which states are on similar crime trends?" rather
than "Which states have similar crime levels?"

### Idea 2: Per-Year Clustering
Run KMeans separately for each year. This produces a cluster assignment per state per year.
Then you can track: Did Maharashtra move from Cluster 3 (high crime) in 2005 to Cluster 2
(moderate) in 2014? This shows improvement over time.

The dashboard could show an animated map where cluster colors shift year by year, or a
"transition matrix" showing how many states moved between clusters.

### Idea 3: Cluster Transition Tracking
Once you have per-year clusters, compute a transition matrix: for each pair of consecutive
years, how many states moved from cluster A to cluster B? This reveals systemic shifts —
e.g., "between 2010 and 2014, 5 states moved from high-crime to moderate-crime clusters."

### Idea 4: Better Cluster Labels
Currently the labeling checks one feature (`avg_ipc_crimes`) for the primary label and picks
one secondary qualifier. Improve this to:
- Use **all features** to generate a multi-dimensional profile label
- Allow **multiple tags** (e.g., "High Crime, High Women Crime, Low Recovery")
- Include the **trend direction** in the label (e.g., "High but Declining" vs "High and Rising")

### What This Adds to the Paper
This directly addresses the reviewer's concern about KMeans being shallow. You can now say:
"The platform performs both static profiling (average-based clustering) and temporal trajectory
analysis (trend-based clustering), enabling researchers to identify not just which states have
similar crime levels but which states are on similar crime trajectories."

### Files Affected
- **Modified:** `analytics.py` (clustering section ~L350–470)
- **New output:** trajectory cluster JSONs, transition matrices
- **Depends on:** Phase 2 (dynamic year detection)

---

## Phase 6: Dashboard Dynamic Updates

### Problem
The dashboard currently works with whatever JSON the analytics stage produces, but the year
selector and chart rendering assume a fixed set of years. If the data suddenly includes 2021,
the dashboard needs to handle it gracefully — including showing predicted vs actual differently.

### Idea 1: Dynamic Year Dropdown
The year dropdown in `dashboard/index.html` should be populated from the data itself. When
the dashboard loads, it reads the available years from the JSON and builds the dropdown. No
hardcoded year list.

### Idea 2: Actual vs Predicted Visual Distinction
On trend charts, actual data points should be solid dots with solid lines. Predicted data
points should be hollow dots with dashed lines. If there's a gap (e.g., 2015–2020 predicted,
2021 actual), the chart should clearly show the transition — maybe a vertical marker or
color change at the boundary.

### Idea 3: Gap Visualization on Maps
On the choropleth map, if the user selects a predicted year (from gap-fill), add a subtle
banner or indicator saying "Predicted — no actual data for this year." This prevents users
from mistaking predictions for real data.

### Idea 4: Cluster Animation
If per-year clustering is implemented (Phase 5), add a "Play" button that animates through
years, showing how cluster assignments change. This is visually compelling and immediately
communicates temporal patterns.

### Idea 5: Data Freshness Indicator
Show metadata on the dashboard: "Data covers 2001–2014 (actual), 2015–2020 (predicted),
2021 (actual), 2022–2027 (predicted). Generated on 2026-04-18, pipeline v1.3.0." This gives
users confidence in what they're looking at.

### Files Affected
- **Modified:** `dashboard/index.html` (JavaScript rendering logic)
- **Depends on:** Phase 4 (gap-aware output with actual/predicted tags)

---

## Phase 7: CLI Arguments & Output Metadata

### Problem
Currently the pipeline runs with no arguments — everything is baked in. Users should be able
to customize behavior without editing config files.

### Idea 1: Command-Line Arguments
Add CLI args to the spark-submit commands:
- `--config path/to/config.yaml` — use a custom config
- `--years 2001-2021` — override year range
- `--forecast-horizon 10` — predict 10 years ahead instead of default 6
- `--k-range 2-12` — search wider range for optimal clusters
- `--output-dir /custom/path` — write output somewhere specific

These override config.yaml values, which override defaults. Three-tier precedence:
CLI > config.yaml > built-in defaults.

### Idea 2: Output Metadata
Every JSON output file should include a metadata block:
- Pipeline version
- Data years (actual and predicted)
- When it was generated
- Config values used (so results are fully reproducible)
- Model performance metrics (silhouette score, RMSE, etc.)

This means anyone who receives a dashboard can trace exactly how it was produced.

### Idea 3: Validation Warnings
When the pipeline runs, it should print clear warnings for:
- Unknown state names not in the mapping
- CSV files that don't match any registry pattern
- Years with suspiciously few data points
- Columns with >50% missing values

These don't stop the pipeline — they just inform the user so they can fix data quality issues.

### Files Affected
- **Modified:** `data_preparation.py`, `analytics.py` (add argparse)
- **Modified:** all output-writing functions (add metadata blocks)
- **Depends on:** Phase 1

---

  ---
                                                                                                                    
  ## Phase 10: Reproducibility Gaps                                                                                 
                                                                                                                    
  These are issues that the v1.3 code changes alone will NOT fix. They need separate action.                        
                                                                                                                    
  ### 10.1 Data Source Permanence                                                                                   
                                                                                                                    
  NCRB does not maintain stable download URLs. Historical PDF and CSV reports get moved, renamed,                   
  or removed from ncrb.gov.in without notice. If a researcher tries to reproduce this work in 2028,                 
  the raw data may no longer be downloadable from the same location.                                                
                                                                                                                    
  This means the code being reproducible is not enough — the DATA must also be archived independently               
  of the source website.                                                                                            
                                                                                                                    
  ### 10.2 Loose Dependency Versions                                                                                
                                                                                                                    
  Currently requirements.txt likely uses loose version specifiers like pyspark>=3.0 or                              
  scikit-learn>=1.0. This means two people installing dependencies a year apart will get different                  
  library versions. Different versions can introduce subtle behavioral changes — especially in                      
  numerical libraries.                                                                                              
                                                                                                                    
  This applies to the Dockerfile as well. If the base image uses a generic tag like python:3.9                      
  instead of python:3.9.18-slim-bookworm, the underlying OS packages can drift over time.                           
                                                                                                                    
  ### 10.3 Cross-Version Numerical Drift                                                                            
                                                                                                                    
  Even with the same algorithm and same data, different versions of scikit-learn can produce slightly               
  different Ridge regression coefficients due to internal solver changes. KMeans is protected by                    
  seed=42, but Ridge has no such guarantee across library versions.                                                 
                                                                                                                    
  This means two researchers running the same pipeline with different scikit-learn versions might get               
  slightly different forecast numbers. The clusters will be identical, but the regression outputs                   
  could vary at the decimal level.                                                                                  
                                                                                                                    
  ### 10.4 State Reorganizations Not in Mappings                                                                    
                                                                                                                    
  India reorganized states in 2019 — Jammu & Kashmir was split into J&K UT and Ladakh UT. The                       
  current state_mapping.py has 16 entries covering historical name variations, but does NOT account                 
  for post-2019 reorganizations.                                                                                    
                                                                                                                    
  If someone adds 2020+ NCRB data, new state/UT names like "Ladakh" will appear and won't match                     
  any canonical name. The pipeline won't crash, but those rows will be silently excluded from                       
  analysis — which is worse than crashing because the user won't know data is missing.                              
                                                                                                                    
  ### 10.5 No Integration Tests for Output Validation                                                               
                                                                                                                    
  The project has 51 unit tests that verify code quality — functions run without errors, types are                  
  correct, etc. But there is no integration test that says: "Given exactly THIS input CSV, the                      
  pipeline MUST produce exactly THESE numbers."                                                                     
                                                                                                                    
  Without golden-output tests, there is no way to detect if a dependency upgrade or code refactor                   
  subtly changed the analytical results. A researcher cannot verify that their run matches the                      
  published results.                                                                                                
                                                                                                                    
  ---                                                                                                               
                                                                                                                    
  ## Phase 11: Dependency and Data Archival                                                                         
                                                                                                                    
  These are the fixes for the gaps identified in Phase 10.                                                          
                                                                                                                    
  ### 11.1 Pin All Dependency Versions                                                                              
                                                                                                                    
  Go through requirements.txt and replace every >= with ==. For example:                                            
  - pyspark>=3.0 becomes pyspark==3.5.1 (or whatever version you are currently using)                               
  - scikit-learn>=1.0 becomes scikit-learn==1.4.2                                                                   
  - numpy, pandas, folium — all pinned to exact versions                                                            
                                                                                                                    
  Run pip freeze to get your current exact versions and use those.                                                  
                                                                                                                    
  In the Dockerfile, pin the base image to a specific digest or full tag. For example use                           
  python:3.9.18-slim-bookworm instead of python:3.9. Also pin the Spark and Hadoop versions                         
  in any download URLs inside the Dockerfile.                                                                       
                                                                                                                    
  ### 11.2 Archive Raw Data on Zenodo                                                                               
                                                                                                                    
  The project already has a Zenodo DOI (C3 in the paper metadata). But verify that the Zenodo                       
  archive includes the raw CSV files in the data/ directory — not just the source code.                             
                                                                                                                    
  If the Zenodo archive only has code, create a new Zenodo upload or update the existing one to                     
  include all 17 CSV files and the GeoJSON file. This ensures the data survives even if NCRB                        
  removes it from their website.                                                                                    
                                                                                                                    
  ### 11.3 Add Golden-Output Integration Tests                                                                      
                                                                                                                    
  Create a small test dataset — maybe 3 states, 3 years, 2 crime categories — with known values.                    
  Run the full pipeline on this small dataset and record the exact outputs: cluster assignments,                    
  forecast values, safety index scores, composition percentages.                                                    
                                                                                                                    
  Save these expected outputs as JSON files in the tests/ directory. The integration test runs the                  
  pipeline on the small dataset and compares the output to the saved expected values. If they                       
  differ, the test fails.                                                                                           
                                                                                                                    
  This catches any silent numerical drift from dependency upgrades.                                                 
                                                                                                                    
  ### 11.4 Document State Reorganizations                                                                           
                                                                                                                    
  Create a section in the state_mappings.yaml (from Phase 3) that documents known state                             
  reorganizations with dates:                                                                                       
  - 2000: Jharkhand split from Bihar, Chhattisgarh from MP, Uttarakhand from UP                                     
  - 2014: Telangana split from Andhra Pradesh                                                                       
  - 2019: J&K split into J&K UT and Ladakh UT                                                                       
                                                                                                                    
  Include guidance on how to handle these: when a state splits, should historical data be attributed                
  to the parent state? Should the new state appear only from its creation year? Document the                        
  decision so future users know the convention.                                                                     
                                                                                                                    
  ### 11.5 Add a REPRODUCIBILITY.md File                                                                            
                                                                                                                    
  Create a standalone file in the project root that explains:                                                       
  - Exact steps to reproduce published results                                                                      
  - Which dependency versions were used                                                                             
  - Where to get the raw data if NCRB links are dead (Zenodo)                                                       
  - Known numerical precision limitations                                                                           
  - How to run the integration tests to verify output matches                                                       
                                                                                                                    
  ---                                                                                                               
                                                                                                                    
  ## Phase 12: Paper Language for Reproducibility                                                                   
                                                                                                                    
  After implementing v1.3, update the SoftwareX paper with the following.                                           
                                                                                                                    
  ### 12.1 Reproducibility Claim (add to Section 4 — Impact)                                                        
                                                                                                                    
  In the "Reproducibility and transparency" paragraph, strengthen the language to:                                  
                                                                                                                    
  "All code, data, and configuration are publicly available under the MIT license with a permanent                  
  Zenodo archive that includes both source code and raw NCRB datasets. The platform externalizes                    
  all configuration through a central YAML file, auto-discovers input datasets, and dynamically                     
  generates all analytical findings — ensuring that results are fully determined by the input data                  
  rather than hardcoded values. The CI/CD pipeline includes both unit tests and golden-output                       
  integration tests that verify analytical results remain consistent across environments."                          
                                                                                                                    
  ### 12.2 Limitations (add as new paragraph at end of Section 5 — Conclusions)                                     
                                                                                                                    
  Add before the future work list:                                                                                  
                                                                                                                    
  "The platform's reproducibility is bounded by two external factors: (i) NCRB data availability,                   
  as historical reports may be removed from the official website without notice, mitigated by our                   
  Zenodo data archive; and (ii) minor numerical variations in regression coefficients across                        
  scikit-learn versions, mitigated by pinned dependencies and integration tests that verify output                  
  consistency."                                                                                                     
                                                                                                                    
  ### 12.3 Updated Metadata Table                                                                                   
                                                                                                                    
  Update C1 from v1.2.0 to v1.3.0 and update C7 to include pinned versions instead of >= symbols.                   
                                                                                                                    
  ---                                                                                                               
                                                                    
## Implementation Order & Dependencies

```
Phase 1 (Config) ─────────┬──> Phase 2 (Remove Hardcoded) ──> Phase 4 (Gap Forecasting)
                           │                                         │
                           ├──> Phase 3 (Smart Ingestion)            ├──> Phase 6 (Dashboard)
                           │                                         │
                           └──> Phase 7 (CLI & Metadata)       Phase 5 (Temporal KMeans)
                                                                     │
                                                               Phase 8 (Paper Updates)
```

**Phase 1** is the foundation — everything else depends on it.
**Phases 2, 3, 7** can be done in parallel after Phase 1.
**Phase 4** needs Phase 2 done first.
**Phase 5** needs Phase 2 done first.
**Phase 6** needs Phase 4 done first.
**Phase 8** is last — update the paper after everything works.

---

## Estimated Effort

| Phase | Effort | Notes |
|-------|--------|-------|
| 1. Config Layer | 2–3 hours | Straightforward but foundational |
| 2. Remove Hardcoded | 3–4 hours | 8 changes, need to test each |
| 3. Smart Ingestion | 4–5 hours | Registry design + validation logic |
| 4. Gap Forecasting | 4–5 hours | Algorithm is clear but edge cases need care |
| 5. Temporal KMeans | 5–6 hours | New analytical capability |
| 6. Dashboard Updates | 4–5 hours | JavaScript/HTML changes + testing |
| 7. CLI & Metadata | 2–3 hours | Argparse + JSON metadata |
| 8. Paper Updates | 2–3 hours | Text changes only |
| **Total** | **~26–34 hours** | |

---

## What NOT to Change

- **KMeans seed=42** — standard practice, keep it
- **k_max = min(8, ...)** — already adaptive, fine as default
- **TimeSeriesSplit(n_splits=3)** — sensible for short series
- **Ridge alpha values** — reasonable defaults, just make them configurable
- **Test structure** — keep existing 51 tests, add new ones for new features
- **Docker setup** — works well, just update config mount
