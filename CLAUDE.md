# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PySpark-based big data pipeline for analyzing India crime statistics (2001–2014) using Hadoop/HDFS. The pipeline ingests multiple CSV datasets from NCRB (National Crime Records Bureau), cleans and joins them into a master dataset, performs analytics (clustering, forecasting, composition analysis), and generates interactive HTML visualizations.

## Prerequisites

- **Java 11** (required by Spark)
- **Apache Spark 3.5.3** with PySpark
- **Hadoop HDFS** running locally on `hdfs://localhost:9000`
- Python libraries: `pyspark`, `pandas`, `numpy`, `scikit-learn`, `folium`

## HDFS Setup

Before running scripts, data must be loaded into HDFS:

```bash
# Create HDFS directories
hdfs dfs -mkdir -p /crime/input
hdfs dfs -mkdir -p /crime/output

# Upload all CSV files from data/ to HDFS
hdfs dfs -put data/*.csv /crime/input/
```

## Running the Pipeline

Scripts must be run **in order** — each depends on the outputs of the previous:

```bash
# Step 1: Clean and prepare all 8 datasets, join into master table
spark-submit src/data_preparation.py

# Step 2: Run analytics (clustering, forecasting, composition, women safety index)
spark-submit src/analytics.py

# Step 3: Generate interactive visualizations
spark-submit src/visualization.py
```

## Architecture

### Three-stage pipeline

1. **`data_preparation.py`** — Ingests 8 raw CSV datasets from HDFS, normalizes column names across inconsistent schemas (2001-2012 vs 2013 vs 2014 files use different column headers), standardizes Indian state names (handles ~15 known aliases via `STATE_NAME_MAP`), aggregates district-level data to state-level, and left-joins everything into a master dataset. Outputs to HDFS as CSV and Parquet.

2. **`analytics.py`** — Reads the master dataset from HDFS and performs:
   - **KMeans clustering** (Spark MLlib) with silhouette-based k selection across 12 features
   - **Crime composition analysis** (percentage breakdown by crime type per state)
   - **Women Safety Index** (inverse normalized score from crimes-against-women data)
   - **Property recovery analysis**
   - **State-wise forecasting** (2015–2020) using model selection between Ridge polynomial regression and GradientBoostingRegressor with TimeSeriesSplit CV

3. **`visualization.py`** — Reads time series from HDFS, generates:
   - Interactive choropleth heatmap (Folium + Leaflet) with year selector
   - Per-state crime trend chart (Chart.js)
   - Outputs to `output/` as standalone HTML files

### Data flow

```
data/*.csv → HDFS /crime/input/
    → data_preparation.py → HDFS /crime/output/ (master_crime_data, cleaned_ipc_crime_data, etc.)
    → analytics.py → HDFS /crime/output/ (clustered_crime_data, state_crime_time_series, model_report, etc.)
    → visualization.py → output/*.html
```

### Key HDFS paths

| Path | Format | Written by |
|------|--------|------------|
| `/crime/input/` | CSV | Manual upload |
| `/crime/output/master_crime_data` | CSV | data_preparation |
| `/crime/output/cleaned_ipc_crime_data` | CSV | data_preparation |
| `/crime/output/crimes_against_women` | CSV | data_preparation |
| `/crime/output/property_stolen_recovered` | CSV | data_preparation |
| `/crime/output/clustered_crime_data` | Parquet | analytics |
| `/crime/output/state_crime_time_series` | Parquet | analytics |
| `/crime/output/model_report` | Parquet | analytics |
| `/crime/output/women_safety_index` | Parquet | analytics |
| `/crime/output/crime_composition` | Parquet | analytics |
| `/crime/output/property_recovery_analysis` | Parquet | analytics |
| `/crime/output/cluster_labels` | Parquet | analytics |

### Datasets (in `data/`)

- `01_*` — District-wise IPC crimes (3 files: 2001-2012, 2013, 2014)
- `42_*` — Crimes against women (district-wise + cases summary)
- `10_*` — Property stolen and recovered
- `30_*` — Auto theft
- `31_*` — Serious fraud
- `32_*` — Murder victim demographics (age/sex)
- `33_*` — Culpable homicide victim demographics
- `34_*` — Firearms in murder
- `39_*` — Kidnapping by purpose
- `india_states.geojson` — GeoJSON for choropleth map (state boundaries, uses `NAME_1` property)

## Important Patterns

- **State name standardization**: `STATE_NAME_MAP` in `data_preparation.py` maps ~15 variant spellings to canonical names. When adding new data sources, check for state name inconsistencies and add mappings there.
- **Schema normalization**: Each dataset era (2001-2012, 2013, 2014) uses different column headers. The `normalize_*_df()` functions handle renaming. The `safe_select()` helper fills missing columns with zeros.
- **Aggregate rows filtering**: Raw data contains "TOTAL (ALL-INDIA)" type rows that must be filtered out — handled by `standardize_state()`.
- **Spark config**: All scripts use `spark.sql.shuffle.partitions=4` (appropriate for this dataset size). Outputs use `.coalesce(1)` for single-file output.
