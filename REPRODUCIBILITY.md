# Reproducibility Guide

## India Crime Intelligence Platform — Exact Steps to Reproduce Published Results

### 1. Environment Setup

**Python version:** 3.11.9 (pinned via Docker base image `python:3.11.9-slim-bookworm`)

**Pinned dependencies** (`requirements.txt`):
```
pyspark==3.5.1
pandas==2.2.2
numpy==1.26.4
folium==0.16.0
branca==0.7.1
scikit-learn==1.4.2
PyYAML==6.0.1
pytest==8.1.1
```

All versions are exact-pinned (`==`) to ensure deterministic installs.

**Java:** OpenJDK 21 (required by PySpark/Spark)

### 2. Raw Data

The raw NCRB crime datasets are available from:

- **Zenodo:** https://zenodo.org/records/crime-india-ncrb (search for "India NCRB Crime Data 2001-2014")
- **data.gov.in:** https://data.gov.in — search for "Crime in India" datasets

Place all CSV files in `data/` at the project root before running the pipeline.

Expected files:
- District-level crime statistics (2001–2014)
- State-level crime statistics (2001–2014)

### 3. Running the Full Pipeline

#### Option A: Docker (recommended for exact reproducibility)

```bash
docker build -t crime-platform .
docker run --rm -v $(pwd)/output:/app/output crime-platform
```

This uses the pinned base image and dependencies, ensuring byte-identical environments.

#### Option B: Local with Hadoop/HDFS

```bash
pip install -r requirements.txt
# Ensure HDFS is running and data is loaded
bash scripts/run_pipeline.sh
```

#### Option C: Local Spark (no HDFS)

```bash
pip install -r requirements.txt
bash scripts/run_pipeline_local.sh
```

### 4. Configuration

The pipeline reads `config.yaml` at the project root. Key reproducibility-relevant settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `forecast.horizon` | 5 | Years to forecast ahead |
| `kmeans.k_min` | 2 | Minimum clusters to evaluate |
| `kmeans.k_max` | 8 | Maximum clusters to evaluate |
| `kmeans.seed` | 42 | Random seed for KMeans |
| `year_range.mode` | `auto` | `auto` or `fixed` |

Override via CLI:
```bash
python -m src.analytics --forecast-horizon 5 --k-range 2-8 --config config.yaml
```

### 5. Running Tests

#### Unit tests
```bash
pytest tests/ -v
```

#### Golden-output integration tests
```bash
pytest tests/test_integration_golden.py -v
```

These tests verify:
- Forecast structure, non-negativity, and determinism
- Cluster assignment completeness and label validity
- National trend aggregation correctness
- Single-year edge case handling

#### All tests with coverage
```bash
pytest tests/ -v --tb=short
```

### 6. Expected Outputs

After a successful run, `output/dashboard_data/` will contain:

| File | Description |
|------|-------------|
| `national_trends.json` | Yearly crime totals, fastest growing/declining categories |
| `clusters.json` | KMeans state clustering with silhouette scores |
| `trajectory_clusters.json` | States clustered by crime trajectory over time |
| `yearly_clusters.json` | Per-year clustering with transition tracking |
| `forecasts.json` | Ridge regression forecasts with confidence intervals |
| `crime_profiles.json` | Crime composition breakdown per state |
| `women_safety.json` | Women safety index and justice pipeline metrics |
| `district_analysis.json` | District-level severity, hotspots, YoY growth |
| `supplementary.json` | Property recovery, firearms, kidnapping motives, etc. |

Each JSON file includes a `_pipeline_metadata` block with the pipeline version,
generation timestamp, and configuration snapshot used.

### 7. Known Numerical Precision Limitations

1. **Floating-point rounding:** All output floats are rounded to 2 decimal places via `to_native()`. Intermediate computations use full IEEE 754 double precision, so results may differ by ±0.01 across platforms.

2. **KMeans non-determinism across Spark versions:** While the seed is fixed (`kmeans.seed=42`), Spark's KMeans implementation may produce slightly different cluster assignments across minor Spark versions due to internal parallelism and floating-point accumulation order. Cluster *labels* (High/Moderate/Low) are derived from relative ordering, so they are more stable than raw cluster IDs.

3. **Forecast confidence intervals:** The CI formula `1.96 × residual_std × (1 + 0.1 × distance)` grows linearly with distance from observed data. For states with large year-gaps (>5 years), gap-fill values carry wider uncertainty.

4. **scikit-learn Ridge regression:** Results are deterministic for a given version but may shift slightly between scikit-learn minor versions due to solver implementation changes. Pin to `scikit-learn==1.4.2` for exact reproduction.

5. **Cross-validation splits:** `TimeSeriesSplit(n_splits=3)` is deterministic given sorted input. Model selection (poly2 vs poly3) depends on these splits, so changing the number of splits or input ordering will change which model is selected per state.

6. **MinMax scaling:** The women safety index and severity scores use per-year MinMax normalization. Adding or removing states/districts from the input data will shift all normalized values.

### 8. Verifying Reproducibility

To confirm your results match the published outputs:

```bash
# Run the pipeline
bash scripts/run_pipeline_local.sh

# Run golden-output tests
pytest tests/test_integration_golden.py -v

# Compare JSON structure (ignoring timestamps)
python -c "
import json, sys
for f in ['clusters.json', 'forecasts.json', 'national_trends.json']:
    with open(f'output/dashboard_data/{f}') as fh:
        data = json.load(fh)
    print(f'{f}: {len(str(data))} chars, keys={sorted(data.keys())}')
"
```

If cluster assignments differ by 1–2 states, this is within expected KMeans
precision tolerance (see §7.2 above). Forecast values should match to ±0.01.
