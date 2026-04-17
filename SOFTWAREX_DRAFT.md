# India Crime Intelligence Platform: A PySpark Big Data Pipeline for Analyzing National Crime Statistics

**Authors:** Aritra Sarkar¹ (aritra.sarkar2022@vitstudent.ac.in), Shaheen Ali¹ (shaheen.ali2022@vitstudent.ac.in), Varshini S¹ (varshini.s2022@vitstudent.ac.in)

¹ School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, Tamil Nadu, India

---

## Abstract

The India Crime Intelligence Platform is an open-source, PySpark-based big data pipeline for ingesting, cleaning, analyzing, and visualizing 14 years (2001–2014) of crime statistics published by India's National Crime Records Bureau (NCRB). The software addresses the challenge of harmonizing 18 heterogeneous CSV datasets with inconsistent schemas, variant state naming conventions, and mixed granularity levels into a unified analytical framework. The pipeline implements a three-stage architecture—data preparation, analytics, and visualization—leveraging Apache Spark on Hadoop HDFS for scalable processing. It performs KMeans clustering, polynomial Ridge regression forecasting, crime composition analysis, and women safety indexing, producing interactive choropleth maps and trend charts as standalone HTML dashboards. The platform enables researchers, policymakers, and analysts to derive actionable insights from India's crime data without requiring big data expertise.

**Keywords:** crime analytics, PySpark, Hadoop, big data pipeline, NCRB, data visualization

---

## Metadata

| Nr | Code metadata description | Metadata |
|----|---------------------------|----------|
| C1 | Current code version | v1.0.0 |
| C2 | Permanent link to code/repository used for this code version | https://github.com/aritraSarkar03/hadoop-crime-project |
| C3 | Permanent link to reproducible capsule | https://doi.org/10.5281/zenodo.19631326 |
| C4 | Legal code license | MIT License |
| C5 | Code versioning system used | Git |
| C6 | Software code languages, tools, and services used | Python 3.8+, PySpark, Apache Hadoop HDFS |
| C7 | Compilation requirements, operating environments, & dependencies | pyspark ≥ 3.0, pandas, numpy, scikit-learn, folium, branca; Hadoop 3.x with HDFS; Linux/macOS/WSL |
| C8 | If available, link to developer documentation/manual | https://github.com/aritraSarkar03/hadoop-crime-project/blob/main/README.md |
| C9 | Support email for questions | aritra.sarkar2022@vitstudent.ac.in; GitHub Issues: https://github.com/aritraSarkar03/hadoop-crime-project/issues |

---

## 1. Motivation and Significance

India's National Crime Records Bureau (NCRB) publishes annual compilations of crime statistics covering all states and union territories, representing one of the largest publicly available crime datasets in the developing world [1]. These datasets span multiple crime categories—Indian Penal Code (IPC) offenses, crimes against women, juvenile crimes, economic offenses, cybercrimes, and more—across 36 states and union territories over multiple decades. However, working with this data presents three significant challenges that existing tools fail to address collectively.

**First, data harmonization.** The 18 raw CSV files published by NCRB exhibit inconsistent schemas: column names vary across years, state and union territory names are spelled differently (e.g., "ANDAMAN & NICOBAR" vs "A&N ISLANDS" vs "ANDAMAN & NICOBAR ISLANDS"), missing values are encoded heterogeneously, and granularity levels differ between datasets. Researchers currently spend substantial time on manual preprocessing before any analysis can begin. Our platform automates this entirely through a configurable state name mapping engine (`state_mapping.py`) that resolves 60+ naming variants to canonical forms, along with automated schema normalization and null handling.

**Second, scalability.** As NCRB datasets grow and researchers combine them with auxiliary sources (census data, economic indicators, geographic boundaries), the volume exceeds what pandas-based workflows handle efficiently. By building on Apache Spark with Hadoop HDFS as the storage layer, our pipeline processes the full 18-dataset corpus in a distributed manner, and the architecture scales naturally to larger datasets without code changes.

**Third, accessibility.** Existing crime analysis tools either require proprietary software (SPSS, SAS), demand deep programming expertise, or produce static outputs. Our platform generates self-contained interactive HTML dashboards with choropleth maps, trend charts, and clustering visualizations that can be opened in any browser and shared without software installation. This lowers the barrier for policymakers, journalists, and students who need insights but lack big data expertise.

No existing open-source tool combines all three capabilities—automated NCRB data harmonization, Spark-based scalable analytics, and interactive visualization—in a single reproducible pipeline. The India Crime Intelligence Platform fills this gap.

---


### 1.1 Comparison with Existing Tools

| Feature | NCRB Website | CrimeAnalyzer (proprietary) | Manual Excel Analysis | **This Platform** |
|---|---|---|---|---|
| Open source | ✗ | ✗ | N/A | **✓ (MIT)** |
| Handles schema inconsistencies across years | ✗ | Unknown | Manual | **✓ Automated** |
| Scalable (big data ready) | ✗ | ✗ | ✗ | **✓ (Spark/HDFS)** |
| State name harmonization | ✗ | Unknown | Manual | **✓ (15+ aliases)** |
| Machine learning analytics | ✗ | Limited | ✗ | **✓ (KMeans, Ridge)** |
| Interactive visualizations | Static PDF | ✗ | Basic charts | **✓ (Choropleth, Chart.js)** |
| Reproducible pipeline | ✗ | ✗ | ✗ | **✓ (Docker + scripts)** |
| Forecasting | ✗ | ✗ | ✗ | **✓ (2015–2020)** |
| Women safety index | ✗ | ✗ | ✗ | **✓** |

To the best of our knowledge, no existing open-source tool provides an end-to-end, scalable pipeline for NCRB crime data that combines automated data harmonization, machine learning analytics, and interactive visualization in a single reproducible framework.

## 2. Software Description

### 2.1 Software Architecture

The platform follows a modular three-stage pipeline architecture, with each stage operating as an independent PySpark application that reads from and writes to HDFS:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HDFS Storage Layer                          │
│  /crime_data/raw/   →   /crime_data/cleaned/   →   /output/       │
└──────────┬──────────────────────┬──────────────────────┬───────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  STAGE 1         │  │  STAGE 2         │  │  STAGE 3             │
│  Data Preparation│  │  Analytics       │  │  Visualization       │
│                  │  │                  │  │                      │
│ • Schema normal. │  │ • KMeans cluster │  │ • Choropleth maps    │
│ • State mapping  │  │ • Ridge forecast │  │ • Trend line charts  │
│ • Null handling  │  │ • Crime compose. │  │ • Cluster bar charts │
│ • Type casting   │  │ • Women safety   │  │ • Interactive HTML   │
│ • Deduplication  │  │ • Silhouette val │  │ • Dashboard assembly │
└──────────────────┘  └──────────────────┘  └──────────────────────┘
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                    Shared Utilities Layer
              (utils.py, state_mapping.py)
```

**Stage 1 — Data Preparation** (`src/data_preparation.py`): Ingests 18 raw CSV files from HDFS, applies the state name mapping engine to resolve naming inconsistencies across all datasets, normalizes schemas, handles missing values, casts data types, removes duplicates, and writes cleaned Parquet/CSV files back to HDFS. This stage processes datasets including: IPC crimes by state, crimes against women, juvenile crimes, economic offenses, cybercrimes, property crimes, arrests data, court disposals, police strength, prison statistics, crime against children, crime against senior citizens, crimes against SCs/STs, auto theft, and human trafficking.

**Stage 2 — Analytics** (`src/analytics.py`): Reads cleaned data from HDFS and performs four categories of analysis:
- **KMeans Clustering**: Groups states into crime profile clusters using Spark MLlib's KMeans implementation with Silhouette Score validation for optimal k selection.
- **Crime Forecasting**: Applies polynomial feature expansion with Ridge regression to predict future crime trends per state.
- **Crime Composition Analysis**: Computes proportional breakdowns of crime categories per state to identify dominant crime types.
- **Women Safety Index**: Calculates a composite safety score for each state based on weighted crime-against-women indicators.

**Stage 3 — Visualization** (`src/visualization.py`): Reads analytics outputs and GeoJSON boundary data to generate interactive visualizations using Folium and Branca:
- Choropleth maps colored by crime rates, cluster membership, and safety indices.
- Time-series trend charts for crime categories.
- Cluster profile bar charts.
- A unified HTML dashboard (`dashboard/index.html`) that aggregates all visualizations.

**Shared Utilities** (`src/utils.py`, `src/state_mapping.py`): Common functions for Spark session management, HDFS I/O, logging, and the state name resolution engine that maps 60+ naming variants to 36 canonical state/UT names.

### 2.2 Software Functionalities

The platform provides the following core functionalities:

1. **Automated Data Harmonization**: Resolves 60+ state/UT naming variants across 18 NCRB datasets to canonical forms. Handles schema differences, missing value patterns, and type inconsistencies automatically.

2. **KMeans Crime Clustering**: Groups India's 36 states/UTs into crime profile clusters using Spark MLlib. Automatically selects optimal cluster count via Silhouette Score evaluation (k=2 to k=10). Outputs cluster assignments and centroid profiles.

3. **Crime Trend Forecasting**: Generates polynomial Ridge regression models for each state's crime trajectory. Produces multi-year forecasts with configurable prediction horizons.

4. **Crime Composition Profiling**: Computes proportional breakdowns of IPC crime categories per state, identifying dominant and anomalous crime types relative to national averages.

5. **Women Safety Index**: Calculates a composite safety score per state using weighted indicators from crimes-against-women data (rape, kidnapping, dowry deaths, assault, cruelty by husband, etc.).

6. **Interactive Geospatial Visualization**: Generates Folium-based choropleth maps with GeoJSON overlays for all 36 states/UTs. Maps are interactive (zoom, pan, hover tooltips) and export as standalone HTML files.

7. **Self-Contained Dashboard**: Produces a single `index.html` dashboard aggregating all visualizations, charts, and maps. Requires only a web browser to view—no server or software installation needed.

---


### 2.3 Testing

The platform includes a comprehensive test suite covering:

- **Unit tests** for state name normalization (36 canonical states, 15+ alias mappings)
- **Unit tests** for PySpark utility functions (standardization, type casting, column handling)
- **Data integrity tests** validating all 18 CSV files and GeoJSON structural correctness

Tests run on a local Spark session without requiring HDFS:

```bash
python -m pytest tests/ -v
```

## 3. Illustrative Examples

### 3.1 Running the Full Pipeline

The complete pipeline is executed in three sequential steps:

```bash
# Stage 1: Prepare and clean data
spark-submit --master yarn src/data_preparation.py

# Stage 2: Run analytics
spark-submit --master yarn src/analytics.py

# Stage 3: Generate visualizations and dashboard
spark-submit --master yarn src/visualization.py
```

Execution logs are written to `output.log`. On a single-node Hadoop cluster, the full pipeline completes in approximately 3–5 minutes for the included 18-dataset corpus.

### 3.2 Choropleth Map Output

The visualization stage produces interactive choropleth maps. For example, the crime rate choropleth colors each state by its total IPC crime rate per capita, with hover tooltips showing exact values. The women safety index map highlights states with the highest and lowest composite safety scores. Screenshots are available in `docs/screenshots/`:

- `docs/screenshots/choropleth_map.png` — Crime rate choropleth of India
- `docs/screenshots/crime_trends.png` — Time-series trend visualization
- `docs/screenshots/clustering_results.png` — KMeans cluster assignments
- `docs/screenshots/dashboard_overview.png` — Full dashboard view

### 3.3 Interactive Dashboard

Opening `dashboard/index.html` in any web browser presents the unified dashboard. The dashboard includes:
- Navigation between different analysis views
- Interactive maps with zoom/pan/tooltip functionality
- Trend charts showing crime evolution over 2001–2014
- Cluster membership visualizations with profile summaries

### 3.4 State Name Mapping

The state mapping engine demonstrates the harmonization capability:

```python
# Input variants from different NCRB datasets:
"ANDAMAN & NICOBAR"          → "Andaman & Nicobar Islands"
"A&N ISLANDS"                → "Andaman & Nicobar Islands"
"ANDAMAN & NICOBAR ISLANDS"  → "Andaman & Nicobar Islands"
"D&N HAVELI"                 → "Dadra & Nagar Haveli"
"DADRA & NAGAR HAVELI"       → "Dadra & Nagar Haveli"
"DELHI"                      → "Delhi"
"DELHI UT"                   → "Delhi"
```

This mapping is applied consistently across all 18 datasets, ensuring join integrity for cross-dataset analyses.

---


### 3.5 Docker Quick Start (No HDFS Required)

For users without a Hadoop cluster, the platform can run entirely in Spark local mode:

```bash
# Option A: Docker
docker compose up

# Option B: Direct execution
bash scripts/run_pipeline_local.sh
```

This produces the same outputs as the full HDFS pipeline, making the platform accessible to researchers without big data infrastructure experience.

## 4. Impact

The India Crime Intelligence Platform addresses a tangible need in the Indian data science and public policy ecosystem:

**Research enablement.** Researchers studying crime patterns in India currently spend significant effort on data cleaning before analysis. By providing a reproducible, automated pipeline that harmonizes 18 NCRB datasets, the platform eliminates this preprocessing bottleneck and allows researchers to focus on hypothesis testing and model development. The modular architecture also makes it straightforward to add new analysis modules.

**Accessibility for non-technical users.** The self-contained HTML dashboard output means that policymakers, journalists, NGOs, and students can explore crime insights without installing any software or writing code. This democratizes access to data-driven crime analysis, which has traditionally required expensive proprietary tools or advanced programming skills.

**Reproducibility and transparency.** The entire pipeline—from raw data to final visualizations—is open-source, version-controlled, and executable with a single sequence of `spark-submit` commands. This supports reproducible research practices and enables independent verification of analytical results.

**Policy applications.** The women safety index, crime clustering, and forecasting outputs are directly relevant to state-level policy planning, resource allocation for law enforcement, and evaluation of crime prevention programs. The choropleth visualizations make geographic patterns immediately apparent to decision-makers.

**Educational value.** The platform serves as a practical reference implementation for courses on big data analytics, demonstrating a complete PySpark + Hadoop pipeline with real-world data, from ingestion through analysis to visualization.

---

## 5. Conclusions

The India Crime Intelligence Platform provides a complete, open-source big data pipeline for analyzing India's national crime statistics. By combining automated data harmonization, Spark-based scalable analytics (clustering, forecasting, composition analysis, safety indexing), and interactive visualization in a single modular framework, it addresses a gap in the existing landscape of crime analysis tools for Indian data.

The current version (v1.0.0) processes 18 NCRB datasets spanning 2001–2014 across 36 states and union territories. Future development directions include:

1. **Extended temporal coverage**: Incorporating NCRB data from 2015 onward as datasets become available in machine-readable formats.
2. **Real-time data ingestion**: Adding support for streaming crime data sources via Spark Structured Streaming.
3. **Hive integration**: Enabling SQL-based ad-hoc querying of the cleaned crime data warehouse through Apache Hive (infrastructure scaffolding is already present in the repository).
4. **Advanced modeling**: Integrating deep learning-based forecasting (LSTM, Prophet) and spatial autocorrelation analysis (Moran's I).
5. **District-level analysis**: Extending the pipeline to support district-level granularity where NCRB data permits.

---

## Acknowledgements

The authors would like to thank **Prof. Ashish Bhatt** (ashish.bhatt@vit.ac.in), School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, for his guidance and supervision of this project.

---

## References

[1] National Crime Records Bureau (NCRB), Ministry of Home Affairs, Government of India. "Crime in India" Annual Reports, 2001–2014. Available: https://ncrb.gov.in/

[2] Apache Software Foundation. "HDFS Architecture." Available: https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html

[3] M. Zaharia, R.S. Xin, P. Wendell, T. Das, M. Armbrust, A. Dave, X. Meng, J. Rosen, S. Venkataraman, M.J. Franklin, A. Ghodsi, J. Gonzalez, S. Shenker, and I. Stoica, "Apache Spark: A Unified Engine for Big Data Processing," *Communications of the ACM*, vol. 59, no. 11, pp. 56–65, 2016.

[4] X. Meng, J. Bradley, B. Yavuz, E. Sparks, S. Venkataraman, D. Liu, J. Freeman, D.B. Tsai, M. Amde, S. Owen, D. Xin, R. Xin, M.J. Franklin, R. Zadeh, M. Zaharia, and A. Talwalkar, "MLlib: Machine Learning in Apache Spark," *Journal of Machine Learning Research*, vol. 17, no. 34, pp. 1–7, 2016.

[5] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and É. Duchesnay, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[6] Folium Contributors. "Folium: Python Data, Leaflet.js Maps." Available: https://python-visualization.github.io/folium/

[7] J.A. Hartigan and M.A. Wong, "Algorithm AS 136: A K-Means Clustering Algorithm," *Journal of the Royal Statistical Society. Series C (Applied Statistics)*, vol. 28, no. 1, pp. 100–108, 1979.

[8] A.E. Hoerl and R.W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems," *Technometrics*, vol. 12, no. 1, pp. 55–67, 1970.

[9] P.J. Rousseeuw, "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis," *Journal of Computational and Applied Mathematics*, vol. 20, pp. 53–65, 1987.

[10] R.J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, Melbourne, Australia, 2021. Available: https://otexts.com/fpp3/

---

*Manuscript prepared following the SoftwareX Original Software Publication (OSP) template, v4 (October 2024).*
