# NOTE: Superseded by src/analytics.py for Phase 2 dashboard JSON analytics.

import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg, stddev, sum as spark_sum, col, when, lit, coalesce
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.ensemble import GradientBoostingRegressor


# =====================================
# START SPARK
# =====================================

spark = (
    SparkSession.builder
    .appName("Crime Analytics - Comprehensive")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

HDFS_OUTPUT = "hdfs://localhost:9000/crime/output"


# =====================================================================
# LOAD MASTER DATASET
# =====================================================================

print("=" * 60)
print("LOADING MASTER CRIME DATASET")
print("=" * 60)

master_df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_OUTPUT}/master_crime_data"
)

print(f"Loaded {master_df.count()} rows, {len(master_df.columns)} columns")

# Also load IPC-only for backward compatibility
ipc_df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_OUTPUT}/cleaned_ipc_crime_data"
)


# =====================================================================
# SECTION 1: STATE AGGREGATION
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 1: STATE AGGREGATION")
print("=" * 60)

state_agg = (
    ipc_df
    .groupBy("state", "year")
    .agg(spark_sum("total_ipc_crimes").alias("total_crimes"))
)

state_agg.show(20, False)


# =====================================================================
# SECTION 2: ENHANCED CLUSTERING (Multi-feature + Silhouette)
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 2: ENHANCED STATE CLUSTERING")
print("=" * 60)

# Build richer features for clustering
state_features = (
    master_df
    .groupBy("state")
    .agg(
        # IPC crime features
        avg("total_ipc_crimes").alias("avg_ipc_crimes"),
        stddev("total_ipc_crimes").alias("ipc_variability"),

        # Crimes against women features
        avg("total_crimes_women").alias("avg_crimes_women"),

        # Violence indicators
        avg("murder").alias("avg_murder"),
        avg("rape").alias("avg_rape"),
        avg("robbery").alias("avg_robbery"),
        avg("burglary").alias("avg_burglary"),

        # Property crime
        avg("value_stolen").alias("avg_value_stolen"),
        avg("recovery_rate").alias("avg_recovery_rate"),

        # Auto theft
        avg("auto_theft_stolen").alias("avg_auto_theft"),

        # Fraud
        avg("total_fraud_cases").alias("avg_fraud"),

        # Firearms
        avg("murder_by_firearms").alias("avg_firearms_murder"),
    )
)

state_features = state_features.na.fill(0)
state_features = state_features.dropna()

# Assemble features
feature_cols = [
    "avg_ipc_crimes", "ipc_variability",
    "avg_crimes_women", "avg_murder", "avg_rape",
    "avg_robbery", "avg_burglary",
    "avg_value_stolen", "avg_recovery_rate",
    "avg_auto_theft", "avg_fraud", "avg_firearms_murder"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
state_features = assembler.transform(state_features)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=True,
    withStd=True
)
state_scaled = scaler.fit(state_features).transform(state_features)


# ── Silhouette-based k selection (better than elbow heuristic) ────────

print("\nFinding optimal k using silhouette score...")

evaluator = ClusteringEvaluator(
    featuresCol="scaledFeatures",
    predictionCol="prediction",
    metricName="silhouette"
)

silhouette_scores = []

for k in range(2, 8):
    kmeans = KMeans(k=k, seed=42, featuresCol="scaledFeatures")
    model = kmeans.fit(state_scaled)
    predictions = model.transform(state_scaled)
    score = evaluator.evaluate(predictions)
    silhouette_scores.append((k, score))
    print(f"  k={k}  silhouette={score:.4f}")

best_k = max(silhouette_scores, key=lambda x: x[1])[0]
print(f"\n✓ Best k = {best_k} (highest silhouette score)")


# ── Final KMeans ──────────────────────────────────────────────────────

kmeans = KMeans(
    k=best_k,
    seed=42,
    featuresCol="scaledFeatures",
    predictionCol="cluster"
)

model = kmeans.fit(state_scaled)
clustered_states = model.transform(state_scaled)

# Assign descriptive cluster labels based on avg crime level
cluster_stats = (
    clustered_states
    .groupBy("cluster")
    .agg(avg("avg_ipc_crimes").alias("cluster_avg"))
    .orderBy("cluster_avg")
    .toPandas()
)

# Label clusters from low to high
labels = ["Low Crime", "Moderate Crime", "High Crime",
          "Very High Crime", "Extreme Crime"]
cluster_label_map = {}
for i, row in cluster_stats.iterrows():
    cluster_label_map[int(row["cluster"])] = labels[min(i, len(labels) - 1)]

print("\nCluster Labels:")
for c, l in sorted(cluster_label_map.items()):
    print(f"  Cluster {c} → {l}")

print("\nClustered States:")
clustered_states.select("state", "cluster", "avg_ipc_crimes", "avg_crimes_women").show(40, False)


# ── Save clusters ─────────────────────────────────────────────────────

(
    clustered_states
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet(f"{HDFS_OUTPUT}/clustered_crime_data")
)

print("✓ Saved clustered data to HDFS")


# =====================================================================
# SECTION 3: CRIME COMPOSITION ANALYSIS
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 3: CRIME COMPOSITION ANALYSIS")
print("=" * 60)

# Calculate what % of total IPC crimes each type represents per state
composition = (
    master_df
    .groupBy("state")
    .agg(
        spark_sum("murder").alias("murder"),
        spark_sum("rape").alias("rape"),
        spark_sum("kidnapping").alias("kidnapping"),
        spark_sum("robbery").alias("robbery"),
        spark_sum("burglary").alias("burglary"),
        spark_sum("theft").alias("theft"),
        spark_sum("riots").alias("riots"),
        spark_sum("cheating").alias("cheating"),
        spark_sum("arson").alias("arson"),
        spark_sum("dowry_deaths").alias("dowry_deaths"),
        spark_sum("total_ipc_crimes").alias("total"),
    )
)

# Calculate percentages
crime_types = ["murder", "rape", "kidnapping", "robbery", "burglary",
               "theft", "riots", "cheating", "arson", "dowry_deaths"]

for ct in crime_types:
    composition = composition.withColumn(
        f"{ct}_pct",
        when(col("total") > 0, (col(ct) / col("total")) * 100).otherwise(0)
    )

composition.select("state", *[f"{ct}_pct" for ct in crime_types]).show(10, False)

(
    composition
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet(f"{HDFS_OUTPUT}/crime_composition")
)

print("✓ Saved crime composition data")


# =====================================================================
# SECTION 4: WOMEN SAFETY INDEX
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 4: WOMEN SAFETY INDEX")
print("=" * 60)

women_agg = (
    master_df
    .groupBy("state")
    .agg(
        avg("rape_women").alias("avg_rape"),
        avg("kidnapping_women").alias("avg_kidnapping"),
        avg("dowry_deaths_women").alias("avg_dowry_deaths"),
        avg("assault_women").alias("avg_assault"),
        avg("domestic_cruelty").alias("avg_domestic_cruelty"),
        avg("total_crimes_women").alias("avg_total_women_crimes"),
    )
)

women_agg = women_agg.na.fill(0)

# Higher total = worse safety → invert for "safety index"
women_pdf = women_agg.toPandas()
max_crimes = women_pdf["avg_total_women_crimes"].max()

if max_crimes > 0:
    women_pdf["safety_index"] = (
        (1 - women_pdf["avg_total_women_crimes"] / max_crimes) * 100
    ).round(2)
else:
    women_pdf["safety_index"] = 100.0

women_pdf = women_pdf.sort_values("safety_index", ascending=False)

print("\nWomen Safety Index (higher = safer):")
print(women_pdf[["state", "avg_total_women_crimes", "safety_index"]].to_string(index=False))

women_spark = spark.createDataFrame(women_pdf)
(
    women_spark
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet(f"{HDFS_OUTPUT}/women_safety_index")
)

print("✓ Saved women safety index")


# =====================================================================
# SECTION 5: PROPERTY CRIME ANALYSIS
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 5: PROPERTY CRIME RECOVERY ANALYSIS")
print("=" * 60)

property_df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_OUTPUT}/property_stolen_recovered"
)

prop_summary = (
    property_df
    .groupBy("state")
    .agg(
        spark_sum("value_stolen").alias("total_value_stolen"),
        spark_sum("value_recovered").alias("total_value_recovered"),
        avg("recovery_rate").alias("avg_recovery_rate"),
    )
    .orderBy(col("avg_recovery_rate").desc())
)

print("\nProperty Recovery Rates by State:")
prop_summary.show(40, False)

(
    prop_summary
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet(f"{HDFS_OUTPUT}/property_recovery_analysis")
)

print("✓ Saved property recovery analysis")


# =====================================================================
# SECTION 6: OPTIMIZED FORECASTING
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 6: STATE-WISE CRIME FORECASTING (2015-2020)")
print("=" * 60)


def make_features(years_array):
    """
    Enrich raw year values with trend and cyclical features.
    Features: t (linear), t² (quadratic), sin/cos (~7-year cycle)
    """
    years_array = years_array.flatten()
    t = years_array - 2001
    return np.column_stack([
        t,
        t ** 2,
        np.sin(2 * np.pi * t / 7),
        np.cos(2 * np.pi * t / 7),
    ])


def interpolate_gaps(state_data, year_range):
    """
    Reindex to full year range and linearly interpolate gaps.
    Marks interpolated rows so we can track them.
    """
    original_years = set(state_data.index)
    state_data = state_data.reindex(year_range)
    state_data["is_interpolated"] = ~state_data.index.isin(original_years)
    state_data["total_crimes"] = (
        state_data["total_crimes"]
        .interpolate(method="linear", limit_direction="both")
    )
    return state_data


print("Running optimized state-wise forecasting...")

pdf = state_agg.toPandas()
pdf = pdf.sort_values(["state", "year"])
states = pdf["state"].unique()

predictions = []
model_report = []

for state in states:

    state_data = (
        pdf[pdf["state"] == state]
        .copy()
        .set_index("year")
    )

    state_data = interpolate_gaps(state_data, range(2001, 2015))

    years_train = state_data.index.values.reshape(-1, 1)
    values_train = state_data["total_crimes"].values.astype(float)

    valid_mask = ~np.isnan(values_train)
    if valid_mask.sum() < 5:
        print(f"  Skipping {state} — insufficient data ({valid_mask.sum()} valid rows)")
        continue

    years_train = years_train[valid_mask]
    values_train = values_train[valid_mask]

    X_train = make_features(years_train)
    X_future = make_features(np.arange(2015, 2021).reshape(-1, 1))

    # ── Candidate models ──────────────────────────────────────────
    candidates = {
        "ridge_poly2": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=10.0)),
        ]),
        "ridge_poly3": Pipeline([
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("ridge", Ridge(alpha=50.0)),
        ]),
        # Reduced complexity GBR for small datasets
        "gbr": GradientBoostingRegressor(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
    }

    # ── TimeSeriesSplit CV ────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=3)

    best_name, best_model, best_mae = None, None, np.inf

    for name, candidate in candidates.items():
        try:
            scores = cross_val_score(
                candidate, X_train, values_train,
                cv=tscv,
                scoring="neg_mean_absolute_error"
            )
            mae = -scores.mean()
            if mae < best_mae:
                best_mae = mae
                best_name = name
                best_model = candidate
        except Exception as e:
            pass

    if best_model is None:
        print(f"  All models failed for {state}, skipping")
        continue

    # ── Refit on full training data ───────────────────────────────
    best_model.fit(X_train, values_train)

    train_preds = best_model.predict(X_train)
    train_r2 = r2_score(values_train, train_preds)
    train_mae = mean_absolute_error(values_train, train_preds)

    forecast = best_model.predict(X_future)
    forecast = np.clip(forecast, 0, None)

    model_report.append({
        "state": state,
        "best_model": best_name,
        "cv_mae": round(best_mae, 2),
        "train_r2": round(train_r2, 4),
        "train_mae": round(train_mae, 2),
    })

    for yr, val in zip(range(2015, 2021), forecast):
        predictions.append({
            "state": state,
            "year": yr,
            "total_crimes": float(val),
            "type": "predicted",
        })


# ── Model selection summary ───────────────────────────────────────────

report_df = pd.DataFrame(model_report).sort_values("cv_mae")
print("\n" + "─" * 50)
print("         Model Selection Report             ")
print("─" * 50)
print(report_df.to_string(index=False))
print(f"\nAvg CV MAE  : {report_df['cv_mae'].mean():>12,.0f}")
print(f"Avg Train R²: {report_df['train_r2'].mean():>12.4f}")
print("─" * 50)

pred_df = pd.DataFrame(predictions)


# =====================================================================
# COMBINE HISTORICAL + PREDICTED
# =====================================================================

historical_df = pdf.copy()
historical_df["type"] = "actual"

combined_df = pd.concat([historical_df, pred_df])
combined_df = combined_df.sort_values(["state", "year"])

# Drop the interpolation tracking column if present
if "is_interpolated" in combined_df.columns:
    combined_df = combined_df.drop(columns=["is_interpolated"])

print(f"\nCombined dataset: {len(combined_df)} rows")


# =====================================================================
# SAVE ALL ANALYTICS OUTPUTS
# =====================================================================

spark_combined = spark.createDataFrame(combined_df)

(
    spark_combined
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet(f"{HDFS_OUTPUT}/state_crime_time_series")
)

print("✓ Saved time series (actual + predicted)")

# Save model report
report_spark = spark.createDataFrame(report_df)
(
    report_spark
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet(f"{HDFS_OUTPUT}/model_report")
)

print("✓ Saved model report")

# Save cluster label mapping as a small dataset
cluster_label_rows = [{"cluster": k, "label": v} for k, v in cluster_label_map.items()]
cluster_label_spark = spark.createDataFrame(cluster_label_rows)
(
    cluster_label_spark
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet(f"{HDFS_OUTPUT}/cluster_labels")
)

print("✓ Saved cluster labels")


spark.stop()

print("\n" + "=" * 60)
print("ANALYTICS COMPLETE")
print(f"  → Clustering: {best_k} clusters using {len(feature_cols)} features")
print(f"  → Forecasting: {len(states)} states, best models selected via CV")
print(f"  → Crime composition, women safety index, property analysis")
print("=" * 60)
