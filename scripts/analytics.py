import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, stddev, sum as spark_sum

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor


# =====================================
# START SPARK
# =====================================

spark = (
    SparkSession.builder
    .appName("Crime Analytics")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

print("Loading cleaned data from HDFS...")

crime_df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "hdfs://localhost:9000/crime/output/cleaned_ipc_crime_data"
)

print("Rows:", crime_df.count())


# =====================================
# STATE AGGREGATION
# =====================================

print("\nAggregating crime by state and year...")

state_agg = (
    crime_df
    .groupBy("state", "year")
    .agg(spark_sum("total_ipc_crimes").alias("total_crimes"))
)

state_agg.show(20, False)


# =====================================
# CLUSTER FEATURES
# =====================================

print("\nCreating clustering features...")

state_features = (
    state_agg
    .groupBy("state")
    .agg(
        spark_sum("total_crimes").alias("total_crimes"),
        avg("total_crimes").alias("avg_crimes"),
        stddev("total_crimes").alias("crime_variability")
    )
)

state_features = state_features.dropna()


# =====================================
# VECTOR ASSEMBLER
# =====================================

assembler = VectorAssembler(
    inputCols=["avg_crimes", "crime_variability"],
    outputCol="features"
)

state_features = assembler.transform(state_features)


# =====================================
# SCALE FEATURES
# =====================================

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=True,
    withStd=True
)

state_scaled = scaler.fit(state_features).transform(state_features)


# =====================================
# ELBOW METHOD
# =====================================

print("\nRunning elbow method...")

costs = []

for k in range(2, 10):

    kmeans = KMeans(
        k=k,
        seed=42,
        featuresCol="scaledFeatures"
    )

    model = kmeans.fit(state_scaled)
    cost  = model.summary.trainingCost
    costs.append((k, cost))

print("\nElbow Results:")
for k, cost in costs:
    print("k =", k, "  cost =", cost)

# ── Pick k using largest drop in inertia (elbow heuristic) ────────────
cost_vals  = [c for _, c in costs]
drops      = [cost_vals[i] - cost_vals[i + 1] for i in range(len(cost_vals) - 1)]
best_k     = costs[drops.index(max(drops)) + 1][0]   # k AFTER biggest drop

print("\nSelected k =", best_k)


# =====================================
# FINAL KMEANS
# =====================================

kmeans = KMeans(
    k=best_k,
    seed=42,
    featuresCol="scaledFeatures",
    predictionCol="cluster"
)

model = kmeans.fit(state_scaled)

clustered_states = model.transform(state_scaled)

clustered_states.show(20, False)


# =====================================
# SAVE CLUSTERS
# =====================================

(
    clustered_states
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet("hdfs://localhost:9000/crime/output/clustered_crime_data")
)

print("Saved clusters to HDFS")


# =====================================
# HELPER FUNCTIONS FOR FORECASTING
# =====================================

def make_features(years_array):
    """
    Enrich raw year values with trend and cyclical features.
    Features:
      - t        : linear time index (year - 2001)
      - t²       : quadratic trend
      - sin/cos  : ~7-year crime cycle approximation
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
    Reindex to full year range and use linear interpolation
    for interior missing years; edge years are extrapolated.
    This is more accurate than ffill/bfill which just copies
    the nearest known value.
    """
    state_data = state_data.reindex(year_range)
    state_data["total_crimes"] = (
        state_data["total_crimes"]
        .interpolate(method="linear", limit_direction="both")
    )
    return state_data


# =====================================
# STATE-WISE OPTIMIZED FORECASTING
# =====================================

print("\nRunning optimized state-wise forecasting...")

pdf    = state_agg.toPandas()
pdf    = pdf.sort_values(["state", "year"])
states = pdf["state"].unique()

predictions  = []
model_report = []

for state in states:

    state_data = (
        pdf[pdf["state"] == state]
        .copy()
        .set_index("year")
    )

    # Fill gaps with interpolation instead of ffill/bfill
    state_data = interpolate_gaps(state_data, range(2001, 2015))

    years_train  = state_data.index.values.reshape(-1, 1)
    values_train = state_data["total_crimes"].values.astype(float)

    # Drop any remaining NaNs
    valid_mask   = ~np.isnan(values_train)
    if valid_mask.sum() < 5:
        print(f"  Skipping {state} — insufficient data ({valid_mask.sum()} valid rows)")
        continue

    years_train  = years_train[valid_mask]
    values_train = values_train[valid_mask]

    X_train  = make_features(years_train)
    X_future = make_features(np.arange(2015, 2021).reshape(-1, 1))

    # ── Candidate models ──────────────────────────────────────────────
    candidates = {
        "ridge_poly2": Pipeline([
            ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=10.0)),
        ]),
        "ridge_poly3": Pipeline([
            ("poly",  PolynomialFeatures(degree=3, include_bias=False)),
            ("ridge", Ridge(alpha=50.0)),     # higher alpha to dampen cubic
        ]),
        "gbr": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
    }

    # ── TimeSeriesSplit CV — respects temporal order ───────────────────
    # Regular k-fold would let the model peek at future data during
    # validation, giving falsely optimistic MAE scores.
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
                best_mae   = mae
                best_name  = name
                best_model = candidate
        except Exception as e:
            print(f"  CV failed for {state}/{name}: {e}")

    if best_model is None:
        print(f"  All models failed for {state}, skipping")
        continue

    # ── Refit winner on full 2001–2014 training data ──────────────────
    best_model.fit(X_train, values_train)

    train_preds = best_model.predict(X_train)
    train_r2    = r2_score(values_train, train_preds)
    train_mae   = mean_absolute_error(values_train, train_preds)

    forecast = best_model.predict(X_future)
    forecast = np.clip(forecast, 0, None)   # crime counts cannot be negative

    model_report.append({
        "state":      state,
        "best_model": best_name,
        "cv_mae":     round(best_mae, 2),
        "train_r2":   round(train_r2, 4),
        "train_mae":  round(train_mae, 2),
    })

    for yr, val in zip(range(2015, 2021), forecast):
        predictions.append({
            "state":        state,
            "year":         yr,
            "total_crimes": float(val),
            "type":         "predicted",
        })

# ── Model selection summary ───────────────────────────────────────────
report_df = pd.DataFrame(model_report).sort_values("cv_mae")
print("\n────────────────────────────────────────────")
print("         Model Selection Report             ")
print("────────────────────────────────────────────")
print(report_df.to_string(index=False))
print(f"\nAvg CV MAE  : {report_df['cv_mae'].mean():>12,.0f}")
print(f"Avg Train R²: {report_df['train_r2'].mean():>12.4f}")
print("────────────────────────────────────────────")

pred_df = pd.DataFrame(predictions)


# =====================================
# HISTORICAL DATA
# =====================================

historical_df         = pdf.copy()
historical_df["type"] = "actual"


# =====================================
# COMBINE DATA
# =====================================

combined_df = pd.concat([historical_df, pred_df])
combined_df = combined_df.sort_values(["state", "year"])

print("\nCombined dataset preview:")
print(combined_df.head(10))


# =====================================
# SAVE TIME SERIES
# =====================================

spark_combined = spark.createDataFrame(combined_df)

(
    spark_combined
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet("hdfs://localhost:9000/crime/output/state_crime_time_series")
)

print("\nSaved state-wise crime time series")


# =====================================
# SAVE STATE AGGREGATION
# =====================================

(
    state_agg
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet("hdfs://localhost:9000/crime/output/state_agg_with_predictions")
)

print("Saved aggregated data")


spark.stop()

print("\n✓ Analytics pipeline complete")