from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, desc
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import folium
from folium.plugins import HeatMap
import pandas as pd

spark = SparkSession.builder.appName("Crime Pattern Analysis").getOrCreate()

def normalize_ipc_df(df):
    rename_map = {
        "STATE/UT": "state",
        "States/UTs": "state",
        "DISTRICT": "district",
        "District": "district",
        "YEAR": "year",
        "Year": "year",
        "TOTAL IPC CRIMES": "total_ipc_crimes",
        "Total Cognizable IPC crimes": "total_ipc_crimes",
        "Other IPC crimes": "other_ipc_crimes"
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)
    return df

# Load data
print("Loading crime data...")
df_2001_2012 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "data/01_District_wise_crimes_committed_IPC_2001_2012.csv"
)

df_2013 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "data/01_District_wise_crimes_committed_IPC_2013.csv"
)

df_2014 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "data/01_District_wise_crimes_committed_IPC_2014.csv"
)

# Normalize schemas
df_2001_2012 = normalize_ipc_df(df_2001_2012)
df_2013 = normalize_ipc_df(df_2013)
df_2014 = normalize_ipc_df(df_2014)

required_cols = ["state", "district", "year", "total_ipc_crimes"]

df_2001_2012 = df_2001_2012.select(*[c for c in required_cols if c in df_2001_2012.columns])
df_2013 = df_2013.select(*[c for c in required_cols if c in df_2013.columns])
df_2014 = df_2014.select(*[c for c in required_cols if c in df_2014.columns])

# Combine data
crime_df = df_2001_2012.unionByName(df_2013, allowMissingColumns=True).unionByName(
    df_2014, allowMissingColumns=True
)

# Clean data - convert to proper types
crime_df = crime_df.filter(col("total_ipc_crimes").isNotNull())
crime_df = crime_df.withColumn("total_ipc_crimes", col("total_ipc_crimes").cast("double"))
crime_df = crime_df.withColumn("year", col("year").cast("int"))

print("Data loaded and normalized")
crime_df.printSchema()
crime_df.show(10, False)
print(f"Total rows: {crime_df.count()}")

# Save cleaned data
crime_df.write.mode("overwrite").option("header", "true").csv("output/cleaned_ipc_crime_data")
print("✓ Cleaned data written to output/cleaned_ipc_crime_data")

# LOCATION-BASED ANALYTICS: Aggregate crimes by state and district
print("\n=== Location-Based Analytics ===")
location_agg = crime_df.groupBy("state", "district").agg(
    spark_sum("total_ipc_crimes").alias("total_crimes")
).orderBy(desc("total_crimes"))

location_agg.write.mode("overwrite").option("header", "true").csv("output/location_aggregated_crime_data")
print("✓ Location aggregated data written")
location_agg.show(20, False)

# Get state-level aggregation for clustering
print("\n=== State-Level Crime Summary ===")
state_agg = crime_df.groupBy("state").agg(
    spark_sum("total_ipc_crimes").alias("total_crimes")
).orderBy(desc("total_crimes"))

state_agg.show(30, False)

# CLUSTERING: K-means on state-level data
print("\n=== K-Means Clustering ===")
assembler = VectorAssembler(inputCols=["total_crimes"], outputCol="features")
feature_df = assembler.transform(state_agg)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(feature_df)
scaled_df = scaler_model.transform(feature_df)

# Fit K-means model with k=5
kmeans = KMeans(k=5, seed=1, maxIter=20)
km_model = kmeans.fit(scaled_df)
clustered_df = km_model.transform(scaled_df)

print("Cluster assignments:")
clustered_df.select("state", "total_crimes", "prediction").show(30, False)

# Save clustered data
clustered_df.write.mode("overwrite").parquet("output/clustered_crime_data")
print("✓ Clustered data written to output/clustered_crime_data")

# PREDICTION: Linear regression on yearly trends (national level)
print("\n=== Time Series Prediction ===")
yearly_agg = crime_df.groupBy("year").agg(spark_sum("total_ipc_crimes").alias("total_crimes")).orderBy("year")

yearly_agg.show()

# Prepare data for linear regression
assembler_pred = VectorAssembler(inputCols=["year"], outputCol="features")
pred_data = assembler_pred.transform(yearly_agg)

# Split into train/test
train_data, test_data = pred_data.randomSplit([0.8, 0.2], seed=1)

# Train linear regression model
lr = LinearRegression(featuresCol="features", labelCol="total_crimes", maxIter=100)
lr_model = lr.fit(train_data)

# Evaluate on test set
predictions = lr_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="total_crimes", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

print(f"✓ Linear Regression RMSE: {rmse:.2f}")
print(f"✓ R² Score: {r2:.4f}")

# Predict future years (2015-2020)
print("\n=== Future Predictions (2015-2020) ===")
future_years = spark.createDataFrame([(2015,), (2016,), (2017,), (2018,), (2019,), (2020,)], ["year"])
future_data = assembler_pred.transform(future_years)
future_predictions = lr_model.transform(future_data)

print("Predicted total crimes:")
future_predictions.select("year", "prediction").show()

# Calculate scaling factor
predicted_total = future_predictions.agg(spark_sum("prediction")).collect()[0][0]
actual_total = location_agg.agg(spark_sum("total_crimes")).collect()[0][0]
scale_factor = predicted_total / actual_total if actual_total > 0 else 1.0

print(f"\nActual total crimes (2001-2014): {actual_total:.0f}")
print(f"Predicted total crimes (2015-2020): {predicted_total:.0f}")

# Create mock coordinates for states (representative Indian locations)
state_coords = {
    "ANDHRA PRADESH": (15.9129, 78.6675),
    "ARUNACHAL PRADESH": (28.2180, 94.7278),
    "ASSAM": (26.2006, 92.9376),
    "BIHAR": (25.0961, 85.3131),
    "CHHATTISGARH": (21.2787, 81.8661),
    "GOA": (15.2993, 73.8243),
    "GUJARAT": (22.2587, 71.1924),
    "HARYANA": (29.0588, 77.0745),
    "HIMACHAL PRADESH": (31.7433, 77.1205),
    "JHARKHAND": (23.6102, 85.2799),
    "KARNATAKA": (15.3173, 75.7139),
    "KERALA": (10.8505, 76.2711),
    "MADHYA PRADESH": (22.9375, 78.6553),
    "MAHARASHTRA": (19.7515, 75.7139),
    "MANIPUR": (24.6637, 93.9063),
    "MEGHALAYA": (25.4670, 91.3662),
    "MIZORAM": (23.1815, 92.9789),
    "NAGALAND": (26.1584, 94.5624),
    "ODISHA": (20.9517, 85.0985),
    "PUNJAB": (31.5204, 74.3587),
    "RAJASTHAN": (27.0238, 74.2179),
    "SIKKIM": (27.5330, 88.5122),
    "TAMIL NADU": (11.1271, 79.2787),
    "TELANGANA": (18.1124, 79.0193),
    "TRIPURA": (23.4104, 91.9882),
    "UTTAR PRADESH": (26.8467, 80.9462),
    "UTTARAKHAND": (30.0668, 79.0193),
    "WEST BENGAL": (24.3745, 88.4631),
    "DELHI": (28.7041, 77.1025),
    "JAMMU & KASHMIR": (33.7782, 76.5769),
    "LADAKH": (34.2045, 77.5771),
}

# Convert state aggregation to pandas and add coordinates
state_pd = state_agg.toPandas()
state_pd['latitude'] = state_pd['state'].map(lambda x: state_coords.get(x.upper(), (20.5937, 78.9629))[0])
state_pd['longitude'] = state_pd['state'].map(lambda x: state_coords.get(x.upper(), (20.5937, 78.9629))[1])

# Calculate predicted crimes per state
state_pd['predicted_crimes'] = state_pd['total_crimes'] * scale_factor

# CREATE MAP VISUALIZATION
print("\n=== Generating Interactive Map ===")
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="OpenStreetMap")

# Add heatmap for actual crimes (2001-2014)
heat_data_actual = []
for _, row in state_pd.iterrows():
    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
        # Normalize crime values for heatmap
        normalized_crime = min(row['total_crimes'] / 1000000, 1.0)
        heat_data_actual.append([row['latitude'], row['longitude'], normalized_crime])

if heat_data_actual:
    HeatMap(heat_data_actual, name="Actual Crimes (2001-2014)", radius=30, blur=15, max_zoom=1).add_to(m)
    print(f"✓ Added heatmap for {len(heat_data_actual)} states (actual crimes)")

# Add heatmap for predicted crimes (2015-2020)
heat_data_pred = []
for _, row in state_pd.iterrows():
    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
        normalized_pred = min(row['predicted_crimes'] / 1000000, 1.0)
        heat_data_pred.append([row['latitude'], row['longitude'], normalized_pred])

if heat_data_pred:
    HeatMap(heat_data_pred, name="Predicted Crimes (2015-2020)", gradient={0.4: 'blue', 0.65: 'lime', 1.0: 'red'}, 
            radius=30, blur=15, max_zoom=1).add_to(m)
    print(f"✓ Added heatmap for {len(heat_data_pred)} states (predicted crimes)")

# Add layer control
folium.LayerControl().add_to(m)

# Save map
m.save("crime_hotspots_map.html")
print("✓ Map saved as crime_hotspots_map.html")

# Summary statistics
print("\n=== ANALYSIS SUMMARY ===")
print(f"Total states analyzed: {len(state_pd)}")
print(f"Highest crime state: {state_pd.iloc[0]['state']} ({state_pd.iloc[0]['total_crimes']:.0f} crimes)")
print(f"Lowest crime state: {state_pd.iloc[-1]['state']} ({state_pd.iloc[-1]['total_crimes']:.0f} crimes)")
print(f"Average crimes per state: {state_pd['total_crimes'].mean():.0f}")
print(f"Crime trend (2015-2020): {'+' if predicted_total > actual_total else ''}{((predicted_total/actual_total - 1) * 100):.1f}%")

spark.stop()
print("\n✓ Analysis complete!")
