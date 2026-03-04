from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, year as spark_year
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
import time

spark = SparkSession.builder.appName("Crime Data Analytics").getOrCreate()

print("Loading cleaned data...")
crime_df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "output/cleaned_ipc_crime_data"
)

print(f"Data loaded: {crime_df.count()} rows")
crime_df.show(5, False)

# State-level aggregation
print("\n" + "="*50)
print("STATE AGGREGATION")
print("="*50)

state_agg = (
    crime_df
    .groupBy("state", "year")
    .agg(spark_sum("total_ipc_crimes").alias("total_crimes"))
    .orderBy("state", "year")
)

state_agg.show(10, False)

print("\nState aggregation complete")

# Clustering preparation
print("\n" + "="*50)
print("K-MEANS CLUSTERING")
print("="*50)

state_df = state_agg.groupBy("state").agg(
    spark_sum("total_crimes").alias("total_crimes")
)

# Create vector column for clustering
assembler = VectorAssembler(inputCols=["total_crimes"], outputCol="features")
state_features = assembler.transform(state_df)

# Standardize features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
state_scaled = scaler.fit(state_features).transform(state_features)

# K-means clustering
kmeans = KMeans(k=5, featuresCol="scaledFeatures", predictionCol="cluster")
model = kmeans.fit(state_scaled)

clustered_states = model.transform(state_scaled)
print("Clustered states (5 clusters):")
clustered_states.show(10, False)

# Save clustered data
print("\nSaving clustered data...")
clustered_states.write.mode("overwrite").parquet("output/clustered_crime_data")

# Time-series prediction
print("\n" + "="*50)
print("LINEAR REGRESSION - CRIME PREDICTION")
print("="*50)

train_df = state_agg.filter((col("year") >= 2001) & (col("year") <= 2014))

# Prepare features for regression
assembler_reg = VectorAssembler(inputCols=["year"], outputCol="features")
train_df_features = assembler_reg.transform(train_df.select("year", "total_crimes"))

# Create and train model
lr_model = LinearRegression(featuresCol="features", labelCol="total_crimes", maxIter=10, regParam=0.3)
lr_model = lr_model.fit(train_df_features)

# Predict for 2015-2020
future_years = spark.createDataFrame([(year,) for year in range(2015, 2021)], ["year"])
future_features = assembler_reg.transform(future_years)
predictions = lr_model.transform(future_features)

print(f"\nPredictions for 2015-2020:")
predictions.select("year", "prediction").show(10, False)

# Combine historical and predicted data
historical = state_agg.select("state", "year", col("total_crimes").alias("actual_crimes"), 
                              col("total_crimes").alias("predicted_crimes"))

# Add state column to predictions for proper output structure
predictions_pivot = predictions.select("year", col("prediction").alias("predicted_crimes"))
print("Saving state aggregation with predictions...")

state_agg.write.mode("overwrite").parquet("output/state_agg_with_predictions")
predictions.write.mode("overwrite").parquet("output/predictions_2015_2020")

print("✓ Clustering complete")
print("✓ Predictions saved")

spark.stop()
print("\n✓ Analytics processing complete!")
