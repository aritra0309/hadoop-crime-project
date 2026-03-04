from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

spark = SparkSession.builder.appName("Crime Data Preparation").getOrCreate()

def normalize_ipc_df(df):
    """Normalize column names across different datasets"""
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
print("Normalizing schemas...")
df_2001_2012 = normalize_ipc_df(df_2001_2012)
df_2013 = normalize_ipc_df(df_2013)
df_2014 = normalize_ipc_df(df_2014)

required_cols = ["state", "district", "year", "total_ipc_crimes"]

df_2001_2012 = df_2001_2012.select(*[c for c in required_cols if c in df_2001_2012.columns])
df_2013 = df_2013.select(*[c for c in required_cols if c in df_2013.columns])
df_2014 = df_2014.select(*[c for c in required_cols if c in df_2014.columns])

# Combine data
print("Combining datasets...")
crime_df = df_2001_2012.unionByName(df_2013, allowMissingColumns=True).unionByName(
    df_2014, allowMissingColumns=True
)

# Clean data - convert to proper types
print("Cleaning data...")
crime_df = crime_df.filter(col("total_ipc_crimes").isNotNull())
crime_df = crime_df.withColumn("total_ipc_crimes", col("total_ipc_crimes").cast("double"))
crime_df = crime_df.withColumn("year", col("year").cast("int"))

print("Data loaded and normalized")
crime_df.printSchema()
crime_df.show(10, False)
print(f"Total rows: {crime_df.count()}")

# Save cleaned data
print("\nSaving cleaned data...")
crime_df.write.mode("overwrite").option("header", "true").csv("output/cleaned_ipc_crime_data")
print("✓ Cleaned data written to output/cleaned_ipc_crime_data")

spark.stop()
print("\n✓ Data preparation complete!")
