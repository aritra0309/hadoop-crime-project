from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, upper, trim, regexp_replace
import os


# =====================================
# START SPARK
# =====================================

spark = (
    SparkSession.builder
    .appName("Crime Data Preparation")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)


# =====================================
# PROJECT PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


# =====================================
# COLUMN NORMALIZATION
# =====================================

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


# =====================================
# LOAD DATA
# =====================================

print("Loading crime data...")

df_2001_2012 = spark.read.option("header","true").option("inferSchema","true").csv(
    "hdfs://localhost:9000/crime/input/01_District_wise_crimes_committed_IPC_2001_2012.csv"
)

df_2013 = spark.read.option("header","true").option("inferSchema","true").csv(
    "hdfs://localhost:9000/crime/input/01_District_wise_crimes_committed_IPC_2013.csv"
)

df_2014 = spark.read.option("header","true").option("inferSchema","true").csv(
    "hdfs://localhost:9000/crime/input/01_District_wise_crimes_committed_IPC_2014.csv"
)


# =====================================
# NORMALIZE SCHEMA
# =====================================

print("Normalizing schemas...")

df_2001_2012 = normalize_ipc_df(df_2001_2012)
df_2013 = normalize_ipc_df(df_2013)
df_2014 = normalize_ipc_df(df_2014)

required_cols = ["state","district","year","total_ipc_crimes"]

df_2001_2012 = df_2001_2012.select(*required_cols)
df_2013 = df_2013.select(*required_cols)
df_2014 = df_2014.select(*required_cols)


# =====================================
# COMBINE DATASETS
# =====================================

print("Combining datasets...")

crime_df = df_2001_2012.unionByName(df_2013).unionByName(df_2014)


# =====================================
# CLEAN DATA
# =====================================

print("Cleaning data...")

crime_df = crime_df.filter(col("total_ipc_crimes").isNotNull())

crime_df = crime_df.withColumn(
    "total_ipc_crimes",
    col("total_ipc_crimes").cast("double")
)

crime_df = crime_df.withColumn(
    "year",
    col("year").cast("int")
)


# =====================================
# NORMALIZE STATE NAMES
# =====================================

print("Normalizing state names...")

crime_df = crime_df.withColumn(
    "state",
    upper(trim(col("state")))
)

# Fix spacing issues like A&N → A & N
crime_df = crime_df.withColumn(
    "state",
    regexp_replace(col("state"), "&", " & ")
)

# Remove duplicate spaces
crime_df = crime_df.withColumn(
    "state",
    regexp_replace(col("state"), " +", " ")
)

crime_df = crime_df.withColumn(
    "state",
    trim(col("state"))
)


# =====================================
# AGGREGATE DISTRICT → STATE
# =====================================

print("Aggregating district data to state level...")

state_year_df = (
    crime_df
    .groupBy("state","year")
    .agg(
        spark_sum("total_ipc_crimes").alias("total_ipc_crimes")
    )
)


# =====================================
# SORT DATA
# =====================================

state_year_df = state_year_df.orderBy("state","year")


# =====================================
# PREVIEW DATA
# =====================================

print("\nCleaned dataset preview:")

state_year_df.show(20, False)

print("Total rows:", state_year_df.count())


# =====================================
# SAVE CLEAN DATA
# =====================================

print("\nSaving cleaned data to HDFS...")

(
    state_year_df
    .coalesce(1)   # prevents Spark memory warnings
    .write
    .mode("overwrite")
    .option("header","true")
    .csv("hdfs://localhost:9000/crime/output/cleaned_ipc_crime_data")
)

print("✓ Cleaned data written to HDFS")


# =====================================
# STOP SPARK
# =====================================

spark.stop()

print("\n✓ Data preparation complete")