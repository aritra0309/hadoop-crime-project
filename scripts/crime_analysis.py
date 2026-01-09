from pyspark.sql import SparkSession
from pyspark.sql.functions import col

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

df_2001_2012 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "hdfs://localhost:9000/crime/input/01_District_wise_crimes_committed_IPC_2001_2012.csv"
)

df_2013 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "hdfs://localhost:9000/crime/input/01_District_wise_crimes_committed_IPC_2013.csv"
)

df_2014 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "hdfs://localhost:9000/crime/input/01_District_wise_crimes_committed_IPC_2014.csv"
)

df_2001_2012 = normalize_ipc_df(df_2001_2012)
df_2013 = normalize_ipc_df(df_2013)
df_2014 = normalize_ipc_df(df_2014)

required_cols = ["state", "district", "year", "total_ipc_crimes"]

df_2001_2012 = df_2001_2012.select(*[c for c in required_cols if c in df_2001_2012.columns])
df_2013 = df_2013.select(*[c for c in required_cols if c in df_2013.columns])
df_2014 = df_2014.select(*[c for c in required_cols if c in df_2014.columns])

crime_df = df_2001_2012.unionByName(df_2013, allowMissingColumns=True).unionByName(
    df_2014, allowMissingColumns=True
)

crime_df.printSchema()
crime_df.show(10, False)
print("Total rows:", crime_df.count())

crime_df.write.mode("overwrite").option("header", "true").csv(
    "hdfs://localhost:9000/crime/output/cleaned_ipc_crime_data"
)

spark.stop()