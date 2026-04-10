# NOTE: This script is superseded by src/data_preparation.py which produces
# district_master and state_master Parquet tables with expanded dataset coverage.
# This file is kept for reference only.

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum as spark_sum, upper, trim, regexp_replace,
    when, lit, coalesce
)
from pyspark.sql.types import DoubleType, IntegerType
import os


# =====================================
# START SPARK
# =====================================

spark = (
    SparkSession.builder
    .appName("Crime Data Preparation - All Datasets")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)


# =====================================
# PROJECT PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

HDFS_INPUT  = "hdfs://localhost:9000/crime/input"
HDFS_OUTPUT = "hdfs://localhost:9000/crime/output"


# =====================================
# STATE NAME STANDARDIZATION
# =====================================

STATE_NAME_MAP = {
    "A & N ISLANDS": "ANDAMAN & NICOBAR ISLANDS",
    "A&N ISLANDS": "ANDAMAN & NICOBAR ISLANDS",
    "ANDAMAN & NICOBAR": "ANDAMAN & NICOBAR ISLANDS",
    "D & N HAVELI": "DADRA & NAGAR HAVELI",
    "D&N HAVELI": "DADRA & NAGAR HAVELI",
    "DAMAN & DIU": "DAMAN & DIU",
    "DELHI UT": "DELHI",
    "DELHI": "DELHI",
    "JAMMU & KASHMIR": "JAMMU & KASHMIR",
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
    "UTTARANCHAL": "UTTARAKHAND",
    "CHHATTISGARH": "CHHATTISGARH",
    "CHATTISGARH": "CHHATTISGARH",
}


def standardize_state(df, state_col="state"):
    """Normalize state names: uppercase, trim, fix known aliases."""

    df = df.withColumn(state_col, upper(trim(col(state_col))))
    df = df.withColumn(state_col, regexp_replace(col(state_col), r'"', ""))
    df = df.withColumn(state_col, regexp_replace(col(state_col), "&", " & "))
    df = df.withColumn(state_col, regexp_replace(col(state_col), " +", " "))
    df = df.withColumn(state_col, trim(col(state_col)))

    # Apply known name mappings
    for old, new in STATE_NAME_MAP.items():
        df = df.withColumn(
            state_col,
            when(col(state_col) == old, lit(new)).otherwise(col(state_col))
        )

    # Filter out aggregate rows
    df = df.filter(
        ~col(state_col).isin("TOTAL (ALL-INDIA)", "TOTAL (ALL INDIA)",
                              "TOTAL", "TOTAL (STATES)", "TOTAL (UTS)",
                              "TOTAL (ALL INDIA)", "ALL-INDIA")
    )

    return df


# =====================================================================
# DATASET 1: IPC CRIMES (District-level → State-level)
# =====================================================================

print("=" * 60)
print("LOADING DATASET 1: IPC CRIMES")
print("=" * 60)


def normalize_ipc_df(df):
    rename_map = {
        "STATE/UT": "state", "States/UTs": "state",
        "DISTRICT": "district", "District": "district",
        "YEAR": "year", "Year": "year",
        "MURDER": "murder", "Murder": "murder",
        "RAPE": "rape", "Rape": "rape",
        "KIDNAPPING & ABDUCTION": "kidnapping",
        "Kidnapping & Abduction_Total": "kidnapping",
        "ROBBERY": "robbery", "Robbery": "robbery",
        "BURGLARY": "burglary", "Burglary": "burglary",
        "THEFT": "theft", "Theft": "theft",
        "RIOTS": "riots", "Riots": "riots",
        "CHEATING": "cheating", "Cheating": "cheating",
        "ARSON": "arson", "Arson": "arson",
        "DOWRY DEATHS": "dowry_deaths", "Dowry Deaths": "dowry_deaths",
        "HURT/GREVIOUS HURT": "hurt",
        "TOTAL IPC CRIMES": "total_ipc_crimes",
        "Total Cognizable IPC crimes": "total_ipc_crimes",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)
    return df


df_01a = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/01_District_wise_crimes_committed_IPC_2001_2012.csv"
)
df_01b = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/01_District_wise_crimes_committed_IPC_2013.csv"
)
df_01c = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/01_District_wise_crimes_committed_IPC_2014.csv"
)

df_01a = normalize_ipc_df(df_01a)
df_01b = normalize_ipc_df(df_01b)
df_01c = normalize_ipc_df(df_01c)

# Select common crime-type columns + total
ipc_cols = ["state", "district", "year", "murder", "rape", "kidnapping",
            "robbery", "burglary", "theft", "riots", "cheating",
            "arson", "dowry_deaths", "total_ipc_crimes"]

# Only select columns that exist in each dataframe
def safe_select(df, cols):
    available = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    result = df.select(*available)
    for m in missing:
        result = result.withColumn(m, lit(0).cast(DoubleType()))
    return result

df_01a = safe_select(df_01a, ipc_cols)
df_01b = safe_select(df_01b, ipc_cols)
df_01c = safe_select(df_01c, ipc_cols)

ipc_df = df_01a.unionByName(df_01b).unionByName(df_01c)

# Cast numeric columns
for c in ipc_cols[3:]:
    ipc_df = ipc_df.withColumn(c, col(c).cast(DoubleType()))
ipc_df = ipc_df.withColumn("year", col("year").cast(IntegerType()))

ipc_df = standardize_state(ipc_df)
ipc_df = ipc_df.filter(col("total_ipc_crimes").isNotNull())

# Aggregate district → state
ipc_state = (
    ipc_df
    .groupBy("state", "year")
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
        spark_sum("total_ipc_crimes").alias("total_ipc_crimes"),
    )
)

ipc_state = ipc_state.orderBy("state", "year")
print(f"IPC crimes: {ipc_state.count()} rows")
ipc_state.show(10, False)


# =====================================================================
# DATASET 2: CRIMES AGAINST WOMEN (District-level → State-level)
# =====================================================================

print("=" * 60)
print("LOADING DATASET 2: CRIMES AGAINST WOMEN")
print("=" * 60)


def normalize_women_df(df):
    rename_map = {
        "STATE/UT": "state", "States/UTs": "state",
        "DISTRICT": "district", "District": "district",
        "Year": "year",
        "Rape": "rape_women",
        "Kidnapping and Abduction": "kidnapping_women",
        "Kidnapping & Abduction_Total": "kidnapping_women",
        "Dowry Deaths": "dowry_deaths_women",
        "Assault on women with intent to outrage her modesty": "assault_women",
        "Assault on Women with intent to outrage her Modesty_Total": "assault_women",
        "Insult to modesty of Women": "insult_modesty",
        "Insult to the Modesty of Women_Total": "insult_modesty",
        "Cruelty by Husband or his Relatives": "domestic_cruelty",
        "Importation of Girls": "importation_girls",
        "Importation of Girls from Foreign Country": "importation_girls",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)
    return df


df_42a = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/42_District_wise_crimes_committed_against_women_2001_2012.csv"
)
df_42b = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/42_District_wise_crimes_committed_against_women_2013.csv"
)
df_42c = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/42_District_wise_crimes_committed_against_women_2014.csv"
)

df_42a = normalize_women_df(df_42a)
df_42b = normalize_women_df(df_42b)
df_42c = normalize_women_df(df_42c)

women_cols = ["state", "district", "year", "rape_women", "kidnapping_women",
              "dowry_deaths_women", "assault_women", "insult_modesty",
              "domestic_cruelty", "importation_girls"]

df_42a = safe_select(df_42a, women_cols)
df_42b = safe_select(df_42b, women_cols)
df_42c = safe_select(df_42c, women_cols)

women_df = df_42a.unionByName(df_42b).unionByName(df_42c)

for c in women_cols[3:]:
    women_df = women_df.withColumn(c, col(c).cast(DoubleType()))
women_df = women_df.withColumn("year", col("year").cast(IntegerType()))

women_df = standardize_state(women_df)

# Add total crimes against women
women_df = women_df.withColumn(
    "total_crimes_women",
    coalesce(col("rape_women"), lit(0)) +
    coalesce(col("kidnapping_women"), lit(0)) +
    coalesce(col("dowry_deaths_women"), lit(0)) +
    coalesce(col("assault_women"), lit(0)) +
    coalesce(col("insult_modesty"), lit(0)) +
    coalesce(col("domestic_cruelty"), lit(0)) +
    coalesce(col("importation_girls"), lit(0))
)

women_state = (
    women_df
    .groupBy("state", "year")
    .agg(
        spark_sum("rape_women").alias("rape_women"),
        spark_sum("kidnapping_women").alias("kidnapping_women"),
        spark_sum("dowry_deaths_women").alias("dowry_deaths_women"),
        spark_sum("assault_women").alias("assault_women"),
        spark_sum("insult_modesty").alias("insult_modesty"),
        spark_sum("domestic_cruelty").alias("domestic_cruelty"),
        spark_sum("importation_girls").alias("importation_girls"),
        spark_sum("total_crimes_women").alias("total_crimes_women"),
    )
)

women_state = women_state.orderBy("state", "year")
print(f"Crimes against women: {women_state.count()} rows")
women_state.show(10, False)


# =====================================================================
# DATASET 3: PROPERTY STOLEN & RECOVERED (State-level)
# =====================================================================

print("=" * 60)
print("LOADING DATASET 3: PROPERTY STOLEN & RECOVERED")
print("=" * 60)

df_10 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/10_Property_stolen_and_recovered.csv"
)

df_10 = df_10.withColumnRenamed("Area_Name", "state") \
             .withColumnRenamed("Year", "year")

df_10 = standardize_state(df_10)

for c in ["Cases_Property_Stolen", "Cases_Property_Recovered",
          "Value_of_Property_Stolen", "Value_of_Property_Recovered"]:
    df_10 = df_10.withColumn(c, col(c).cast(DoubleType()))

property_state = (
    df_10
    .groupBy("state", "year")
    .agg(
        spark_sum("Cases_Property_Stolen").alias("cases_property_stolen"),
        spark_sum("Cases_Property_Recovered").alias("cases_property_recovered"),
        spark_sum("Value_of_Property_Stolen").alias("value_stolen"),
        spark_sum("Value_of_Property_Recovered").alias("value_recovered"),
    )
)

property_state = property_state.withColumn(
    "recovery_rate",
    when(col("value_stolen") > 0,
         (col("value_recovered") / col("value_stolen")) * 100
    ).otherwise(0)
)

property_state = property_state.orderBy("state", "year")
print(f"Property data: {property_state.count()} rows")
property_state.show(10, False)


# =====================================================================
# DATASET 4: AUTO THEFT (State-level)
# =====================================================================

print("=" * 60)
print("LOADING DATASET 4: AUTO THEFT")
print("=" * 60)

df_30 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/30_Auto_theft.csv"
)

df_30 = df_30.withColumnRenamed("Area_Name", "state") \
             .withColumnRenamed("Year", "year")

df_30 = standardize_state(df_30)

for c in ["Auto_Theft_Stolen", "Auto_Theft_Recovered"]:
    df_30 = df_30.withColumn(c, when(col(c) == "NULL", lit(None)).otherwise(col(c)).cast(DoubleType()))

auto_state = (
    df_30
    .groupBy("state", "year")
    .agg(
        spark_sum("Auto_Theft_Stolen").alias("auto_theft_stolen"),
        spark_sum("Auto_Theft_Recovered").alias("auto_theft_recovered"),
    )
)

auto_state = auto_state.orderBy("state", "year")
print(f"Auto theft data: {auto_state.count()} rows")


# =====================================================================
# DATASET 5: MURDER VICTIMS BY AGE/SEX (State-level)
# =====================================================================

print("=" * 60)
print("LOADING DATASET 5: MURDER VICTIM DEMOGRAPHICS")
print("=" * 60)

df_32 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/32_Murder_victim_age_sex.csv"
)

df_32 = df_32.withColumnRenamed("Area_Name", "state") \
             .withColumnRenamed("Year", "year")

df_32 = standardize_state(df_32)

for c in ["Victims_Total", "Victims_Upto_18_30_Yrs", "Victims_Upto_30_50_Yrs",
          "Victims_Above_50_Yrs", "Victims_Upto_10_Yrs", "Victims_Upto_15_18_Yrs"]:
    df_32 = df_32.withColumn(c, when(col(c) == "NULL", lit(None)).otherwise(col(c)).cast(DoubleType()))

murder_demo = (
    df_32
    .groupBy("state", "year", "Sub_Group_Name")
    .agg(
        spark_sum("Victims_Total").alias("victims_total"),
        spark_sum("Victims_Upto_18_30_Yrs").alias("victims_18_30"),
        spark_sum("Victims_Upto_30_50_Yrs").alias("victims_30_50"),
        spark_sum("Victims_Above_50_Yrs").alias("victims_above_50"),
    )
)

# Pivot: separate male/female victim counts
murder_male = murder_demo.filter(col("Sub_Group_Name").contains("Male")) \
    .withColumnRenamed("victims_total", "murder_victims_male") \
    .select("state", "year", "murder_victims_male")

murder_female = murder_demo.filter(col("Sub_Group_Name").contains("Female")) \
    .withColumnRenamed("victims_total", "murder_victims_female") \
    .select("state", "year", "murder_victims_female")

murder_state = murder_male.join(murder_female, ["state", "year"], "outer")
murder_state = murder_state.orderBy("state", "year")
print(f"Murder demographics: {murder_state.count()} rows")


# =====================================================================
# DATASET 6: KIDNAPPING PURPOSE (State-level)
# =====================================================================

print("=" * 60)
print("LOADING DATASET 6: KIDNAPPING BY PURPOSE")
print("=" * 60)

df_39 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/39_Specific_purpose_of_kidnapping_and_abduction.csv"
)

df_39 = df_39.withColumnRenamed("Area_Name", "state") \
             .withColumnRenamed("Year", "year")

df_39 = standardize_state(df_39)

for c in ["K_A_Cases_Reported", "K_A_Female_Total", "K_A_Male_Total", "K_A_Grand_Total"]:
    df_39 = df_39.withColumn(c, when(col(c) == "NULL", lit(None)).otherwise(col(c)).cast(DoubleType()))

kidnap_state = (
    df_39
    .groupBy("state", "year")
    .agg(
        spark_sum("K_A_Cases_Reported").alias("kidnap_cases_total"),
        spark_sum("K_A_Female_Total").alias("kidnap_victims_female"),
        spark_sum("K_A_Male_Total").alias("kidnap_victims_male"),
    )
)

kidnap_state = kidnap_state.orderBy("state", "year")
print(f"Kidnapping data: {kidnap_state.count()} rows")


# =====================================================================
# DATASET 7: FIREARMS IN MURDER (State-level)
# =====================================================================

print("=" * 60)
print("LOADING DATASET 7: FIREARMS IN MURDER")
print("=" * 60)

df_34 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/34_Use_of_fire_arms_in_murder_cases.csv"
)

df_34 = df_34.withColumnRenamed("Area_Name", "state") \
             .withColumnRenamed("Year", "year")

df_34 = standardize_state(df_34)

for c in ["Victims_of_Murder_by_Fire_arms", "Victims_of_Murder_by_Licensed_arms"]:
    df_34 = df_34.withColumn(c, col(c).cast(DoubleType()))

firearms_state = (
    df_34
    .groupBy("state", "year")
    .agg(
        spark_sum("Victims_of_Murder_by_Fire_arms").alias("murder_by_firearms"),
        spark_sum("Victims_of_Murder_by_Licensed_arms").alias("murder_by_licensed_arms"),
    )
)

firearms_state = firearms_state.orderBy("state", "year")
print(f"Firearms data: {firearms_state.count()} rows")


# =====================================================================
# DATASET 8: SERIOUS FRAUD (State-level)
# =====================================================================

print("=" * 60)
print("LOADING DATASET 8: SERIOUS FRAUD")
print("=" * 60)

df_31 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    f"{HDFS_INPUT}/31_Serious_fraud.csv"
)

df_31 = df_31.withColumnRenamed("Area_Name", "state") \
             .withColumnRenamed("Year", "year")

df_31 = standardize_state(df_31)

for c in ["Loss_of_Property_1_10_Crores", "Loss_of_Property_10_25_Crores",
          "Loss_of_Property_25_50_Crores", "Loss_of_Property_50_100_Crores",
          "Loss_of_Property_Above_100_Crores"]:
    df_31 = df_31.withColumn(c, col(c).cast(DoubleType()))

fraud_state = (
    df_31
    .groupBy("state", "year")
    .agg(
        spark_sum("Loss_of_Property_1_10_Crores").alias("fraud_1_10cr"),
        spark_sum("Loss_of_Property_10_25_Crores").alias("fraud_10_25cr"),
        spark_sum("Loss_of_Property_25_50_Crores").alias("fraud_25_50cr"),
        spark_sum("Loss_of_Property_50_100_Crores").alias("fraud_50_100cr"),
        spark_sum("Loss_of_Property_Above_100_Crores").alias("fraud_above_100cr"),
    )
)

fraud_state = fraud_state.withColumn(
    "total_fraud_cases",
    coalesce(col("fraud_1_10cr"), lit(0)) +
    coalesce(col("fraud_10_25cr"), lit(0)) +
    coalesce(col("fraud_25_50cr"), lit(0)) +
    coalesce(col("fraud_50_100cr"), lit(0)) +
    coalesce(col("fraud_above_100cr"), lit(0))
)

fraud_state = fraud_state.orderBy("state", "year")
print(f"Fraud data: {fraud_state.count()} rows")


# =====================================================================
# MASTER JOIN: Combine all datasets
# =====================================================================

print("=" * 60)
print("JOINING ALL DATASETS INTO MASTER TABLE")
print("=" * 60)

master = ipc_state

master = master.join(women_state, ["state", "year"], "left")
master = master.join(property_state, ["state", "year"], "left")
master = master.join(auto_state, ["state", "year"], "left")
master = master.join(murder_state, ["state", "year"], "left")
master = master.join(kidnap_state, ["state", "year"], "left")
master = master.join(firearms_state, ["state", "year"], "left")
master = master.join(fraud_state, ["state", "year"], "left")

master = master.orderBy("state", "year")

print(f"\nMaster dataset: {master.count()} rows, {len(master.columns)} columns")
print("\nColumns:", master.columns)
master.show(10, False)


# =====================================================================
# SAVE ALL OUTPUTS
# =====================================================================

print("\nSaving all datasets to HDFS...")

# 1. Master combined dataset
(
    master.coalesce(1).write.mode("overwrite").option("header", "true")
    .csv(f"{HDFS_OUTPUT}/master_crime_data")
)
print("✓ Master crime data saved")

# 2. IPC crimes (backward compatible with old pipeline)
(
    ipc_state.select("state", "year", "total_ipc_crimes")
    .coalesce(1).write.mode("overwrite").option("header", "true")
    .csv(f"{HDFS_OUTPUT}/cleaned_ipc_crime_data")
)
print("✓ IPC crime data saved")

# 3. Crimes against women
(
    women_state.coalesce(1).write.mode("overwrite").option("header", "true")
    .csv(f"{HDFS_OUTPUT}/crimes_against_women")
)
print("✓ Crimes against women saved")

# 4. Property data
(
    property_state.coalesce(1).write.mode("overwrite").option("header", "true")
    .csv(f"{HDFS_OUTPUT}/property_stolen_recovered")
)
print("✓ Property data saved")


# =====================================================================
# STOP SPARK
# =====================================================================

spark.stop()

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print(f"  → 8 datasets loaded and cleaned")
print(f"  → Master table: {len(master.columns)} columns")
print("=" * 60)
