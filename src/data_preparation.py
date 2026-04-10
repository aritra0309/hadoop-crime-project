"""
Phase 0/1 data preparation pipeline for the India Crime Intelligence Platform.

Outputs:
- hdfs://localhost:9000/crime/output/district_master (Parquet)
- hdfs://localhost:9000/crime/output/state_master (Parquet)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

try:
    from src.state_mapping import CANONICAL_TO_GEOJSON, get_geojson_name
    from src.utils import (
        cast_numeric_columns,
        clean_column_names,
        read_csv_from_hdfs,
        standardize_district,
        standardize_state,
    )
except ModuleNotFoundError:
    from state_mapping import CANONICAL_TO_GEOJSON, get_geojson_name
    from utils import (
        cast_numeric_columns,
        clean_column_names,
        read_csv_from_hdfs,
        standardize_district,
        standardize_state,
    )


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

HDFS_INPUT = "hdfs://localhost:9000/crime/input"
HDFS_OUTPUT = "hdfs://localhost:9000/crime/output"


def ensure_directories() -> None:
    for p in [
        BASE_DIR / "src",
        BASE_DIR / "output" / "dashboard_data",
        BASE_DIR / "dashboard",
        BASE_DIR / "dashboard" / "assets",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def upload_csvs_to_hdfs() -> None:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")

    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", "/crime/input"], check=True)
    for csv_file in csv_files:
        subprocess.run(
            ["hdfs", "dfs", "-put", "-f", str(csv_file), "/crime/input/"],
            check=True,
        )


def load_geojson_name1_values() -> List[str]:
    geojson_path = DATA_DIR / "india_states.geojson"
    with geojson_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)
    values = sorted({feature["properties"].get("NAME_1") for feature in geojson["features"]})
    print("\n[Phase 0] GeoJSON NAME_1 values")
    print(f"Count: {len(values)}")
    for name in values:
        print(f"- {name}")
    return values


def read_hdfs_csv(spark: SparkSession, filename: str) -> DataFrame:
    df = read_csv_from_hdfs(spark, f"{HDFS_INPUT}/{filename}")
    return clean_column_names(df)


def as_num(col_name: str) -> F.Column:
    return F.coalesce(F.col(col_name).cast("double"), F.lit(0.0))


def first_existing(df: DataFrame, *col_names: str, default: float = 0.0) -> F.Column:
    existing = [F.col(c) for c in col_names if c in df.columns]
    if not existing:
        return F.lit(default)
    return F.coalesce(*existing)


def with_columns(df: DataFrame, mapping: Dict[str, F.Column]) -> DataFrame:
    for name, expr in mapping.items():
        df = df.withColumn(name, expr)
    return df


def normalize_ipc_district(df: DataFrame) -> DataFrame:
    rename_map = {
        "STATE/UT": "state",
        "States/UTs": "state",
        "DISTRICT": "district",
        "District": "district",
        "YEAR": "year",
        "Year": "year",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    df = with_columns(
        df,
        {
            "murder": first_existing(df, "MURDER", "Murder").cast("double"),
            "attempt_murder": first_existing(df, "ATTEMPT TO MURDER", "Attempt to commit Murder").cast("double"),
            "culpable_homicide": first_existing(
                df,
                "CULPABLE HOMICIDE NOT AMOUNTING TO MURDER",
                "Culpable Homicide not amounting to Murder",
            ).cast("double"),
            "rape": first_existing(df, "RAPE", "Rape").cast("double"),
            "kidnapping": first_existing(
                df,
                "KIDNAPPING & ABDUCTION",
                "Kidnapping & Abduction_Total",
            ).cast("double"),
            "dacoity": first_existing(df, "DACOITY", "Dacoity").cast("double"),
            "robbery": first_existing(df, "ROBBERY", "Robbery").cast("double"),
            "burglary": first_existing(
                df,
                "BURGLARY",
                "Burglary",
                "Criminal Trespass/Burglary",
                "Criminal Trespass or Burglary",
            ).cast("double"),
            "theft": first_existing(df, "THEFT", "Theft").cast("double"),
            "auto_theft": first_existing(df, "AUTO THEFT", "Auto Theft").cast("double"),
            "riots": first_existing(df, "RIOTS", "Riots").cast("double"),
            "cheating": first_existing(df, "CHEATING", "Cheating").cast("double"),
            "criminal_breach_of_trust": first_existing(df, "CRIMINAL BREACH OF TRUST", "Criminal Breach of Trust").cast("double"),
            "arson": first_existing(df, "ARSON", "Arson").cast("double"),
            "dowry_deaths": first_existing(df, "DOWRY DEATHS", "Dowry Deaths").cast("double"),
            "hurt": F.coalesce(
                first_existing(df, "HURT/GREVIOUS HURT", default=None),
                F.when(
                    first_existing(df, "Hurt", default=None).isNotNull() | first_existing(df, "Grievous Hurt", default=None).isNotNull(),
                    first_existing(df, "Hurt") + first_existing(df, "Grievous Hurt"),
                ),
            ).cast("double"),
            "other_ipc": first_existing(df, "OTHER IPC CRIMES", "Other IPC crimes").cast("double"),
            "total_ipc": first_existing(df, "TOTAL IPC CRIMES", "Total Cognizable IPC crimes").cast("double"),
        },
    )

    keep = [
        "state",
        "district",
        "year",
        "murder",
        "attempt_murder",
        "culpable_homicide",
        "rape",
        "kidnapping",
        "dacoity",
        "robbery",
        "burglary",
        "theft",
        "auto_theft",
        "riots",
        "cheating",
        "criminal_breach_of_trust",
        "arson",
        "dowry_deaths",
        "hurt",
        "other_ipc",
        "total_ipc",
    ]
    df = df.select(*keep)

    df = standardize_state(df, "state")
    df = standardize_district(df, "district")
    df = df.withColumn("year", F.col("year").cast("int"))

    for c in keep[3:]:
        df = df.withColumn(c, as_num(c))

    return df.filter(F.col("year").between(2001, 2014))


def normalize_women_district(df: DataFrame) -> DataFrame:
    rename_map = {
        "STATE/UT": "state",
        "States/UTs": "state",
        "DISTRICT": "district",
        "District": "district",
        "YEAR": "year",
        "Year": "year",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    df = with_columns(
        df,
        {
            "rape_women": first_existing(df, "Rape", "RAPE").cast("double"),
            "kidnapping_women": first_existing(
                df,
                "Kidnapping and Abduction",
                "Kidnapping & Abduction_Total",
            ).cast("double"),
            "dowry_deaths_women": first_existing(df, "Dowry Deaths", "DOWRY DEATHS").cast("double"),
            "assault_women": first_existing(
                df,
                "Assault on women with intent to outrage her modesty",
                "Assault on Women with intent to outrage her Modesty_Total",
            ).cast("double"),
            "insult_modesty": first_existing(
                df,
                "Insult to modesty of Women",
                "Insult to the Modesty of Women_Total",
            ).cast("double"),
            "domestic_cruelty": first_existing(
                df,
                "Cruelty by Husband or his Relatives",
                "CRUELTY BY HUSBAND OR HIS RELATIVES",
            ).cast("double"),
        },
    )

    keep = [
        "state",
        "district",
        "year",
        "rape_women",
        "kidnapping_women",
        "dowry_deaths_women",
        "assault_women",
        "insult_modesty",
        "domestic_cruelty",
    ]
    df = df.select(*keep)
    df = standardize_state(df, "state")
    df = standardize_district(df, "district")
    df = df.withColumn("year", F.col("year").cast("int"))
    for c in keep[3:]:
        df = df.withColumn(c, as_num(c))

    df = df.withColumn(
        "total_women",
        as_num("rape_women")
        + as_num("kidnapping_women")
        + as_num("dowry_deaths_women")
        + as_num("assault_women")
        + as_num("insult_modesty")
        + as_num("domestic_cruelty"),
    )

    return df.filter(F.col("year").between(2001, 2014))


def build_district_master(spark: SparkSession) -> DataFrame:
    ipc_files = [
        "01_District_wise_crimes_committed_IPC_2001_2012.csv",
        "01_District_wise_crimes_committed_IPC_2013.csv",
        "01_District_wise_crimes_committed_IPC_2014.csv",
    ]
    women_files = [
        "42_District_wise_crimes_committed_against_women_2001_2012.csv",
        "42_District_wise_crimes_committed_against_women_2013.csv",
        "42_District_wise_crimes_committed_against_women_2014.csv",
    ]

    ipc_df = None
    for f in ipc_files:
        d = normalize_ipc_district(read_hdfs_csv(spark, f))
        ipc_df = d if ipc_df is None else ipc_df.unionByName(d)
    ipc_metric_cols = [c for c in ipc_df.columns if c not in {"state", "district", "year"}]
    ipc_df = ipc_df.groupBy("state", "district", "year").agg(*[F.sum(c).alias(c) for c in ipc_metric_cols])

    women_df = None
    for f in women_files:
        d = normalize_women_district(read_hdfs_csv(spark, f))
        women_df = d if women_df is None else women_df.unionByName(d)
    women_metric_cols = [c for c in women_df.columns if c not in {"state", "district", "year"}]
    women_df = women_df.groupBy("state", "district", "year").agg(*[F.sum(c).alias(c) for c in women_metric_cols])

    join_keys = ["state", "district", "year"]

    unmatched_ipc = ipc_df.join(women_df.select(*join_keys), join_keys, "left_anti")
    unmatched_women = women_df.join(ipc_df.select(*join_keys), join_keys, "left_anti")

    print("\n[Phase 1.1] District join mismatch report")
    print(f"IPC rows without women match: {unmatched_ipc.count()}")
    unmatched_ipc.select(*join_keys).orderBy(*join_keys).show(10, truncate=False)
    print(f"Women rows without IPC match: {unmatched_women.count()}")
    unmatched_women.select(*join_keys).orderBy(*join_keys).show(10, truncate=False)

    district = (
        ipc_df.join(women_df, join_keys, "left")
        .withColumn(
            "total_violent",
            as_num("murder")
            + as_num("attempt_murder")
            + as_num("culpable_homicide")
            + as_num("rape")
            + as_num("kidnapping")
            + as_num("dacoity")
            + as_num("robbery"),
        )
        .withColumn(
            "total_property",
            as_num("burglary")
            + as_num("theft")
            + as_num("auto_theft")
            + as_num("cheating")
            + as_num("criminal_breach_of_trust"),
        )
        .withColumn(
            "total_women",
            as_num("rape_women")
            + as_num("dowry_deaths_women")
            + as_num("kidnapping_women")
            + as_num("domestic_cruelty")
            + as_num("assault_women")
            + as_num("insult_modesty"),
        )
    )

    district = district.select(
        "state",
        "district",
        "year",
        "murder",
        "attempt_murder",
        "culpable_homicide",
        "rape",
        "kidnapping",
        "dacoity",
        "robbery",
        "burglary",
        "theft",
        "auto_theft",
        "riots",
        "cheating",
        "criminal_breach_of_trust",
        "arson",
        "dowry_deaths",
        "hurt",
        "other_ipc",
        "total_ipc",
        "rape_women",
        "kidnapping_women",
        "dowry_deaths_women",
        "assault_women",
        "insult_modesty",
        "domestic_cruelty",
        "total_women",
        "total_violent",
        "total_property",
    )

    district.write.mode("overwrite").parquet(f"{HDFS_OUTPUT}/district_master")
    return district


def _normalize_state_year(df: DataFrame, state_col: str, year_col: str = "Year") -> DataFrame:
    if state_col != "state":
        df = df.withColumnRenamed(state_col, "state")
    if year_col != "year":
        df = df.withColumnRenamed(year_col, "year")
    df = standardize_state(df, "state")
    return df.withColumn("year", F.col("year").cast("int"))


def build_property_metrics(spark: SparkSession) -> DataFrame:
    df = read_hdfs_csv(spark, "10_Property_stolen_and_recovered.csv")
    df = _normalize_state_year(df, "Area_Name", "Year")
    df = cast_numeric_columns(df, {"state", "year", "Group_Name", "Sub_Group_Name"})

    base = (
        df.groupBy("state", "year")
        .agg(
            F.sum("Cases_Property_Stolen").alias("property_cases_stolen"),
            F.sum("Cases_Property_Recovered").alias("property_cases_recovered"),
            F.sum("Value_of_Property_Stolen").alias("value_stolen"),
            F.sum("Value_of_Property_Recovered").alias("value_recovered"),
        )
        .withColumn(
            "recovery_rate",
            F.when(F.col("value_stolen") > 0, F.col("value_recovered") / F.col("value_stolen")).otherwise(F.lit(None)),
        )
    )

    grouped = (
        df.withColumn("group_bucket", F.upper(F.trim(F.col("Group_Name"))))
        .withColumn(
            "group_bucket",
            F.when(F.col("group_bucket").contains("BURGLARY"), F.lit("burglary"))
            .when(F.col("group_bucket").contains("CRIMINAL BREACH"), F.lit("criminal_breach"))
            .when(F.col("group_bucket").contains("DACOITY"), F.lit("dacoity"))
            .when(F.col("group_bucket").contains("ROBBERY"), F.lit("robbery"))
            .when(F.col("group_bucket").contains("THEFT"), F.lit("theft"))
            .when(F.col("group_bucket").contains("OTHER"), F.lit("other"))
            .when(F.col("group_bucket").contains("TOTAL"), F.lit("total"))
            .otherwise(F.lit(None)),
        )
        .filter(F.col("group_bucket").isNotNull())
        .groupBy("state", "year")
        .pivot("group_bucket", ["burglary", "criminal_breach", "dacoity", "robbery", "theft", "other", "total"])
        .agg(F.sum("Cases_Property_Stolen"))
    )

    for c in ["burglary", "criminal_breach", "dacoity", "robbery", "theft", "other", "total"]:
        if c in grouped.columns:
            grouped = grouped.withColumnRenamed(c, f"property_cases_stolen_{c}")

    return base.join(grouped, ["state", "year"], "left")


def build_auto_theft_metrics(spark: SparkSession) -> DataFrame:
    df = read_hdfs_csv(spark, "30_Auto_theft.csv")
    df = _normalize_state_year(df, "Area_Name", "Year")
    df = cast_numeric_columns(df, {"state", "year", "Group_Name", "Sub_Group_Name"})

    base = df.groupBy("state", "year").agg(
        F.sum("Auto_Theft_Stolen").alias("auto_theft_stolen_state"),
        F.sum("Auto_Theft_Recovered").alias("auto_theft_recovered_state"),
    )

    bucketed = (
        df.withColumn("sg", F.upper(F.trim(F.col("Sub_Group_Name"))))
        .withColumn("bucket", F.regexp_replace(F.col("sg"), r"^[0-9]+\\.\\s*", ""))
        .withColumn(
            "bucket",
            F.when(F.col("bucket").contains("MOTOR CYCLES"), F.lit("motor_cycles"))
            .when(F.col("bucket").contains("MOTOR CAR"), F.lit("motor_car"))
            .when(F.col("bucket").contains("BUSES"), F.lit("buses"))
            .when(F.col("bucket").contains("GOODS"), F.lit("goods"))
            .when(F.col("bucket").contains("TOTAL"), F.lit("total"))
            .when(F.col("bucket").contains("OTHER"), F.lit("other"))
            .otherwise(F.lit(None)),
        )
        .filter(F.col("bucket").isNotNull())
        .groupBy("state", "year")
        .pivot("bucket", ["motor_cycles", "motor_car", "buses", "goods", "other", "total"])
        .agg(F.sum("Auto_Theft_Stolen"))
    )

    for c in ["motor_cycles", "motor_car", "buses", "goods", "other", "total"]:
        if c in bucketed.columns:
            bucketed = bucketed.withColumnRenamed(c, f"auto_theft_stolen_{c}")

    return base.join(bucketed, ["state", "year"], "left")


def build_murder_demographics(spark: SparkSession) -> DataFrame:
    df = read_hdfs_csv(spark, "32_Murder_victim_age_sex.csv")
    df = _normalize_state_year(df, "Area_Name", "Year")
    df = cast_numeric_columns(df, {"state", "year", "Group_Name", "Sub_Group_Name"})

    grouped = (
        df.groupBy("state", "year")
        .agg(
            F.sum(F.when(F.upper(F.col("Group_Name")).contains("MALE"), F.col("Victims_Total")).otherwise(F.lit(0))).alias("murder_victims_male"),
            F.sum(F.when(F.upper(F.col("Group_Name")).contains("FEMALE"), F.col("Victims_Total")).otherwise(F.lit(0))).alias("murder_victims_female"),
            F.sum(F.when(F.upper(F.col("Group_Name")).contains("TOTAL"), F.col("Victims_Total")).otherwise(F.lit(0))).alias("murder_victims_total"),
            F.sum(F.when(F.upper(F.col("Group_Name")).contains("TOTAL"), F.col("Victims_Upto_10_Yrs")).otherwise(F.lit(0))).alias("murder_age_upto_10"),
            F.sum(F.when(F.upper(F.col("Group_Name")).contains("TOTAL"), F.col("Victims_Upto_10_15_Yrs")).otherwise(F.lit(0))).alias("murder_age_10_15"),
            F.sum(F.when(F.upper(F.col("Group_Name")).contains("TOTAL"), F.col("Victims_Upto_15_18_Yrs")).otherwise(F.lit(0))).alias("murder_age_15_18"),
            F.sum(F.when(F.upper(F.col("Group_Name")).contains("TOTAL"), F.col("Victims_Upto_18_30_Yrs")).otherwise(F.lit(0))).alias("murder_age_18_30"),
            F.sum(F.when(F.upper(F.col("Group_Name")).contains("TOTAL"), F.col("Victims_Upto_30_50_Yrs")).otherwise(F.lit(0))).alias("murder_age_30_50"),
            F.sum(F.when(F.upper(F.col("Group_Name")).contains("TOTAL"), F.col("Victims_Above_50_Yrs")).otherwise(F.lit(0))).alias("murder_age_above_50"),
        )
    )
    return grouped


def build_culpable_demographics(spark: SparkSession) -> DataFrame:
    df = read_hdfs_csv(spark, "33_CH_not_murder_victim_age_sex.csv")
    df = _normalize_state_year(df, "Area_Name", "Year")
    df = cast_numeric_columns(df, {"state", "year", "Sub_Group_Name"})

    grouped = (
        df.groupBy("state", "year")
        .agg(
            F.sum(F.when(F.col("Sub_Group_Name").startswith("1."), F.col("Victims_Total")).otherwise(F.lit(0))).alias("ch_victims_male"),
            F.sum(F.when(F.col("Sub_Group_Name").startswith("2."), F.col("Victims_Total")).otherwise(F.lit(0))).alias("ch_victims_female"),
            F.sum(F.when(F.col("Sub_Group_Name").startswith("3."), F.col("Victims_Total")).otherwise(F.lit(0))).alias("ch_victims_total"),
            F.sum(F.when(F.col("Sub_Group_Name").startswith("3."), F.col("Victims_Upto_10_Yrs")).otherwise(F.lit(0))).alias("ch_age_upto_10"),
            F.sum(F.when(F.col("Sub_Group_Name").startswith("3."), F.col("Victims_Upto_10_15_Yrs")).otherwise(F.lit(0))).alias("ch_age_10_15"),
            F.sum(F.when(F.col("Sub_Group_Name").startswith("3."), F.col("Victims_Upto_15_18_Yrs")).otherwise(F.lit(0))).alias("ch_age_15_18"),
            F.sum(F.when(F.col("Sub_Group_Name").startswith("3."), F.col("Victims_Upto_18_30_Yrs")).otherwise(F.lit(0))).alias("ch_age_18_30"),
            F.sum(F.when(F.col("Sub_Group_Name").startswith("3."), F.col("Victims_Upto_30_50_Yrs")).otherwise(F.lit(0))).alias("ch_age_30_50"),
            F.sum(F.when(F.col("Sub_Group_Name").startswith("3."), F.col("Victims_Above_50_Yrs")).otherwise(F.lit(0))).alias("ch_age_above_50"),
        )
    )
    return grouped


def build_firearm_metrics(spark: SparkSession) -> DataFrame:
    df = read_hdfs_csv(spark, "34_Use_of_fire_arms_in_murder_cases.csv")
    df = _normalize_state_year(df, "Area_Name", "Year")
    df = cast_numeric_columns(df, {"state", "year"})

    out = df.groupBy("state", "year").agg(
        F.sum("Victims_of_Murder_by_Fire_arms").alias("murder_by_firearms"),
        F.sum("Victims_of_Murder_by_Licensed_arms").alias("murder_by_licensed_arms"),
        F.sum("Victims_of_Murder_by_Un_licensedImprovisedCrudeCountry_made_Arms_Etc").alias("murder_by_unlicensed_arms"),
    )
    out = out.withColumn(
        "unlicensed_rate",
        F.when(F.col("murder_by_firearms") > 0, F.col("murder_by_unlicensed_arms") / F.col("murder_by_firearms")).otherwise(F.lit(None)),
    )
    return out


def build_kidnap_purpose_metrics(spark: SparkSession) -> DataFrame:
    df = read_hdfs_csv(spark, "39_Specific_purpose_of_kidnapping_and_abduction.csv")
    df = _normalize_state_year(df, "Area_Name", "Year")
    df = cast_numeric_columns(df, {"state", "year", "Group_Name", "Sub_Group_Name"})

    df = df.withColumn("purpose_raw", F.regexp_replace(F.trim(F.col("Sub_Group_Name")), r"^[0-9]+\\.\\s*", ""))
    df = df.withColumn("purpose_up", F.upper(F.col("purpose_raw")))
    df = df.withColumn(
        "purpose",
        F.when(F.col("purpose_up").contains("ADOPTION"), F.lit("adoption"))
        .when(F.col("purpose_up").contains("BEGGING"), F.lit("begging"))
        .when(F.col("purpose_up").contains("ILLICIT"), F.lit("illicit_intercourse"))
        .when(F.col("purpose_up").contains("MARRIAGE"), F.lit("marriage"))
        .when(F.col("purpose_up").contains("PROSTITUTION"), F.lit("prostitution"))
        .when(F.col("purpose_up").contains("RANSOM"), F.lit("ransom"))
        .when(F.col("purpose_up").contains("REVENGE"), F.lit("revenge"))
        .when(F.col("purpose_up").contains("SALE"), F.lit("sale"))
        .when(F.col("purpose_up").contains("UNLAWFUL"), F.lit("unlawful_activity"))
        .when(F.col("purpose_up").contains("TOTAL"), F.lit("total"))
        .when(F.col("purpose_up").contains("OTHER"), F.lit("other_purposes"))
        .otherwise(F.lit(None)),
    )

    pivoted = (
        df.filter(F.col("purpose").isNotNull())
        .groupBy("state", "year")
        .pivot(
            "purpose",
            [
                "adoption",
                "begging",
                "illicit_intercourse",
                "marriage",
                "prostitution",
                "ransom",
                "revenge",
                "sale",
                "unlawful_activity",
                "other_purposes",
                "total",
            ],
        )
        .agg(F.sum("K_A_Cases_Reported"))
    )

    for c in [
        "adoption",
        "begging",
        "illicit_intercourse",
        "marriage",
        "prostitution",
        "ransom",
        "revenge",
        "sale",
        "unlawful_activity",
        "other_purposes",
        "total",
    ]:
        if c in pivoted.columns:
            pivoted = pivoted.withColumnRenamed(c, f"kidnap_cases_{c}")

    totals = (
        df.filter(F.col("purpose") == "total")
        .groupBy("state", "year")
        .agg(
            F.sum("K_A_Male_Total").alias("kidnap_male_total"),
            F.sum("K_A_Female_Total").alias("kidnap_female_total"),
        )
    )
    return pivoted.join(totals, ["state", "year"], "left")


def build_fraud_metrics(spark: SparkSession) -> DataFrame:
    df = read_hdfs_csv(spark, "31_Serious_fraud.csv")
    df = _normalize_state_year(df, "Area_Name", "Year")
    df = cast_numeric_columns(df, {"state", "year", "Group_Name", "Sub_Group_Name"})

    out = df.groupBy("state", "year").agg(
        F.sum("Loss_of_Property_1_10_Crores").alias("fraud_1_10_cr"),
        F.sum("Loss_of_Property_10_25_Crores").alias("fraud_10_25_cr"),
        F.sum("Loss_of_Property_25_50_Crores").alias("fraud_25_50_cr"),
        F.sum("Loss_of_Property_50_100_Crores").alias("fraud_50_100_cr"),
        F.sum("Loss_of_Property_Above_100_Crores").alias("fraud_above_100_cr"),
    )
    out = out.withColumn(
        "fraud_total_bracket_sum",
        as_num("fraud_1_10_cr")
        + as_num("fraud_10_25_cr")
        + as_num("fraud_25_50_cr")
        + as_num("fraud_50_100_cr")
        + as_num("fraud_above_100_cr"),
    )
    return out


def _place_old_schema(df: DataFrame) -> DataFrame:
    df = _normalize_state_year(df, "STATE/UT", "YEAR")

    mapping = {
        "residential_dacoity": "RESIDENTIAL PREMISES - Dacoity",
        "residential_robbery": "RESIDENTIAL PREMISES - Robbery",
        "residential_burglary": "RESIDENTIAL PREMISES - Burglary",
        "residential_theft": "RESIDENTIAL PREMISES - Theft",
        "highway_dacoity": "HIGHWAYS - Dacoity",
        "highway_robbery": "HIGHWAYS - Robbery",
        "highway_burglary": "HIGHWAYS - Burglary",
        "highway_theft": "HIGHWAYS - Theft",
        "river_sea_dacoity": "RIVER and SEA - Dacoity",
        "river_sea_robbery": "RIVER and SEA - Robbery",
        "river_sea_burglary": "RIVER and SEA - Burglary",
        "river_sea_theft": "RIVER and SEA - Theft",
        "railway_dacoity": "RAILWAYS - Dacoity",
        "railway_robbery": "RAILWAYS - Robbery",
        "railway_burglary": "RAILWAYS - Burglary",
        "railway_theft": "RAILWAYS - Theft",
        "bank_dacoity": "BANKS - Dacoity",
        "bank_robbery": "BANKS - Robbery",
        "bank_burglary": "BANKS - Burglary",
        "bank_theft": "BANKS - Theft",
        "commercial_dacoity": "COMMERCIAL ESTABLISHMENTS - Dacoity",
        "commercial_robbery": "COMMERCIAL ESTABLISHMENTS - Robbery",
        "commercial_burglary": "COMMERCIAL ESTABLISHMENTS - Burglary",
        "commercial_theft": "COMMERCIAL ESTABLISHMENTS - Theft",
        "other_places_dacoity": "OTHER PLACES - Dacoity",
        "other_places_robbery": "OTHER PLACES - Robbery",
        "other_places_burglary": "OTHER PLACES - Burglary",
        "other_places_theft": "OTHER PLACES - Theft",
    }

    for new_col, old_col in mapping.items():
        df = df.withColumn(new_col, as_num(old_col))

    return df.select("state", "year", *mapping.keys())


def _place_2014_schema(df: DataFrame) -> DataFrame:
    df = _normalize_state_year(df, "States/UTs", "Year")

    mapping = {
        "residential_dacoity": "Residence_Dacoity_Cases reported",
        "residential_robbery": "Residence_Robbery_Cases reported",
        "residential_burglary": "Residence_Burglary_Cases reported",
        "residential_theft": "Residence_Theft_Cases reported",
        "highway_dacoity": "Highways_Dacoity_Cases reported",
        "highway_robbery": "Highways_Robbery_Cases reported",
        "highway_burglary": "Highways_Burglary_Cases reported",
        "highway_theft": "Highways_Theft_Cases reported",
        "river_sea_dacoity": "RiverOrSea_Dacoity_Cases reported",
        "river_sea_robbery": "RiverOrSea_Robbery_Cases reported",
        "river_sea_burglary": "RiverOrSea_Burglary_Cases reported",
        "river_sea_theft": "RiverOrSea_Theft_Cases reported",
        "railway_dacoity": "Railways_Dacoity_Cases reported",
        "railway_robbery": "Railways_Robbery_Cases reported",
        "railway_burglary": "Railways_Burglary_Cases reported",
        "railway_theft": "Railways_Theft_Cases reported",
        "bank_dacoity": "Bank_Dacoity_Cases reported",
        "bank_robbery": "Bank_Robbery_Cases reported",
        "bank_burglary": "Bank_Burglary_Cases reported",
        "bank_theft": "Bank_Theft_Cases reported",
        "commercial_dacoity": "CommEst_Dacoity_Cases reported",
        "commercial_robbery": "CommEst_Robbery_Cases reported",
        "commercial_burglary": "CommEst_Burglary_Cases reported",
        "commercial_theft": "CommEst_Theft_Cases reported",
        "other_places_dacoity": "OtherPlaces_Dacoity_Cases reported",
        "other_places_robbery": "OtherPlaces_Robbery_Cases reported",
        "other_places_burglary": "OtherPlaces_Burglary_Cases reported",
        "other_places_theft": "OtherPlaces_Theft_Cases reported",
    }

    for new_col, old_col in mapping.items():
        df = df.withColumn(new_col, as_num(old_col))

    return df.select("state", "year", *mapping.keys())


def build_place_metrics(spark: SparkSession) -> DataFrame:
    old_1 = _place_old_schema(read_hdfs_csv(spark, "17_Crime_by_place_of_occurrence_2001_2012.csv"))
    old_2 = _place_old_schema(read_hdfs_csv(spark, "17_Crime_by_place_of_occurrence_2013.csv"))
    y2014 = _place_2014_schema(read_hdfs_csv(spark, "17_Crime_by_place_of_occurrence_2014.csv"))

    return old_1.unionByName(old_2).unionByName(y2014).groupBy("state", "year").agg(
        *[F.sum(c).alias(c) for c in old_1.columns if c not in {"state", "year"}]
    )


def build_women_case_pipeline_metrics(spark: SparkSession) -> DataFrame:
    df = read_hdfs_csv(spark, "42_Cases_under_crime_against_women.csv")
    df = _normalize_state_year(df, "Area_Name", "Year")
    df = cast_numeric_columns(df, {"state", "year", "Group_Name", "Sub_Group_Name"})

    df = df.filter(F.upper(F.col("Sub_Group_Name")).contains("TOTAL CRIMES AGAINST WOMEN"))

    out = df.groupBy("state", "year").agg(
        F.sum("Cases_Reported").alias("women_cases_reported"),
        F.sum("Cases_Chargesheeted").alias("women_cases_chargesheeted"),
        F.sum("Cases_Sent_for_Trial").alias("women_cases_sent_for_trial"),
        F.sum("Cases_Convicted").alias("women_cases_convicted"),
        F.sum("Cases_Acquitted_or_Discharged").alias("women_cases_acquitted_or_discharged"),
        F.sum("Cases_Pending_Trial_at_Year_End").alias("women_cases_pending_trial_year_end"),
    )

    out = out.withColumn(
        "chargesheet_rate",
        F.when(F.col("women_cases_reported") > 0, F.col("women_cases_chargesheeted") / F.col("women_cases_reported")).otherwise(F.lit(None)),
    ).withColumn(
        "conviction_rate",
        F.when(F.col("women_cases_sent_for_trial") > 0, F.col("women_cases_convicted") / F.col("women_cases_sent_for_trial")).otherwise(F.lit(None)),
    )

    return out


def build_state_master(spark: SparkSession, district_master: DataFrame) -> Tuple[DataFrame, Dict[str, Tuple[int, int]]]:
    agg_cols = [c for c in district_master.columns if c not in {"state", "district", "year"}]
    state_base = district_master.groupBy("state", "year").agg(*[F.sum(c).alias(c) for c in agg_cols])

    supplementary = {
        "property_10": build_property_metrics(spark),
        "auto_30": build_auto_theft_metrics(spark),
        "murder_32": build_murder_demographics(spark),
        "culpable_33": build_culpable_demographics(spark),
        "firearms_34": build_firearm_metrics(spark),
        "kidnap_39": build_kidnap_purpose_metrics(spark),
        "fraud_31": build_fraud_metrics(spark),
        "place_17": build_place_metrics(spark),
        "women_cases_42": build_women_case_pipeline_metrics(spark),
    }

    coverage = {}
    for name, df in supplementary.items():
        bounds = df.agg(F.min("year").alias("min_year"), F.max("year").alias("max_year")).collect()[0]
        coverage[name] = (bounds["min_year"], bounds["max_year"])

    state_master = state_base
    for df in supplementary.values():
        state_master = state_master.join(df, ["state", "year"], "left")

    state_master.write.mode("overwrite").parquet(f"{HDFS_OUTPUT}/state_master")
    return state_master, coverage


def print_null_rates(df: DataFrame, name: str) -> None:
    total = df.count()
    if total == 0:
        print(f"[Validation] {name}: empty dataframe")
        return

    print(f"\n[Validation] Null rates for {name} (rows={total})")
    exprs = [
        (F.sum(F.when(F.col(c).isNull(), F.lit(1)).otherwise(F.lit(0))) / F.lit(total)).alias(c)
        for c in df.columns
    ]
    row = df.agg(*exprs).collect()[0].asDict()
    for c in df.columns:
        pct = float(row[c]) * 100
        flag = " <-- HIGH NULL (>20%)" if pct > 20 else ""
        print(f"- {c}: {pct:.2f}%{flag}")


def print_validation_report(
    district_master: DataFrame,
    state_master: DataFrame,
    geojson_name1_values: Iterable[str],
    supplementary_coverage: Dict[str, Tuple[int, int]],
) -> None:
    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)

    dup_count = (
        district_master.groupBy("state", "district", "year")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    print(f"[1] Duplicate (state,district,year) rows in district_master: {dup_count}")

    leak_district = district_master.filter(
        F.upper(F.col("state")).rlike("TOTAL|ALL INDIA|ALL-INDIA|STATES|UTS")
        | F.upper(F.col("district")).rlike("TOTAL|ALL INDIA|ALL-INDIA")
    ).count()
    leak_state = state_master.filter(F.upper(F.col("state")).rlike("TOTAL|ALL INDIA|ALL-INDIA|STATES|UTS")).count()
    print(f"[2] Aggregate-row leakage in district_master: {leak_district}")
    print(f"[2] Aggregate-row leakage in state_master: {leak_state}")

    geojson_set = set(geojson_name1_values)
    canonical_states = {r["state"] for r in state_master.select("state").distinct().collect()}

    missing = []
    for canonical in sorted(canonical_states):
        geo_name = get_geojson_name(canonical)
        if geo_name not in geojson_set:
            missing.append((canonical, geo_name))

    print(f"[3] Canonical states in master: {len(canonical_states)}")
    print(f"[3] GeoJSON NAME_1 values: {len(geojson_set)}")
    if missing:
        print("[3] Missing canonical->GeoJSON matches:")
        for c, g in missing:
            print(f"- {c} -> {g}")
    else:
        print("[3] Canonical->GeoJSON matching: OK")

    extra_map = sorted(set(CANONICAL_TO_GEOJSON.values()) - geojson_set)
    if extra_map:
        print("[3] WARNING: CANONICAL_TO_GEOJSON values not found in GeoJSON:")
        for n in extra_map:
            print(f"- {n}")

    print_null_rates(district_master, "district_master")
    print_null_rates(state_master, "state_master")

    district_bounds = district_master.agg(F.min("year").alias("min_year"), F.max("year").alias("max_year")).collect()[0]
    state_bounds = state_master.agg(F.min("year").alias("min_year"), F.max("year").alias("max_year")).collect()[0]
    print("\n[5] Year coverage")
    print(f"- district_master: {district_bounds['min_year']} to {district_bounds['max_year']}")
    print(f"- state_master base: {state_bounds['min_year']} to {state_bounds['max_year']}")
    print("- supplementary datasets (expected 2001-2010):")
    for name, (mn, mx) in supplementary_coverage.items():
        print(f"  - {name}: {mn} to {mx}")


def main() -> None:
    spark = (
        SparkSession.builder.appName("Crime Data Preparation Phase0-1")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        ensure_directories()
        upload_csvs_to_hdfs()
        geojson_name1_values = load_geojson_name1_values()

        district_master = build_district_master(spark)
        state_master, supplementary_coverage = build_state_master(spark, district_master)

        print_validation_report(
            district_master,
            state_master,
            geojson_name1_values,
            supplementary_coverage,
        )

        print("\nPipeline complete")
        print(f"- district_master: {HDFS_OUTPUT}/district_master")
        print(f"- state_master: {HDFS_OUTPUT}/state_master")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
