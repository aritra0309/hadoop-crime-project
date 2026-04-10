"""
Shared utilities for the India Crime Intelligence Platform PySpark pipeline.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, upper, trim, regexp_replace, when, lit, coalesce, create_map, element_at
)
from pyspark.sql.types import IntegerType, LongType, DoubleType

try:
    from src.state_mapping import STATE_NAME_MAP, AGGREGATE_STATE_PATTERNS, AGGREGATE_DISTRICT_PATTERNS
except ModuleNotFoundError:
    from state_mapping import STATE_NAME_MAP, AGGREGATE_STATE_PATTERNS, AGGREGATE_DISTRICT_PATTERNS


def standardize_state(df, state_col="state"):
    """
    Standardize state names using the authoritative STATE_NAME_MAP.

    1. Uppercase, trim, strip quotes, strip BOM
    2. Normalize ampersand spacing (e.g. "A&N" -> "A & N", "A  &  N" -> "A & N")
    3. Apply STATE_NAME_MAP lookups
    4. Filter out aggregate/header rows
    """
    # Clean the raw state column: strip BOM, quotes, whitespace, carriage returns
    cleaned = upper(trim(regexp_replace(
        regexp_replace(col(state_col), '[\ufeff"\r]', ''),
        r'\s+', ' '
    )))

    # Normalize ampersand: ensure " & " with single spaces
    # First add spaces around & if missing, then collapse multiple spaces
    cleaned_amp = regexp_replace(
        regexp_replace(cleaned, r'&', ' & '),
        r'\s+', ' '
    )
    cleaned_amp = trim(cleaned_amp)

    # Resolve aliases via a Spark map lookup (avoids deep CASE WHEN trees)
    mapping_items = []
    for raw, canonical in STATE_NAME_MAP.items():
        mapping_items.extend([lit(raw), lit(canonical)])
    mapping_expr = create_map(*mapping_items)
    df = df.withColumn(state_col, coalesce(element_at(mapping_expr, cleaned_amp), cleaned_amp))

    # Filter out aggregate rows based on state patterns (case-insensitive)
    for pattern in AGGREGATE_STATE_PATTERNS:
        df = df.filter(upper(trim(col(state_col))) != lit(pattern.upper()))

    # Also filter nulls
    df = df.filter(col(state_col).isNotNull())

    return df


def standardize_district(df, district_col="district"):
    """
    Uppercase, trim, strip quotes from district names.
    Filter out aggregate/total district rows.
    """
    # Clean: strip quotes, BOM, carriage returns, collapse whitespace
    df = df.withColumn(
        district_col,
        upper(trim(regexp_replace(
            regexp_replace(col(district_col), '[\ufeff"\r]', ''),
            r'\s+', ' '
        )))
    )

    # Filter out aggregate district patterns
    for pattern in AGGREGATE_DISTRICT_PATTERNS:
        df = df.filter(upper(trim(col(district_col))) != lit(pattern))

    # Filter nulls and empty strings
    df = df.filter(col(district_col).isNotNull())
    df = df.filter(col(district_col) != lit(""))

    return df


def safe_select(df, columns, fill_value=0):
    """
    Select columns from a DataFrame, filling missing columns with fill_value.
    Columns that exist are kept as-is; missing columns are added as literals.
    """
    existing_cols = set(df.columns)
    select_exprs = []
    for c in columns:
        if c in existing_cols:
            select_exprs.append(col(c))
        else:
            select_exprs.append(lit(fill_value).alias(c))
    return df.select(select_exprs)


def cast_numeric_columns(df, exclude_cols=None):
    """
    Cast all columns except exclude_cols to LongType.
    Handles string 'NULL', 'NA', '' values by converting them to null first.
    """
    if exclude_cols is None:
        exclude_cols = set()
    else:
        exclude_cols = set(exclude_cols)

    for c in df.columns:
        if c not in exclude_cols:
            # Replace common null-like strings with actual null
            df = df.withColumn(
                c,
                when(
                    upper(trim(col(c))).isin("NULL", "NA", "", "-"),
                    lit(None)
                ).otherwise(col(c)).cast(LongType())
            )
    return df


def read_csv_from_hdfs(spark, path, header=True, infer_schema=False):
    """
    Read a CSV from HDFS with common options for this project.
    Handles BOM-prefixed files and quoted headers.
    """
    return (
        spark.read
        .option("header", header)
        .option("inferSchema", infer_schema)
        .option("encoding", "UTF-8")
        .option("quote", '"')
        .option("escape", '"')
        .csv(path)
    )


def clean_column_names(df):
    """
    Strip BOM, quotes, leading/trailing whitespace, and carriage returns
    from all column names.
    """
    for old_name in df.columns:
        new_name = old_name.strip().strip('\ufeff').strip('"').strip().rstrip('\r')
        if new_name != old_name:
            df = df.withColumnRenamed(old_name, new_name)
    return df
