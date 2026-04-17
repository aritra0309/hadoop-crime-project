"""
Tests for utils.py — PySpark utility functions.
Requires a local SparkSession (provided by conftest.py).
"""

import pytest
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType
)

from src.utils import (
    standardize_state,
    standardize_district,
    safe_select,
    cast_numeric_columns,
    clean_column_names,
)


class TestStandardizeState:
    """Tests for standardize_state()."""

    def test_maps_known_alias(self, spark):
        df = spark.createDataFrame([("A & N ISLANDS",)], ["state"])
        result = standardize_state(df, "state").collect()
        assert len(result) == 1
        assert result[0]["state"] == "ANDAMAN & NICOBAR ISLANDS"

    def test_maps_delhi_ut(self, spark):
        df = spark.createDataFrame([("Delhi UT",)], ["state"])
        result = standardize_state(df, "state").collect()
        assert len(result) == 1
        assert result[0]["state"] == "DELHI"

    def test_maps_orissa_to_odisha(self, spark):
        df = spark.createDataFrame([("orissa",)], ["state"])
        result = standardize_state(df, "state").collect()
        assert len(result) == 1
        assert result[0]["state"] == "ODISHA"

    def test_filters_aggregate_total_rows(self, spark):
        df = spark.createDataFrame(
            [("MAHARASHTRA",), ("TOTAL (ALL-INDIA)",), ("TOTAL",)],
            ["state"]
        )
        result = standardize_state(df, "state").collect()
        states = [r["state"] for r in result]
        assert "MAHARASHTRA" in states
        assert "TOTAL (ALL-INDIA)" not in states
        assert "TOTAL" not in states

    def test_filters_null_states(self, spark):
        df = spark.createDataFrame(
            [("KERALA",), (None,)],
            schema=StructType([StructField("state", StringType())])
        )
        result = standardize_state(df, "state").collect()
        assert len(result) == 1
        assert result[0]["state"] == "KERALA"

    def test_strips_quotes_and_bom(self, spark):
        df = spark.createDataFrame([('\ufeff"MAHARASHTRA"',)], ["state"])
        result = standardize_state(df, "state").collect()
        assert result[0]["state"] == "MAHARASHTRA"

    def test_handles_extra_whitespace(self, spark):
        df = spark.createDataFrame([("  TAMIL   NADU  ",)], ["state"])
        result = standardize_state(df, "state").collect()
        assert result[0]["state"] == "TAMIL NADU"

    def test_passthrough_canonical_name(self, spark):
        df = spark.createDataFrame([("WEST BENGAL",)], ["state"])
        result = standardize_state(df, "state").collect()
        assert result[0]["state"] == "WEST BENGAL"


class TestStandardizeDistrict:
    """Tests for standardize_district()."""

    def test_uppercases_district(self, spark):
        df = spark.createDataFrame([("Mumbai",)], ["district"])
        result = standardize_district(df, "district").collect()
        assert result[0]["district"] == "MUMBAI"

    def test_filters_total_rows(self, spark):
        df = spark.createDataFrame(
            [("MUMBAI",), ("TOTAL",), ("ZZ TOTAL",)],
            ["district"]
        )
        result = standardize_district(df, "district").collect()
        districts = [r["district"] for r in result]
        assert "MUMBAI" in districts
        assert "TOTAL" not in districts
        assert "ZZ TOTAL" not in districts

    def test_filters_null_and_empty(self, spark):
        df = spark.createDataFrame(
            [("PUNE",), (None,), ("",)],
            schema=StructType([StructField("district", StringType())])
        )
        result = standardize_district(df, "district").collect()
        assert len(result) == 1
        assert result[0]["district"] == "PUNE"


class TestSafeSelect:
    """Tests for safe_select()."""

    def test_existing_columns_preserved(self, spark):
        df = spark.createDataFrame([(1, 2)], ["a", "b"])
        result = safe_select(df, ["a", "b"]).collect()
        assert result[0]["a"] == 1
        assert result[0]["b"] == 2

    def test_missing_columns_filled_with_default(self, spark):
        df = spark.createDataFrame([(1,)], ["a"])
        result = safe_select(df, ["a", "missing_col"], fill_value=0).collect()
        assert result[0]["a"] == 1
        assert result[0]["missing_col"] == 0

    def test_custom_fill_value(self, spark):
        df = spark.createDataFrame([(1,)], ["a"])
        result = safe_select(df, ["a", "x"], fill_value=-1).collect()
        assert result[0]["x"] == -1


class TestCastNumericColumns:
    """Tests for cast_numeric_columns()."""

    def test_casts_string_numbers(self, spark):
        df = spark.createDataFrame([("DELHI", "100", "200")], ["state", "col1", "col2"])
        result = cast_numeric_columns(df, exclude_cols=["state"]).collect()
        assert result[0]["col1"] == 100
        assert result[0]["col2"] == 200

    def test_excludes_specified_columns(self, spark):
        df = spark.createDataFrame([("DELHI", "100")], ["state", "col1"])
        result = cast_numeric_columns(df, exclude_cols=["state"]).collect()
        assert isinstance(result[0]["state"], str)

    def test_null_like_strings_become_none(self, spark):
        df = spark.createDataFrame([("DELHI", "NULL", "NA", "-", "")], ["state", "a", "b", "c", "d"])
        result = cast_numeric_columns(df, exclude_cols=["state"]).collect()
        assert result[0]["a"] is None
        assert result[0]["b"] is None
        assert result[0]["c"] is None
        assert result[0]["d"] is None


class TestCleanColumnNames:
    """Tests for clean_column_names()."""

    def test_strips_bom_from_column_names(self, spark):
        df = spark.createDataFrame([(1,)], ['\ufeffSTATE'])
        result = clean_column_names(df)
        assert "STATE" in result.columns

    def test_strips_quotes_from_column_names(self, spark):
        df = spark.createDataFrame([(1,)], ['"YEAR"'])
        result = clean_column_names(df)
        assert "YEAR" in result.columns

    def test_strips_whitespace_from_column_names(self, spark):
        df = spark.createDataFrame([(1,)], ["  MURDER  "])
        result = clean_column_names(df)
        assert "MURDER" in result.columns
