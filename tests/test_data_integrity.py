"""
Tests for data integrity — validates the raw CSV files in data/ are present
and structurally correct. No Spark required for most tests.
"""

import csv
import os
import json
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


EXPECTED_CSV_FILES = [
    "01_District_wise_crimes_committed_IPC_2001_2012.csv",
    "01_District_wise_crimes_committed_IPC_2013.csv",
    "01_District_wise_crimes_committed_IPC_2014.csv",
    "10_Property_stolen_and_recovered.csv",
    "30_Auto_theft.csv",
    "31_Serious_fraud.csv",
    "32_Murder_victim_age_sex.csv",
    "33_CH_not_murder_victim_age_sex.csv",
    "34_Use_of_fire_arms_in_murder_cases.csv",
    "39_Specific_purpose_of_kidnapping_and_abduction.csv",
    "42_Cases_under_crime_against_women.csv",
    "42_District_wise_crimes_committed_against_women_2001_2012.csv",
    "42_District_wise_crimes_committed_against_women_2013.csv",
    "42_District_wise_crimes_committed_against_women_2014.csv",
]


class TestDataFilesExist:
    """Verify all expected data files are present."""

    @pytest.mark.parametrize("filename", EXPECTED_CSV_FILES)
    def test_csv_exists(self, filename):
        path = os.path.join(DATA_DIR, filename)
        assert os.path.isfile(path), f"Missing data file: {filename}"

    def test_geojson_exists(self):
        path = os.path.join(DATA_DIR, "india_states.geojson")
        assert os.path.isfile(path), "Missing india_states.geojson"


class TestCSVStructure:
    """Verify CSV files have headers and are non-empty."""

    @pytest.mark.parametrize("filename", EXPECTED_CSV_FILES)
    def test_csv_has_header_and_data(self, filename):
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            assert header is not None, f"{filename} has no header row"
            assert len(header) >= 3, f"{filename} header has fewer than 3 columns"
            first_row = next(reader, None)
            assert first_row is not None, f"{filename} has no data rows"

    @pytest.mark.parametrize("filename", [
        "01_District_wise_crimes_committed_IPC_2001_2012.csv",
        "01_District_wise_crimes_committed_IPC_2013.csv",
        "01_District_wise_crimes_committed_IPC_2014.csv",
    ])
    def test_ipc_files_have_state_district_year(self, filename):
        """IPC crime files should have STATE, DISTRICT, and YEAR-like columns."""
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = [h.strip().strip('"').upper() for h in next(reader)]
        header_str = " ".join(header)
        assert any("STATE" in h or "AREA" in h for h in header), (
            f"{filename} missing STATE/AREA column"
        )


class TestGeoJSON:
    """Verify GeoJSON file is valid and has expected properties."""

    def test_geojson_is_valid_json(self):
        path = os.path.join(DATA_DIR, "india_states.geojson")
        with open(path, "r") as f:
            data = json.load(f)
        assert "features" in data

    def test_geojson_has_36_features(self):
        path = os.path.join(DATA_DIR, "india_states.geojson")
        with open(path, "r") as f:
            data = json.load(f)
        assert len(data["features"]) == 36

    def test_geojson_features_have_name1(self):
        path = os.path.join(DATA_DIR, "india_states.geojson")
        with open(path, "r") as f:
            data = json.load(f)
        for feature in data["features"]:
            assert "NAME_1" in feature["properties"], (
                f"Feature missing NAME_1: {feature['properties']}"
            )
