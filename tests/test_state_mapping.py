"""
Tests for state_mapping.py — canonical state names, GeoJSON mapping, and filtering patterns.
"""

import pytest
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.state_mapping import (
    STATE_NAME_MAP,
    CANONICAL_STATES,
    CANONICAL_TO_GEOJSON,
    GEOJSON_NAME1_VALUES,
    AGGREGATE_STATE_PATTERNS,
    AGGREGATE_DISTRICT_PATTERNS,
    get_geojson_name,
)


class TestStateNameMap:
    """Tests for the STATE_NAME_MAP dictionary."""

    def test_map_is_nonempty(self):
        assert len(STATE_NAME_MAP) > 0

    def test_all_keys_are_uppercase(self):
        for key in STATE_NAME_MAP:
            assert key == key.upper(), f"Key '{key}' is not uppercase"

    def test_all_values_are_uppercase(self):
        for val in STATE_NAME_MAP.values():
            assert val == val.upper(), f"Value '{val}' is not uppercase"

    def test_all_values_are_canonical(self):
        """Every mapped value should be in the canonical states list."""
        for raw, canonical in STATE_NAME_MAP.items():
            assert canonical in CANONICAL_STATES, (
                f"'{raw}' maps to '{canonical}' which is not in CANONICAL_STATES"
            )

    def test_known_aliases(self):
        """Spot-check known variant → canonical mappings."""
        assert STATE_NAME_MAP["A & N ISLANDS"] == "ANDAMAN & NICOBAR ISLANDS"
        assert STATE_NAME_MAP["DELHI UT"] == "DELHI"
        assert STATE_NAME_MAP["ORISSA"] == "ODISHA"
        assert STATE_NAME_MAP["UTTARANCHAL"] == "UTTARAKHAND"
        assert STATE_NAME_MAP["PONDICHERRY"] == "PUDUCHERRY"
        assert STATE_NAME_MAP["CHATTISGARH"] == "CHHATTISGARH"

    def test_no_canonical_maps_to_itself(self):
        """Canonical names should NOT appear as keys (they pass through unchanged)."""
        for canonical in CANONICAL_STATES:
            assert canonical not in STATE_NAME_MAP, (
                f"Canonical name '{canonical}' should not be a key in STATE_NAME_MAP"
            )


class TestCanonicalStates:
    """Tests for the CANONICAL_STATES list."""

    def test_count_is_36(self):
        assert len(CANONICAL_STATES) == 37

    def test_no_duplicates(self):
        assert len(CANONICAL_STATES) == len(set(CANONICAL_STATES))

    def test_all_uppercase(self):
        for s in CANONICAL_STATES:
            assert s == s.upper()

    def test_contains_telangana(self):
        """Telangana was formed in 2014 and should be included."""
        assert "TELANGANA" in CANONICAL_STATES

    def test_contains_major_states(self):
        for state in ["MAHARASHTRA", "UTTAR PRADESH", "TAMIL NADU", "KERALA", "DELHI"]:
            assert state in CANONICAL_STATES


class TestCanonicalToGeojson:
    """Tests for the CANONICAL_TO_GEOJSON mapping."""

    def test_all_keys_are_canonical(self):
        for key in CANONICAL_TO_GEOJSON:
            assert key in CANONICAL_STATES, f"'{key}' not in CANONICAL_STATES"

    def test_all_values_in_geojson_list(self):
        for val in CANONICAL_TO_GEOJSON.values():
            assert val in GEOJSON_NAME1_VALUES, f"'{val}' not in GEOJSON_NAME1_VALUES"

    def test_odisha_maps_to_orissa(self):
        assert CANONICAL_TO_GEOJSON["ODISHA"] == "Orissa"

    def test_uttarakhand_maps_to_uttaranchal(self):
        assert CANONICAL_TO_GEOJSON["UTTARAKHAND"] == "Uttaranchal"


class TestGetGeojsonName:
    """Tests for the get_geojson_name() function."""

    def test_mapped_state(self):
        assert get_geojson_name("ODISHA") == "Orissa"
        assert get_geojson_name("UTTARAKHAND") == "Uttaranchal"

    def test_unmapped_state_returns_title_case(self):
        assert get_geojson_name("MAHARASHTRA") == "Maharashtra"
        assert get_geojson_name("TAMIL NADU") == "Tamil Nadu"

    def test_delhi(self):
        assert get_geojson_name("DELHI") == "Delhi"


class TestAggregatePatterns:
    """Tests for the aggregate row filter patterns."""

    def test_state_patterns_nonempty(self):
        assert len(AGGREGATE_STATE_PATTERNS) > 0

    def test_district_patterns_nonempty(self):
        assert len(AGGREGATE_DISTRICT_PATTERNS) > 0

    def test_total_all_india_in_state_patterns(self):
        upper_patterns = [p.upper() for p in AGGREGATE_STATE_PATTERNS]
        assert "TOTAL (ALL-INDIA)" in upper_patterns
        assert "TOTAL" in upper_patterns

    def test_total_in_district_patterns(self):
        assert "TOTAL" in AGGREGATE_DISTRICT_PATTERNS


class TestGeojsonNameValues:
    """Tests for the GEOJSON_NAME1_VALUES list."""

    def test_count_is_36(self):
        assert len(GEOJSON_NAME1_VALUES) == 36

    def test_no_duplicates(self):
        assert len(GEOJSON_NAME1_VALUES) == len(set(GEOJSON_NAME1_VALUES))
