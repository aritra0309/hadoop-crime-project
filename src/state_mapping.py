"""
Authoritative state-name normalization for the India Crime Intelligence Platform.

Design Decision:
    Canonical names use modern Indian government names (e.g., ODISHA not ORISSA,
    UTTARAKHAND not UTTARANCHAL, ANDAMAN & NICOBAR ISLANDS not ANDAMAN AND NICOBAR).
    A separate CANONICAL_TO_GEOJSON dict handles the names that differ in the
    GeoJSON file (india_states.geojson uses NAME_1 property with older/alternate names).

Sources of state-name variants:
    - IPC 2001-2012/2013: ALL CAPS, e.g. "A & N ISLANDS", "DELHI UT"
    - IPC 2014 / Women 2014: Title Case, e.g. "A&N Islands", "Delhi UT"
    - Women 2001-2012: Quoted ALL CAPS, e.g. '"DELHI"' (not "DELHI UT")
    - Supplementary (10,30,31,32,33,34,39,42_Cases): Title Case, e.g.
      "Andaman & Nicobar Islands", "Delhi", "Dadra & Nagar Haveli"
    - 17_Place 2001-2012: Quoted ALL CAPS with "A & N ISLANDS", "DELHI"
    - 17_Place 2013: ALL CAPS with "A & N ISLANDS", "DELHI"
    - 17_Place 2014: Title Case with "A & N Islands", "Delhi UT"
    - GeoJSON NAME_1: Title Case with older names, e.g. "Orissa", "Uttaranchal",
      "Andaman and Nicobar"
"""

# =========================================================================
# RAW-TO-CANONICAL MAPPING
# =========================================================================
# Maps every observed state/UT name variant (uppercased) to the canonical form.
# Before lookup, apply: upper().strip().replace('"', '')
#
# Canonical names are ALL CAPS modern Indian government names.
# States/UTs that already match their canonical form after uppercasing are
# NOT listed here — they pass through unchanged.

STATE_NAME_MAP = {
    # Andaman & Nicobar Islands variants
    "A & N ISLANDS":                "ANDAMAN & NICOBAR ISLANDS",
    "A&N ISLANDS":                  "ANDAMAN & NICOBAR ISLANDS",
    "ANDAMAN & NICOBAR":            "ANDAMAN & NICOBAR ISLANDS",
    "ANDAMAN AND NICOBAR":          "ANDAMAN & NICOBAR ISLANDS",  # GeoJSON
    "ANDAMAN AND NICOBAR ISLANDS":  "ANDAMAN & NICOBAR ISLANDS",

    # Dadra & Nagar Haveli variants
    "D & N HAVELI":                 "DADRA & NAGAR HAVELI",
    "D&N HAVELI":                   "DADRA & NAGAR HAVELI",
    "DADRA AND NAGAR HAVELI":       "DADRA & NAGAR HAVELI",  # GeoJSON

    # Daman & Diu variants
    "DAMAN AND DIU":                "DAMAN & DIU",  # GeoJSON

    # Delhi variants
    "DELHI UT":                     "DELHI",
    "NCT OF DELHI":                 "DELHI",

    # Jammu & Kashmir variants
    "JAMMU AND KASHMIR":            "JAMMU & KASHMIR",  # GeoJSON

    # Odisha variants
    "ORISSA":                       "ODISHA",  # GeoJSON uses "Orissa"

    # Puducherry variants
    "PONDICHERRY":                  "PUDUCHERRY",

    # Uttarakhand variants
    "UTTARANCHAL":                  "UTTARAKHAND",  # GeoJSON uses "Uttaranchal"

    # Chhattisgarh variants
    "CHATTISGARH":                  "CHHATTISGARH",
}


# =========================================================================
# CANONICAL STATE LIST
# =========================================================================
# All 36 states/UTs that appear in the dataset (35 pre-2014 + Telangana from 2014).

CANONICAL_STATES = [
    "ANDAMAN & NICOBAR ISLANDS",
    "ANDHRA PRADESH",
    "ARUNACHAL PRADESH",
    "ASSAM",
    "BIHAR",
    "CHANDIGARH",
    "CHHATTISGARH",
    "DADRA & NAGAR HAVELI",
    "DAMAN & DIU",
    "DELHI",
    "GOA",
    "GUJARAT",
    "HARYANA",
    "HIMACHAL PRADESH",
    "JAMMU & KASHMIR",
    "JHARKHAND",
    "KARNATAKA",
    "KERALA",
    "LAKSHADWEEP",
    "MADHYA PRADESH",
    "MAHARASHTRA",
    "MANIPUR",
    "MEGHALAYA",
    "MIZORAM",
    "NAGALAND",
    "ODISHA",
    "PUDUCHERRY",
    "PUNJAB",
    "RAJASTHAN",
    "SIKKIM",
    "TAMIL NADU",
    "TELANGANA",
    "TRIPURA",
    "UTTAR PRADESH",
    "UTTARAKHAND",
    "WEST BENGAL",
]


# =========================================================================
# CANONICAL → GEOJSON NAME_1 MAPPING
# =========================================================================
# Maps canonical state names to their GeoJSON NAME_1 equivalents.
# GeoJSON uses "and" instead of "&", plus older names for three states.
# All other canonical names match their GeoJSON NAME_1 in title case.

CANONICAL_TO_GEOJSON = {
    "ANDAMAN & NICOBAR ISLANDS": "Andaman and Nicobar",   # GeoJSON drops "Islands", uses "and"
    "DADRA & NAGAR HAVELI":      "Dadra and Nagar Haveli", # GeoJSON uses "and" not "&"
    "DAMAN & DIU":               "Daman and Diu",          # GeoJSON uses "and" not "&"
    "JAMMU & KASHMIR":           "Jammu and Kashmir",      # GeoJSON uses "and" not "&"
    "ODISHA":                    "Orissa",                  # GeoJSON uses old name
    "UTTARAKHAND":               "Uttaranchal",             # GeoJSON uses old name
}


# =========================================================================
# GEOJSON NAME_1 VALUES (for reference/validation)
# =========================================================================

GEOJSON_NAME1_VALUES = [
    "Andaman and Nicobar",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Chhattisgarh",
    "Dadra and Nagar Haveli",
    "Daman and Diu",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu and Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Lakshadweep",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Orissa",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttaranchal",
    "West Bengal",
]


# =========================================================================
# AGGREGATE ROW PATTERNS TO FILTER OUT
# =========================================================================
# State-name patterns (uppercased) that indicate aggregate/total rows.
# Applied after uppercasing. Uses exact match.

AGGREGATE_STATE_PATTERNS = [
    "TOTAL (ALL-INDIA)",
    "TOTAL (ALL INDIA)",
    "TOTAL (STATES)",
    "TOTAL (STATE)",
    "TOTAL (UTS)",
    "TOTAL",
    "ALL INDIA",
    "ALL-INDIA",
    "STATES/UTS",           # header row that leaks through
    "STATE/UT",             # header row that leaks through
    "AREA_NAME",            # BOM-prefixed header row
    "\ufeffAREA_NAME",          # BOM-prefixed header row (literal BOM)
]

# District-name patterns (uppercased) that indicate aggregate/total rows.
AGGREGATE_DISTRICT_PATTERNS = [
    "TOTAL",
    "TOTAL DISTRICT(S)",
    "TOTAL (ALL-INDIA)",
    "ALL INDIA",
    "DELHI UT TOTAL",
    "ZZ TOTAL",
    "STATE TOTAL",
]


def get_geojson_name(canonical_state):
    """Convert a canonical state name to its GeoJSON NAME_1 equivalent."""
    if canonical_state in CANONICAL_TO_GEOJSON:
        return CANONICAL_TO_GEOJSON[canonical_state]
    # Default: title case the canonical name
    return canonical_state.title()


def get_state_name_mapping():
    """Return the raw-to-canonical state name mapping dict."""
    return STATE_NAME_MAP


# =========================================================================
# YAML-BASED LOADING (v1.3)
# =========================================================================
# Optionally loads state mappings from state_mappings.yaml if it exists.
# Falls back to the hardcoded values above for backward compatibility.

def _load_yaml_mappings():
    """
    Attempt to load state mappings from state_mappings.yaml.
    Returns (state_name_map, canonical_states, canonical_to_geojson,
             aggregate_state_patterns, aggregate_district_patterns)
    or None if YAML is not available.
    """
    try:
        import yaml
    except ImportError:
        return None

    from pathlib import Path
    yaml_path = Path(__file__).resolve().parent.parent / "state_mappings.yaml"
    if not yaml_path.exists():
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None

    return (
        data.get("state_name_map", {}),
        data.get("canonical_states", []),
        data.get("canonical_to_geojson", {}),
        data.get("aggregate_state_patterns", []),
        data.get("aggregate_district_patterns", []),
    )


def _try_yaml_override():
    """Override module-level dicts from YAML if available."""
    global STATE_NAME_MAP, CANONICAL_STATES, CANONICAL_TO_GEOJSON
    global AGGREGATE_STATE_PATTERNS, AGGREGATE_DISTRICT_PATTERNS

    result = _load_yaml_mappings()
    if result is None:
        return  # Keep hardcoded values

    yaml_map, yaml_states, yaml_geojson, yaml_agg_state, yaml_agg_district = result

    # Merge YAML into existing (YAML takes precedence for overlapping keys)
    if yaml_map:
        STATE_NAME_MAP.update(yaml_map)
    if yaml_states:
        CANONICAL_STATES = yaml_states
    if yaml_geojson:
        CANONICAL_TO_GEOJSON.update(yaml_geojson)
    if yaml_agg_state:
        AGGREGATE_STATE_PATTERNS = yaml_agg_state
    if yaml_agg_district:
        AGGREGATE_DISTRICT_PATTERNS = yaml_agg_district


# Run YAML override on module import
_try_yaml_override()
