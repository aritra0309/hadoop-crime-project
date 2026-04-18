"""
Schema validation for incoming CSV datasets.

Checks column presence, state name validity, and data quality.
Returns warnings as lists of strings — never raises exceptions.

Usage:
    from src.schema_validator import validate_all
    warnings = validate_all(df, dataset_config)
    for w in warnings:
        print(f"WARNING: {w}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

try:
    from src.state_mapping import STATE_NAME_MAP, CANONICAL_STATES
except ModuleNotFoundError:
    from state_mapping import STATE_NAME_MAP, CANONICAL_STATES


def validate_csv_schema(df: DataFrame, dataset_config: Dict[str, Any]) -> List[str]:
    """
    Check if a DataFrame has the expected columns from the dataset registry.

    Parameters
    ----------
    df : DataFrame
        The loaded CSV as a Spark DataFrame.
    dataset_config : dict
        Entry from datasets.yaml with 'expected_columns' key.

    Returns
    -------
    list of str
        Warning messages for missing columns.
    """
    warnings: List[str] = []
    expected = dataset_config.get("expected_columns", [])
    if not expected:
        return warnings

    actual_upper = {c.upper().strip() for c in df.columns}
    dataset_name = dataset_config.get("name", "unknown")

    for col_name in expected:
        if col_name.upper().strip() not in actual_upper:
            # Check aliases too
            aliases = []
            for alias_list in dataset_config.get("column_aliases", {}).values():
                aliases.extend(alias_list)
            alias_upper = {a.upper().strip() for a in aliases}

            if col_name.upper().strip() not in alias_upper:
                warnings.append(
                    f"[{dataset_name}] Expected column '{col_name}' not found. "
                    f"Available columns: {sorted(df.columns)}"
                )

    return warnings


def validate_state_names(
    df: DataFrame,
    state_col: str = "state",
    known_states: Optional[Set[str]] = None,
) -> List[str]:
    """
    Check for unknown state names not in the canonical mapping.

    Parameters
    ----------
    df : DataFrame
        DataFrame with a state column.
    state_col : str
        Name of the state column.
    known_states : set of str, optional
        Set of known canonical state names. Defaults to CANONICAL_STATES.

    Returns
    -------
    list of str
        Warning messages for unknown state names.
    """
    warnings: List[str] = []

    if state_col not in df.columns:
        return warnings

    if known_states is None:
        known_states = set(CANONICAL_STATES)

    # Also include all mapped values
    all_known = known_states | set(STATE_NAME_MAP.values()) | set(STATE_NAME_MAP.keys())
    all_known_upper = {s.upper().strip() for s in all_known}

    try:
        distinct_states = [
            r[0] for r in df.select(state_col).distinct().collect()
            if r[0] is not None
        ]
    except Exception:
        return warnings

    for state in distinct_states:
        state_upper = str(state).upper().strip().replace('"', '')
        if state_upper not in all_known_upper:
            # Skip aggregate patterns
            skip_patterns = [
                "TOTAL", "ALL INDIA", "ALL-INDIA", "STATES/UTS",
                "STATE/UT", "AREA_NAME",
            ]
            if any(pat in state_upper for pat in skip_patterns):
                continue
            warnings.append(
                f"Unknown state name '{state}' — not in canonical mapping. "
                f"Please add it to state_mappings.yaml."
            )

    return warnings


def validate_data_quality(df: DataFrame) -> List[str]:
    """
    Check for data quality issues: high null rates, suspicious year counts.

    Parameters
    ----------
    df : DataFrame
        Any DataFrame to validate.

    Returns
    -------
    list of str
        Warning messages for quality issues.
    """
    warnings: List[str] = []

    total_rows = df.count()
    if total_rows == 0:
        warnings.append("DataFrame is empty — no rows to validate.")
        return warnings

    # Check columns with >50% null values
    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        null_pct = null_count / total_rows * 100
        if null_pct > 50:
            warnings.append(
                f"Column '{col_name}' has {null_pct:.1f}% null values "
                f"({null_count}/{total_rows} rows)."
            )

    # Check for years with suspiciously few data points
    if "year" in df.columns:
        year_counts = (
            df.groupBy("year")
            .count()
            .orderBy("year")
            .collect()
        )
        if year_counts:
            counts = [r["count"] for r in year_counts]
            median_count = sorted(counts)[len(counts) // 2]
            threshold = max(1, median_count * 0.1)  # 10% of median

            for r in year_counts:
                if r["count"] < threshold:
                    warnings.append(
                        f"Year {r['year']} has only {r['count']} rows "
                        f"(median is {median_count}). Data may be incomplete."
                    )

    return warnings


def validate_all(
    df: DataFrame,
    dataset_config: Optional[Dict[str, Any]] = None,
    state_col: str = "state",
) -> List[str]:
    """
    Run all validation checks and return combined warnings.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to validate.
    dataset_config : dict, optional
        Entry from datasets.yaml. If None, schema check is skipped.
    state_col : str
        Name of the state column for state name validation.

    Returns
    -------
    list of str
        All warning messages combined.
    """
    warnings: List[str] = []

    if dataset_config:
        warnings.extend(validate_csv_schema(df, dataset_config))

    warnings.extend(validate_state_names(df, state_col))
    warnings.extend(validate_data_quality(df))

    return warnings


def load_dataset_registry(registry_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load the dataset registry from datasets.yaml.

    Parameters
    ----------
    registry_path : str, optional
        Path to datasets.yaml. Defaults to project root.

    Returns
    -------
    list of dict
        Dataset entries from the registry.
    """
    import yaml

    if registry_path is None:
        root = Path(__file__).resolve().parent.parent
        registry_path = str(root / "datasets.yaml")

    path = Path(registry_path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("datasets", [])


def find_matching_dataset(filename: str, registry: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """
    Find the dataset registry entry that matches a given filename.

    Parameters
    ----------
    filename : str
        The CSV filename to match.
    registry : list of dict, optional
        Dataset registry. Loaded from datasets.yaml if not provided.

    Returns
    -------
    dict or None
        The matching registry entry, or None if no match.
    """
    import fnmatch

    if registry is None:
        registry = load_dataset_registry()

    for entry in registry:
        pattern = entry.get("pattern", "")
        if fnmatch.fnmatch(filename, pattern):
            return entry

    return None
