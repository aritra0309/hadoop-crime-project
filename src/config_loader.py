"""
Central configuration loader for the India Crime Intelligence Platform.

Reads config.yaml and exposes values to all modules. Supports three-tier
precedence: CLI args > config.yaml > built-in defaults.

Usage:
    from src.config_loader import get_config
    cfg = get_config()
    print(cfg.hdfs_base_url)
    print(cfg.forecast_horizon)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Project root: parent of src/
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config.yaml"


# ── Built-in defaults (used when config.yaml is missing a key) ──────────────

_DEFAULTS: Dict[str, Any] = {
    "hdfs": {
        "base_url": "hdfs://localhost:9000",
        "input_path": "/crime/input",
        "output_path": "/crime/output",
        "cleaned_path": "/crime/output",
    },
    "year_range": {
        "mode": "auto",
        "min_year": 2001,
        "max_year": 2014,
    },
    "forecast": {
        "horizon": 6,
        "polynomial_degrees": [2, 3],
        "ridge_alphas": [10.0, 50.0],
        "cv_splits": 3,
        "min_observations_for_cv": 6,
    },
    "kmeans": {
        "k_min": 2,
        "k_max": 8,
        "seed": 42,
        "max_iterations": 20,
    },
    "paths": {
        "geojson": "data/india_states.geojson",
        "data_dir": "data",
        "output_dir": "output/dashboard_data",
        "dashboard_dir": "dashboard",
    },
    "spark": {
        "app_name_prep": "Crime Data Preparation Phase0-1",
        "app_name_analytics": "Crime Analytics Phase 2",
        "log_level": "ERROR",
        "shuffle_partitions": 8,
    },
    "severity_weights": {
        "murder": 10,
        "rape": 8,
        "kidnapping": 6,
        "dacoity": 5,
        "robbery": 5,
        "burglary": 2,
        "theft": 1,
        "riots": 3,
        "arson": 4,
        "dowry_deaths": 8,
    },
    "cluster_labeling": {
        "secondary_threshold_z": 0.75,
        "secondary_features": [
            {"feature": "avg_recovery_rate", "pos_label": "Efficient Recovery", "neg_label": "Low Recovery"},
            {"feature": "avg_crimes_women", "pos_label": "High Women Crime", "neg_label": None},
            {"feature": "avg_fraud", "pos_label": "High Fraud", "neg_label": None},
            {"feature": "avg_firearms_murder", "pos_label": "High Firearms Use", "neg_label": None},
        ],
    },
    "pipeline": {
        "version": "1.3.0",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict keys."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


@dataclass
class PipelineConfig:
    """Flat, typed access to all configuration values."""

    # ── HDFS ──
    hdfs_base_url: str = "hdfs://localhost:9000"
    hdfs_input_path: str = "/crime/input"
    hdfs_output_path: str = "/crime/output"
    hdfs_cleaned_path: str = "/crime/output"

    @property
    def hdfs_input(self) -> str:
        return f"{self.hdfs_base_url}{self.hdfs_input_path}"

    @property
    def hdfs_output(self) -> str:
        return f"{self.hdfs_base_url}{self.hdfs_output_path}"

    @property
    def district_master_path(self) -> str:
        return f"{self.hdfs_output}/district_master"

    @property
    def state_master_path(self) -> str:
        return f"{self.hdfs_output}/state_master"

    # ── Year Range ──
    year_range_mode: str = "auto"
    year_range_min: int = 2001
    year_range_max: int = 2014

    def resolve_year_range(self, data_min: Optional[int] = None, data_max: Optional[int] = None) -> Tuple[int, int]:
        """Return (min_year, max_year) based on mode and optional data bounds."""
        if self.year_range_mode == "auto" and data_min is not None and data_max is not None:
            return (data_min, data_max)
        return (self.year_range_min, self.year_range_max)

    # ── Forecast ──
    forecast_horizon: int = 6
    forecast_poly_degrees: List[int] = field(default_factory=lambda: [2, 3])
    forecast_ridge_alphas: List[float] = field(default_factory=lambda: [10.0, 50.0])
    forecast_cv_splits: int = 3
    forecast_min_obs_cv: int = 6

    # ── KMeans ──
    kmeans_k_min: int = 2
    kmeans_k_max: int = 8
    kmeans_seed: int = 42
    kmeans_max_iterations: int = 20

    # ── Paths ──
    geojson_path: str = "data/india_states.geojson"
    data_dir: str = "data"
    output_dir: str = "output/dashboard_data"
    dashboard_dir: str = "dashboard"

    @property
    def abs_geojson_path(self) -> Path:
        return _ROOT / self.geojson_path

    @property
    def abs_data_dir(self) -> Path:
        return _ROOT / self.data_dir

    @property
    def abs_output_dir(self) -> Path:
        return _ROOT / self.output_dir

    # ── Spark ──
    spark_app_name_prep: str = "Crime Data Preparation Phase0-1"
    spark_app_name_analytics: str = "Crime Analytics Phase 2"
    spark_log_level: str = "ERROR"
    spark_shuffle_partitions: int = 8

    # ── Severity Weights ──
    severity_weights: Dict[str, int] = field(default_factory=lambda: {
        "murder": 10, "rape": 8, "kidnapping": 6, "dacoity": 5,
        "robbery": 5, "burglary": 2, "theft": 1, "riots": 3,
        "arson": 4, "dowry_deaths": 8,
    })

    # ── Cluster Labeling ──
    cluster_secondary_threshold_z: float = 0.75
    cluster_secondary_features: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"feature": "avg_recovery_rate", "pos_label": "Efficient Recovery", "neg_label": "Low Recovery"},
        {"feature": "avg_crimes_women", "pos_label": "High Women Crime", "neg_label": None},
        {"feature": "avg_fraud", "pos_label": "High Fraud", "neg_label": None},
        {"feature": "avg_firearms_murder", "pos_label": "High Firearms Use", "neg_label": None},
    ])

    # ── Pipeline ──
    pipeline_version: str = "1.3.0"

    # ── Raw dict (for anything custom) ──
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)


def _build_config(raw: Dict[str, Any]) -> PipelineConfig:
    """Build a PipelineConfig from a raw dict (merged defaults + yaml)."""
    return PipelineConfig(
        # HDFS
        hdfs_base_url=_get(raw, "hdfs", "base_url", default="hdfs://localhost:9000"),
        hdfs_input_path=_get(raw, "hdfs", "input_path", default="/crime/input"),
        hdfs_output_path=_get(raw, "hdfs", "output_path", default="/crime/output"),
        hdfs_cleaned_path=_get(raw, "hdfs", "cleaned_path", default="/crime/output"),
        # Year range
        year_range_mode=_get(raw, "year_range", "mode", default="auto"),
        year_range_min=int(_get(raw, "year_range", "min_year", default=2001)),
        year_range_max=int(_get(raw, "year_range", "max_year", default=2014)),
        # Forecast
        forecast_horizon=int(_get(raw, "forecast", "horizon", default=6)),
        forecast_poly_degrees=_get(raw, "forecast", "polynomial_degrees", default=[2, 3]),
        forecast_ridge_alphas=_get(raw, "forecast", "ridge_alphas", default=[10.0, 50.0]),
        forecast_cv_splits=int(_get(raw, "forecast", "cv_splits", default=3)),
        forecast_min_obs_cv=int(_get(raw, "forecast", "min_observations_for_cv", default=6)),
        # KMeans
        kmeans_k_min=int(_get(raw, "kmeans", "k_min", default=2)),
        kmeans_k_max=int(_get(raw, "kmeans", "k_max", default=8)),
        kmeans_seed=int(_get(raw, "kmeans", "seed", default=42)),
        kmeans_max_iterations=int(_get(raw, "kmeans", "max_iterations", default=20)),
        # Paths
        geojson_path=_get(raw, "paths", "geojson", default="data/india_states.geojson"),
        data_dir=_get(raw, "paths", "data_dir", default="data"),
        output_dir=_get(raw, "paths", "output_dir", default="output/dashboard_data"),
        dashboard_dir=_get(raw, "paths", "dashboard_dir", default="dashboard"),
        # Spark
        spark_app_name_prep=_get(raw, "spark", "app_name_prep", default="Crime Data Preparation Phase0-1"),
        spark_app_name_analytics=_get(raw, "spark", "app_name_analytics", default="Crime Analytics Phase 2"),
        spark_log_level=_get(raw, "spark", "log_level", default="ERROR"),
        spark_shuffle_partitions=int(_get(raw, "spark", "shuffle_partitions", default=8)),
        # Severity
        severity_weights=_get(raw, "severity_weights", default=_DEFAULTS["severity_weights"]),
        # Cluster labeling
        cluster_secondary_threshold_z=float(_get(raw, "cluster_labeling", "secondary_threshold_z", default=0.75)),
        cluster_secondary_features=_get(raw, "cluster_labeling", "secondary_features", default=_DEFAULTS["cluster_labeling"]["secondary_features"]),
        # Pipeline
        pipeline_version=_get(raw, "pipeline", "version", default="1.3.0"),
        # Raw
        _raw=raw,
    )


# ── Module-level singleton ──────────────────────────────────────────────────

_config: Optional[PipelineConfig] = None


def load_config(config_path: Optional[str] = None, cli_overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """
    Load configuration with three-tier precedence:
    CLI overrides > config.yaml > built-in defaults.
    """
    global _config

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    # Start with defaults
    raw = dict(_DEFAULTS)

    # Merge YAML if it exists
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        raw = _deep_merge(raw, yaml_data)

    # Merge CLI overrides
    if cli_overrides:
        raw = _deep_merge(raw, cli_overrides)

    _config = _build_config(raw)
    return _config


def get_config() -> PipelineConfig:
    """Return the loaded config singleton. Loads defaults if not yet loaded."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the singleton (useful for testing)."""
    global _config
    _config = None
