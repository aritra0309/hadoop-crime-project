"""
Shared pytest fixtures for the India Crime Intelligence Platform test suite.
"""

import os
import sys
import pytest
from pyspark.sql import SparkSession

# Ensure src/ is importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession for testing (no HDFS required)."""
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("CrimePlatformTests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()
