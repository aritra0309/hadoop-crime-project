"""
Golden-output integration tests for the India Crime Intelligence Platform.

These tests verify that core analytics functions produce structurally correct
and numerically consistent results against a small, deterministic dataset.
They run in Spark local mode (no HDFS required) and use the shared `spark`
fixture from conftest.py.
"""

import pytest
import json
import os
import sys

# Ensure src/ is importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ---------------------------------------------------------------------------
# Golden test: forecast structure & consistency
# ---------------------------------------------------------------------------
def test_forecast_consistency(spark):
    """Golden output: verify forecast produces expected structure."""
    # Create small test DataFrame
    data = [
        ('STATE_A', 2001, 100, 50),
        ('STATE_A', 2002, 110, 55),
        ('STATE_A', 2003, 105, 52),
        ('STATE_B', 2001, 200, 80),
        ('STATE_B', 2002, 190, 75),
        ('STATE_B', 2003, 210, 85),
    ]
    df = spark.createDataFrame(data, ['state', 'year', 'total_ipc', 'total_women'])

    from src.analytics import compute_forecasts
    notes = []
    result = compute_forecasts(df, notes)

    # Verify structure
    assert 'time_series' in result
    assert 'model_metadata' in result

    # Verify actual data points are present
    actuals = [r for r in result['time_series'] if r['type'] == 'actual']
    assert len(actuals) == 12  # 2 states * 2 metrics * 3 years

    # Verify predictions exist
    preds = [r for r in result['time_series'] if r['type'] == 'predicted']
    assert len(preds) > 0

    # Verify all predicted values are non-negative
    for p in preds:
        assert p['value'] >= 0


# ---------------------------------------------------------------------------
# Golden test: cluster structure
# ---------------------------------------------------------------------------
def test_cluster_structure(spark):
    """Golden output: verify clustering returns valid assignments."""
    data = [
        ('STATE_A', 2001, 100, 50, 10, 5, 3),
        ('STATE_A', 2002, 110, 55, 12, 6, 4),
        ('STATE_B', 2001, 500, 200, 50, 25, 15),
        ('STATE_B', 2002, 520, 210, 55, 28, 16),
        ('STATE_C', 2001, 300, 120, 30, 12, 8),
        ('STATE_C', 2002, 310, 125, 32, 14, 9),
    ]
    df = spark.createDataFrame(
        data,
        ['state', 'year', 'total_ipc', 'total_women', 'murder', 'rape', 'robbery'],
    )

    from src.analytics import compute_clusters
    notes = []
    result = compute_clusters(df, notes)

    # Must have cluster assignments for all 3 states
    assert 'state_assignments' in result
    assigned_states = {r['state'] for r in result['state_assignments']}
    assert assigned_states == {'STATE_A', 'STATE_B', 'STATE_C'}

    # optimal_k must be a sensible integer
    assert result['optimal_k'] is not None
    assert 1 <= result['optimal_k'] <= 3

    # Every assignment must carry a cluster_label string
    for a in result['state_assignments']:
        assert 'cluster_label' in a
        assert isinstance(a['cluster_label'], str)


# ---------------------------------------------------------------------------
# Golden test: national trends structure
# ---------------------------------------------------------------------------
def test_national_trends_structure(spark):
    """Golden output: verify national trends aggregation."""
    data = [
        ('STATE_A', 2001, 100, 50, 10),
        ('STATE_A', 2002, 110, 55, 12),
        ('STATE_B', 2001, 200, 80, 20),
        ('STATE_B', 2002, 190, 75, 18),
    ]
    df = spark.createDataFrame(data, ['state', 'year', 'total_ipc', 'total_women', 'murder'])

    from src.analytics import compute_national_trends
    notes = []
    result = compute_national_trends(df, notes)

    # Must have yearly totals for both years
    assert 'yearly_totals' in result
    years_present = {r['year'] for r in result['yearly_totals']}
    assert years_present == {2001, 2002}

    # 2001 national total_ipc should be 100 + 200 = 300
    row_2001 = [r for r in result['yearly_totals'] if r['year'] == 2001][0]
    assert row_2001['total_ipc'] == 300

    # overall_change_pct must be a number
    assert result['overall_change_pct'] is not None


# ---------------------------------------------------------------------------
# Golden test: forecast determinism (same input → same output)
# ---------------------------------------------------------------------------
def test_forecast_determinism(spark):
    """Running forecast twice on identical data must yield identical results."""
    data = [
        ('STATE_X', 2001, 100, 40),
        ('STATE_X', 2002, 120, 45),
        ('STATE_X', 2003, 115, 42),
        ('STATE_X', 2004, 130, 50),
    ]
    df = spark.createDataFrame(data, ['state', 'year', 'total_ipc', 'total_women'])

    from src.analytics import compute_forecasts

    notes1 = []
    r1 = compute_forecasts(df, notes1)

    notes2 = []
    r2 = compute_forecasts(df, notes2)

    # Time series values must be identical across runs
    vals1 = [(e['state'], e['year'], e['metric'], e['type'], e['value'])
             for e in r1['time_series']]
    vals2 = [(e['state'], e['year'], e['metric'], e['type'], e['value'])
             for e in r2['time_series']]
    assert vals1 == vals2


# ---------------------------------------------------------------------------
# Golden test: forecast with single-year edge case
# ---------------------------------------------------------------------------
def test_forecast_single_year_skip(spark):
    """Forecasting should gracefully skip states with only 1 year of data."""
    data = [
        ('LONELY_STATE', 2005, 42, 10),
    ]
    df = spark.createDataFrame(data, ['state', 'year', 'total_ipc', 'total_women'])

    from src.analytics import compute_forecasts
    notes = []
    result = compute_forecasts(df, notes)

    # The single actual point should still appear
    actuals = [r for r in result['time_series'] if r['type'] == 'actual']
    assert len(actuals) == 2  # 1 state * 2 metrics * 1 year

    # Model metadata should mark it as skipped
    for m in result['model_metadata']:
        assert m['skipped'] is True

    # Notes should mention insufficient data
    assert any('insufficient' in n.lower() or 'only 1' in n.lower() for n in notes)
