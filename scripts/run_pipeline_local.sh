#!/usr/bin/env bash
# ============================================================
# Run the full pipeline in Spark local mode (no HDFS required)
# Usage: bash scripts/run_pipeline_local.sh
# ============================================================

set -euo pipefail

echo "============================================"
echo " India Crime Intelligence Platform"
echo " Running in Spark Local Mode"
echo "============================================"

export SPARK_MODE=local

echo ""
echo "[1/3] Data Preparation..."
spark-submit --master "local[*]" src/data_preparation.py
echo "  ✓ Data preparation complete"

echo ""
echo "[2/3] Analytics (clustering, forecasting, composition)..."
spark-submit --master "local[*]" src/analytics.py
echo "  ✓ Analytics complete"

echo ""
echo "[3/3] Generating visualizations..."
spark-submit --master "local[*]" src/visualization.py
echo "  ✓ Visualizations generated in output/"

echo ""
echo "============================================"
echo " Pipeline complete! Check output/ directory"
echo "============================================"
