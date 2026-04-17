#!/usr/bin/env bash
# ============================================================
# Run the test suite
# Usage: bash scripts/run_tests.sh
# ============================================================

set -euo pipefail

echo "Running India Crime Intelligence Platform test suite..."
echo ""

python -m pytest tests/ -v --tb=short

echo ""
echo "All tests passed ✓"
