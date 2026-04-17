# Contributing to India Crime Intelligence Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/hadoop-crime-project.git
   cd hadoop-crime-project
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest
   ```

## Running Tests

Before submitting any changes, run the test suite:

```bash
bash scripts/run_tests.sh
# or directly:
python -m pytest tests/ -v
```

All tests must pass before a pull request will be reviewed.

## Project Structure

```
src/
├── state_mapping.py    # State name normalization (add new aliases here)
├── utils.py            # Shared PySpark utilities
├── data_preparation.py # Stage 1: data ingestion and cleaning
├── analytics.py        # Stage 2: clustering, forecasting, composition
└── visualization.py    # Stage 3: interactive HTML visualizations
tests/
├── conftest.py              # Shared Spark session fixture
├── test_state_mapping.py    # State mapping unit tests
├── test_utils.py            # PySpark utility tests
└── test_data_integrity.py   # Data file validation tests
```

## How to Contribute

### Adding New Data Sources
1. Place CSV files in `data/`
2. Check for state name variants — add mappings to `STATE_NAME_MAP` in `src/state_mapping.py`
3. Add ingestion logic in `src/data_preparation.py`
4. Add corresponding tests

### Adding New Analytics
1. Add functions in `src/analytics.py`
2. Add corresponding tests in `tests/`
3. Update visualization if needed

### Reporting Issues
- Use [GitHub Issues](https://github.com/aritra0309/hadoop-crime-project/issues)
- Include steps to reproduce, expected vs actual behavior

## Code Style

- Python 3.9+
- PEP 8 compliant
- Type hints encouraged
- Docstrings for all public functions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
