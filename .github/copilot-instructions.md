# Copilot Instructions for Phoenix

## Project Overview

Phoenix is a bioinformatics tool for pathway enrichment analysis in single-cell expression data. It evaluates biological pathways using random forest classification (cell-types) and regression (pseudo-time) models. The pipeline has 4 steps: setup, pathway scoring, background scoring, and aggregation.

## Virtual Environment

Always activate the virtual environment before running any command in the terminal:

```
source .venv/bin/activate
```

## Running Tests

This project uses Python's built-in `unittest` framework.

Run all tests with:

```
python test.py
```

Run a specific test file with:

```
python -m unittest tests.<module_name>
```

## Test Conventions

- Test classes inherit from `tests.interface.Test`, which extends `unittest.TestCase` and provides shared helper methods like `generate_data()`.
- Test files are located in `tests/` and follow the naming pattern `test_*.py`.

## Pull Request Policy

Before creating a PR, run the full test suite (`python test.py`). If any tests fail, fix them first. Never create a PR with failing tests.

## Code Style

- Use type hints on all function signatures.
- Keep imports at the top of the file, standard library first, then third-party, then local.

## Type Checking

This project uses `mypy` for static type checking. Run with:

```
mypy scripts/
```
