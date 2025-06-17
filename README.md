# My First Repo

This repository contains a Streamlit app for analyzing Apple Health data.

Metric groups and aggregation methods are stored in a local SQLite database
(`metrics.db`). You can edit them in the sidebar expander "Управление
метриками".

## Linting

To check code style with PEP8 using `pycodestyle`, run:

```bash
make lint
```

If you use npm-based workflow, run:

```bash
npm run lint
```

## Pre-commit

Install pre-commit hooks to run linting automatically before each commit:

```bash
pip install pre-commit
pre-commit install
```
