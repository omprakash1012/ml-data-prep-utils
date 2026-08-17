# ML Data Prep Utils

A small collection of reusable pandas/numpy helpers for data cleaning and
feature engineering, factored out of patterns repeated across the
churn, RAG, and forecasting projects in this profile.

**Stack:** Python, Pandas, NumPy, pytest

## What's here

- `cap_outliers_iqr` - winsorize a column to the IQR bounds
- `fill_missing_by_group` - fill NaNs using a group-wise median/mean, with a global fallback
- `add_datetime_features` - day-of-week, month, quarter, is_weekend from a date column
- `quick_eda_summary` - one-line-per-column dtype/missing/unique summary
- `train_test_split_by_date` - chronological split for time-series data (no leakage)

## Getting started

```bash
  pip install -r requirements.txt
  pytest test_data_prep_utils.py -v
  ```

  ## Notes

  Kept intentionally small and dependency-light. No project-specific logic -
  these are the kind of first-five-minutes helpers used at the start of most
  tabular/time-series projects.
