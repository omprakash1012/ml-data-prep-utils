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

## Usage

```python
import pandas as pd
from data_prep_utils import (
    add_datetime_features,
    cap_outliers_iqr,
    fill_missing_by_group,
    quick_eda_summary,
    train_test_split_by_date,
)

df = pd.DataFrame({
    "region": ["north", "north", "south", "south"],
    "revenue": [100.0, None, 250.0, 4000.0],
    "order_date": ["2024-01-02", "2024-01-05", "2024-01-08", "2024-01-15"],
})

# Fill missing values using the group median, falling back to the global median
df["revenue"] = fill_missing_by_group(df, "revenue", ["region"])

# Winsorize outliers instead of dropping them
df["revenue"] = cap_outliers_iqr(df["revenue"])

# Add day-of-week / month / quarter / is_weekend columns
df = add_datetime_features(df, "order_date")

# One-line-per-column EDA summary: dtype, % missing, n unique, sample value
print(quick_eda_summary(df))

# Chronological train/test split - no shuffling, no leakage
train, test = train_test_split_by_date(df, "order_date", test_days=7)
```

See `test_data_prep_utils.py` for further examples of each function's expected input/output.

## Getting started

```bash
pip install -r requirements.txt
pytest test_data_prep_utils.py -v
```

## Notes

Kept intentionally small and dependency-light. No project-specific logic -
these are the kind of first-five-minutes helpers used at the start of most
tabular/time-series projects.
