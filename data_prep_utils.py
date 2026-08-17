"""
Small collection of reusable data-cleaning / feature-engineering helpers
used across the churn, RAG, and forecasting projects in this profile.
Pure pandas/numpy, no project-specific dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cap_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Cap values outside [Q1 - k*IQR, Q3 + k*IQR] to the bound (winsorize)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower=lower, upper=upper)


def fill_missing_by_group(
    df: pd.DataFrame, target_col: str, group_cols: list[str], strategy: str = "median"
) -> pd.Series:
    """Fill NaNs in target_col using the median/mean within each group,
    falling back to the global median/mean if a group is entirely NaN."""
    if strategy not in ("median", "mean"):
        raise ValueError("strategy must be 'median' or 'mean'")
    agg = df.groupby(group_cols)[target_col].transform(strategy)
    global_fill = getattr(df[target_col], strategy)()
    return df[target_col].fillna(agg).fillna(global_fill)


def add_datetime_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add day-of-week, month, quarter, and is_weekend columns from a date column."""
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out[f"{date_col}_dow"] = dt.dt.dayofweek
    out[f"{date_col}_month"] = dt.dt.month
    out[f"{date_col}_quarter"] = dt.dt.quarter
    out[f"{date_col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    return out


def quick_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One-line-per-column summary: dtype, % missing, n unique, and a sample value.
    Handy first step before writing a full profiling report."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
            "sample_value": df.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else None),
        }
    )


def train_test_split_by_date(
    df: pd.DataFrame, date_col: str, test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split for time-series data: last `test_days` days become
    the test set, avoiding the leakage a random split would cause."""
    dt = pd.to_datetime(df[date_col])
    cutoff = dt.max() - pd.Timedelta(days=test_days - 1)
    train = df[dt < cutoff].reset_index(drop=True)
    test = df[dt >= cutoff].reset_index(drop=True)
    return train, test

