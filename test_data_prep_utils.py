import numpy as np
import pandas as pd

from data_prep_utils import (
    add_datetime_features,
    cap_outliers_iqr,
    fill_missing_by_group,
    quick_eda_summary,
    train_test_split_by_date,
)


def test_cap_outliers_iqr():
    s = pd.Series([1, 2, 3, 4, 5, 100])
    capped = cap_outliers_iqr(s)
    assert capped.max() < 100
    assert capped.iloc[:5].tolist() == [1, 2, 3, 4, 5]


def test_fill_missing_by_group():
    df = pd.DataFrame({"grp": ["a", "a", "b", "b"], "val": [1.0, np.nan, np.nan, 3.0]})
    filled = fill_missing_by_group(df, "val", ["grp"])
    assert filled.isna().sum() == 0
    assert filled.iloc[1] == 1.0
    assert filled.iloc[2] == 3.0


def test_add_datetime_features():
    df = pd.DataFrame({"date": ["2024-01-06", "2024-01-08"]})
    out = add_datetime_features(df, "date")
    assert out["date_is_weekend"].tolist() == [1, 0]
    assert out["date_month"].tolist() == [1, 1]


def test_quick_eda_summary():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    summary = quick_eda_summary(df)
    assert summary.loc["a", "pct_missing"] > 0
    assert summary.loc["b", "n_unique"] == 3


def test_train_test_split_by_date():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"date": dates, "val": range(10)})
    train, test = train_test_split_by_date(df, "date", test_days=3)
    assert len(test) == 3
    assert len(train) == 7
    assert train["date"].max() < test["date"].min()
