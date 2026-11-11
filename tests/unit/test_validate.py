import numpy as np
import pandas as pd
import pytest

from recsys.data.validate import DataValidationError, validate_ratings


def _make(n: int = 100, **overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1, dtype="int32"),
            "movie_id": np.full(n, 10, dtype="int32"),
            "rating": np.full(n, 3, dtype="int8"),
            "timestamp": np.full(n, 1_000_000, dtype="int64"),
        }
    )
    for col, value in overrides.items():
        df[col] = value
    return df


def _problems(df: pd.DataFrame, **kwargs) -> list[str]:
    with pytest.raises(DataValidationError) as exc_info:
        validate_ratings(df, **kwargs)
    return exc_info.value.problems


def test_valid_data_passes():
    validate_ratings(_make())


def test_nulls_detected():
    df = _make().astype({"rating": "float64"})
    df.loc[0, "rating"] = np.nan
    assert any("null" in p for p in _problems(df))


def test_rating_out_of_range():
    df = _make()
    df.loc[0, "rating"] = 6
    assert any("rating" in p for p in _problems(df))


def test_duplicates_detected():
    df = _make()
    df.loc[1, "user_id"] = df.loc[0, "user_id"]
    assert any("duplicate" in p for p in _problems(df))


def test_count_drop_detected():
    assert any("dropped" in p for p in _problems(_make(100), previous_count=200))


def test_small_drop_allowed():
    validate_ratings(_make(100), previous_count=104)


def test_all_problems_reported_together():
    df = _make().astype({"rating": "float64"})
    df.loc[0, "rating"] = np.nan
    df.loc[1, "user_id"] = df.loc[0, "user_id"]
    assert len(_problems(df)) >= 2
