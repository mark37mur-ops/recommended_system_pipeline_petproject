import dataclasses

import pandas as pd
import pytest

from recsys.data.split import Split, temporal_split

COLS = {"user_id": "int32", "movie_id": "int32", "rating": "int8", "timestamp": "int64"}


def _frame(rows: list[tuple[int, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(COLS)).astype(COLS)


def _user_rows(user: int, n: int, tie_ts: int | None = None) -> list[tuple[int, int, int, int]]:
    return [(user, m, 3, tie_ts if tie_ts is not None else 1000 + m) for m in range(1, n + 1)]


@pytest.fixture
def df() -> pd.DataFrame:
    rows = _user_rows(1, 20) + _user_rows(2, 10) + _user_rows(3, 15, tie_ts=5000)
    return _frame(rows)


def _count(part: pd.DataFrame, user: int) -> int:
    return int((part["user_id"] == user).sum())


def _of(part: pd.DataFrame, user: int) -> pd.DataFrame:
    return part[part["user_id"] == user]


def test_sizes_per_user(df):
    s = temporal_split(df)
    assert _count(s.test, 1) == 5 and _count(s.ranker_train, 1) == 5
    assert _count(s.als_train, 1) == 10


def test_short_history_goes_to_als_train_only(df):
    s = temporal_split(df)
    assert _count(s.als_train, 2) == 10
    assert _count(s.test, 2) == 0 and _count(s.ranker_train, 2) == 0


def test_exactly_min_ratings_is_split(df):
    s = temporal_split(df)
    assert _count(s.test, 3) == 5
    assert _count(s.ranker_train, 3) == 5
    assert _count(s.als_train, 3) == 5


def test_no_future_leakage(df):
    s = temporal_split(df)
    for u in (1, 3):
        assert _of(s.als_train, u).timestamp.max() <= _of(s.ranker_train, u).timestamp.min()
        assert _of(s.ranker_train, u).timestamp.max() <= _of(s.test, u).timestamp.min()
    # user 1 has distinct timestamps, so the ordering must be strict there
    assert _of(s.als_train, 1).timestamp.max() < _of(s.ranker_train, 1).timestamp.min()
    assert _of(s.ranker_train, 1).timestamp.max() < _of(s.test, 1).timestamp.min()


def test_parts_are_disjoint_and_complete(df):
    s = temporal_split(df)
    parts = pd.concat([s.als_train, s.ranker_train, s.test])
    assert len(parts) == len(df)
    assert not parts.duplicated(["user_id", "movie_id"]).any()


def test_ties_broken_by_movie_id_deterministically(df):
    df_c = df[df["user_id"] == 3]
    s = temporal_split(df_c)
    assert sorted(s.test.movie_id) == [11, 12, 13, 14, 15]
    shuffled = df_c.sample(frac=1, random_state=1)
    assert temporal_split(shuffled).test.equals(s.test)


def test_columns_dtypes_and_index_preserved(df):
    s = temporal_split(df)
    for part in (s.als_train, s.ranker_train, s.test):
        assert list(part.columns) == list(df.columns)
        assert part.dtypes.to_dict() == df.dtypes.to_dict()
        assert part.index.equals(pd.RangeIndex(len(part)))


def test_split_is_frozen(df):
    s = temporal_split(df)
    assert isinstance(s, Split)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.test = df  # type: ignore[misc]
