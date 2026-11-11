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


def _canonical(part: pd.DataFrame) -> pd.DataFrame:
    return part.sort_values(["user_id", "movie_id"]).reset_index(drop=True)


def test_split_is_invariant_to_input_row_order(df):
    base = temporal_split(df)
    for seed in (1, 2, 3):
        s = temporal_split(df.sample(frac=1, random_state=seed))
        for got, want in zip(
            (s.als_train, s.ranker_train, s.test),
            (base.als_train, base.ranker_train, base.test),
            strict=True,
        ):
            assert _canonical(got).equals(_canonical(want))


def test_all_ties_user_parts_are_disjoint_and_complete(df):
    s = temporal_split(df)
    sets = [set(_of(p, 3).movie_id) for p in (s.als_train, s.ranker_train, s.test)]
    assert [len(x) for x in sets] == [5, 5, 5]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert sets[0] | sets[1] | sets[2] == set(range(1, 16))


def test_tie_break_is_not_by_movie_id(df):
    # Pinned output of hash_pandas_object over (user_id, movie_id) for user 3 / movies 1..15
    # (verified once by hand): the hash order puts movies {1, 5, 7, 8, 15} last, so they land
    # in test. Ordering by movie_id would give {11..15}.
    s = temporal_split(df)
    assert set(_of(s.test, 3).movie_id) == {1, 5, 7, 8, 15}
    assert set(_of(s.test, 3).movie_id) != {11, 12, 13, 14, 15}


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
