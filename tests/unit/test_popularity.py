import pandas as pd
import pytest

from recsys.models.popularity import PopularityRecommender


@pytest.fixture
def df():
    # movie 2: 3 ratings (users 1, 2, 3); movies 1 and 3: 1 rating each (users 4, 5)
    return pd.DataFrame(
        {
            "user_id": pd.Series([1, 2, 3, 4, 5], dtype="int32"),
            "movie_id": pd.Series([2, 2, 2, 1, 3], dtype="int32"),
            "rating": pd.Series([4, 5, 3, 4, 2], dtype="int8"),
            "timestamp": pd.Series(
                [978300760, 978302109, 978301968, 978300275, 978824291], dtype="int64"
            ),
        }
    )


def test_orders_by_count_then_movie_id(df):
    # фильм 2: 3 оценки, фильмы 1 и 3: по 1
    assert PopularityRecommender().fit(df).recommend(k=3) == [2, 1, 3]


def test_excludes_seen(df):
    assert PopularityRecommender().fit(df).recommend(k=2, exclude={2}) == [1, 3]


def test_k_larger_than_catalog_returns_all_unseen(df):
    assert len(PopularityRecommender().fit(df).recommend(k=100, exclude={1})) == 2


def test_batch_uses_per_user_seen(df):
    recs = PopularityRecommender().fit(df).recommend_batch([7, 8], {7: {2}, 8: set()}, k=1)
    assert recs == {7: [1], 8: [2]}


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError):
        PopularityRecommender().recommend()
