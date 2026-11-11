import math

import numpy as np
import pandas as pd
import pytest

from recsys.features import (
    FEATURE_COLUMNS,
    GENRE_COLUMNS,
    GENRES,
    SHARE_COLUMNS,
    build_feature_frame,
    drop_demographics,
    movie_features,
    movie_genre_matrix,
    user_features,
)

MOVIES = pd.DataFrame(
    {
        "movie_id": pd.Series([1, 2, 3, 4, 5], dtype="int32"),
        "title": ["A", "B", "C", "D", "E"],
        "year": pd.array([1995, 1999, None, 1980, 2000], dtype="Int16"),
        "genres": [["Comedy"], ["Comedy"], ["Drama", "Comedy"], ["Horror"], ["Unknown Genre"]],
    }
)
# user 10: two Comedy, one Drama|Comedy, one Horror; user 20: one Horror
RATINGS = pd.DataFrame(
    {
        "user_id": [10, 10, 10, 10, 20],
        "movie_id": [1, 2, 3, 4, 4],
        "rating": [5, 3, 4, 2, 1],
    }
)
USERS = pd.DataFrame(
    {
        "user_id": pd.Series([10, 20], dtype="int32"),
        "gender": ["M", "F"],
        "age": pd.Series([25, 56], dtype="int8"),
        "occupation": pd.Series([4, 0], dtype="int8"),
        "zip": ["1", "2"],
    }
)


def _col(genre: str) -> str:
    return GENRE_COLUMNS[GENRES.index(genre)]


def _share(genre: str) -> str:
    return SHARE_COLUMNS[GENRES.index(genre)]


def test_genre_vocabulary_is_ml1m():
    assert len(GENRES) == 18 and GENRES[0] == "Action" and "Children's" in GENRES
    assert len(set(GENRE_COLUMNS)) == 18 and all(c.isidentifier() for c in GENRE_COLUMNS)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_movie_features_counts_means_year_genres():
    mf = movie_features(RATINGS, MOVIES)
    assert mf.index.name == "movie_id" and list(mf.index) == [1, 2, 3, 4, 5]
    assert mf.loc[4, "movie_n_ratings"] == 2 and mf.loc[4, "movie_mean_rating"] == 1.5
    assert mf.loc[1, "movie_year"] == 1995 and math.isnan(mf.loc[3, "movie_year"])
    assert mf.loc[3, _col("Drama")] == 1 and mf.loc[3, _col("Comedy")] == 1
    assert mf.loc[3, _col("Horror")] == 0
    assert mf.loc[5, GENRE_COLUMNS].sum() == 0  # unknown genre names are ignored


def test_unrated_movie_zero_count_nan_mean():
    mf = movie_features(RATINGS, MOVIES)
    assert mf.loc[5, "movie_n_ratings"] == 0 and math.isnan(mf.loc[5, "movie_mean_rating"])


def test_user_features_counts_means_demographics():
    uf = user_features(RATINGS, movie_genre_matrix(MOVIES), USERS)
    assert uf.loc[10, "user_n_ratings"] == 4 and uf.loc[10, "user_mean_rating"] == 3.5
    assert uf.loc[10, "user_gender"] == 1.0 and uf.loc[20, "user_gender"] == 0.0
    assert uf.loc[10, "user_age"] == 25 and uf.loc[20, "user_occupation"] == 0
    assert uf.loc[10, _share("Comedy")] == pytest.approx(0.75)
    assert uf.loc[10, _share("Drama")] == pytest.approx(0.25)
    assert uf.loc[20, _share("Horror")] == pytest.approx(1.0)


def test_user_without_profile_has_nan_demographics():
    uf = user_features(RATINGS, movie_genre_matrix(MOVIES), users=None)
    assert uf[["user_age", "user_gender", "user_occupation"]].isna().all().all()
    partial = user_features(RATINGS, movie_genre_matrix(MOVIES), USERS[USERS["user_id"] == 20])
    assert math.isnan(partial.loc[10, "user_age"]) and partial.loc[20, "user_age"] == 56


def test_genre_affinity_hand_computed():
    uf = user_features(RATINGS, movie_genre_matrix(MOVIES), USERS)
    cands = pd.DataFrame({"user_id": [10, 10], "movie_id": [3, 4], "als_score": [0.9, 0.1]})
    cands["als_rank"] = [1, 2]
    frame = build_feature_frame(cands, movie_features(RATINGS, MOVIES), uf)
    # movie 3 = Drama|Comedy, shares 0.25 and 0.75 -> mean 0.5, max 0.75; movie 4 = Horror 0.25
    assert frame.loc[0, "genre_affinity_mean"] == pytest.approx(0.5)
    assert frame.loc[0, "genre_affinity_max"] == pytest.approx(0.75)
    assert frame.loc[1, "genre_affinity_mean"] == pytest.approx(0.25)


def test_movie_without_known_genres_has_nan_affinity():
    uf = user_features(RATINGS, movie_genre_matrix(MOVIES), USERS)
    cands = pd.DataFrame({"user_id": [20], "movie_id": [5], "als_score": [0.1], "als_rank": [1]})
    frame = build_feature_frame(cands, movie_features(RATINGS, MOVIES), uf)
    assert frame[["genre_affinity_mean", "genre_affinity_max"]].isna().all().all()


def test_single_user_features_match_batch():
    genres = movie_genre_matrix(MOVIES)
    batch = user_features(RATINGS, genres, USERS)
    single = user_features(RATINGS[RATINGS["user_id"] == 10], genres, USERS)
    pd.testing.assert_frame_equal(single.loc[[10]], batch.loc[[10]])


def test_feature_frame_columns_and_row_alignment():
    uf = user_features(RATINGS, movie_genre_matrix(MOVIES), USERS)
    cands = pd.DataFrame(
        {"user_id": [20, 10, 20], "movie_id": [1, 5, 2], "als_score": [0.3, 0.2, 0.1]}
    )
    cands["als_rank"] = [1, 1, 2]
    frame = build_feature_frame(cands, movie_features(RATINGS, MOVIES), uf)
    assert list(frame.columns) == ["user_id", "movie_id", *FEATURE_COLUMNS]
    assert frame["user_id"].tolist() == [20, 10, 20] and frame["movie_id"].tolist() == [1, 5, 2]
    assert frame["als_score"].tolist() == pytest.approx([0.3, 0.2, 0.1])
    assert frame.loc[1, "user_n_ratings"] == 4 and frame.loc[0, "user_n_ratings"] == 1
    assert frame.loc[0, "movie_year"] == 1995 and frame.loc[2, "movie_year"] == 1999
    assert all(frame[c].dtype == np.float32 for c in FEATURE_COLUMNS)


def test_drop_demographics_deterministic_fraction():
    uf = pd.DataFrame(
        {
            "user_n_ratings": np.arange(100, dtype="float32"),
            "user_age": np.full(100, 25.0),
            "user_gender": np.ones(100),
            "user_occupation": np.full(100, 3.0),
        },
        index=pd.Index(range(100), name="user_id"),
    )
    a = drop_demographics(uf, fraction=0.1, seed=1)
    b = drop_demographics(uf.sample(frac=1, random_state=0), fraction=0.1, seed=1).loc[a.index]
    masked = a["user_age"].isna()
    assert masked.sum() == 10
    assert a.loc[masked, ["user_gender", "user_occupation"]].isna().all().all()
    pd.testing.assert_frame_equal(a, b)
    assert a["user_n_ratings"].equals(uf["user_n_ratings"])
    assert not uf["user_age"].isna().any()  # input untouched
    assert drop_demographics(uf, fraction=0.0, seed=1).equals(uf)
