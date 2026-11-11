"""Snapshot extraction from the real `app` schema (Postgres-specific SQL: epoch, ordering)."""

import pandas as pd
from sqlalchemy import func
from synthetic import N_MOVIES, N_USERS, PER_USER, make_ratings, write_raw_dir

from recsys.db.extract import extract_snapshot
from recsys.db.load_movielens import load_movielens
from recsys.db.schema import WEB_USER_ID_START, ratings, users


def _add_web_user(db, user_id: int, movie_ratings: list[tuple[int, int]]) -> None:
    with db.begin() as conn:
        conn.execute(users.insert().values(user_id=user_id, source="web"))
        for movie_id, rating in movie_ratings:
            conn.execute(
                ratings.insert().values(
                    user_id=user_id,
                    movie_id=movie_id,
                    rating=rating,
                    source="web",
                    rated_at=func.now(),
                )
            )


def test_extract_returns_movielens_and_web_ratings(db, tmp_path):
    load_movielens(db, write_raw_dir(tmp_path / "raw"))
    _add_web_user(db, WEB_USER_ID_START, [(1, 5), (2, 4), (3, 1)])

    rating_frame, movie_frame, user_frame = extract_snapshot(db)

    assert len(rating_frame) == N_USERS * PER_USER + 3
    assert list(rating_frame.columns) == ["user_id", "movie_id", "rating", "timestamp", "source"]
    assert rating_frame["source"].value_counts().to_dict() == {
        "movielens": N_USERS * PER_USER,
        "web": 3,
    }
    # timestamps survive the round trip through timestamptz as unix seconds
    expected = make_ratings().sort_values(["user_id", "movie_id"]).reset_index(drop=True)
    got = rating_frame[rating_frame["source"] == "movielens"].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        got[["user_id", "movie_id", "rating", "timestamp"]],
        expected[["user_id", "movie_id", "rating", "timestamp"]],
        check_dtype=False,
    )
    web_ts = rating_frame.loc[rating_frame["source"] == "web", "timestamp"]
    assert (web_ts > 1_700_000_000).all()  # "now", far after MovieLens

    assert len(movie_frame) == N_MOVIES
    assert movie_frame.loc[movie_frame["movie_id"] == 5, "genres"].iloc[0] == ["Horror", "Romance"]
    assert movie_frame.loc[movie_frame["movie_id"] == 5, "year"].iloc[0] == 1985
    # demographics only for MovieLens users; web visitors get NaN features downstream
    assert len(user_frame) == N_USERS
    assert user_frame["user_id"].max() < WEB_USER_ID_START


def test_extract_without_web_ratings(db, tmp_path):
    load_movielens(db, write_raw_dir(tmp_path / "raw"))
    _add_web_user(db, WEB_USER_ID_START, [(1, 5)])
    rating_frame, _, _ = extract_snapshot(db, include_web=False)
    assert set(rating_frame["source"]) == {"movielens"}
    assert len(rating_frame) == N_USERS * PER_USER


def test_extract_from_an_empty_database_returns_empty_frames(db):
    rating_frame, movie_frame, user_frame = extract_snapshot(db)
    assert rating_frame.empty and movie_frame.empty and user_frame.empty
    assert list(rating_frame.columns) == ["user_id", "movie_id", "rating", "timestamp", "source"]
