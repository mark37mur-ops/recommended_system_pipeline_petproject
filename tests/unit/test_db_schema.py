"""App DB schema and loader helpers that need no database (the SQL side is in tests/integration)."""

import pandas as pd
from synthetic import make_movies, make_ratings, make_users

from recsys.db.load_movielens import movie_rows, ratings_csv, user_rows
from recsys.db.schema import SOURCES, WEB_USER_ID_START, metadata, ratings, users

ML1M_MAX_USER_ID = 6040


def test_movie_rows_join_genres_and_keep_missing_year_null():
    movies = make_movies(3)
    movies.loc[1, "year"] = pd.NA
    rows = movie_rows(movies)
    assert rows[0] == {
        "movie_id": 1,
        "title": "Movie 1",
        "year": 1981,
        "genres": "|".join(movies.loc[0, "genres"]),
    }
    assert rows[1]["year"] is None
    assert all(type(r["movie_id"]) is int for r in rows)


def test_user_rows_mark_source_movielens_and_use_python_types():
    rows = user_rows(make_users(2))
    assert rows[0] == {
        "user_id": 1,
        "source": "movielens",
        "gender": "M",
        "age": 18,
        "occupation": 1,
        "zip": "10001",
    }
    assert type(rows[0]["age"]) is int


def test_ratings_csv_has_one_line_per_rating_in_copy_order():
    frame = make_ratings(n_users=2, per_user=5)
    lines = ratings_csv(frame).splitlines()
    assert len(lines) == len(frame)
    first = frame.iloc[0]
    assert lines[0] == f"{first.user_id},{first.movie_id},{first.rating},{first.timestamp}"


def test_web_ids_cannot_collide_with_movielens_ids():
    assert WEB_USER_ID_START > ML1M_MAX_USER_ID
    checks = {c.name: str(c.sqltext) for c in users.constraints if c.name}
    assert str(WEB_USER_ID_START) in checks["users_web_id_range"]
    assert set(SOURCES) == {"movielens", "web"}


def test_ratings_unique_per_user_and_movie():
    assert [c.name for c in ratings.primary_key.columns] == ["user_id", "movie_id"]
    assert {"movies", "users", "ratings"} <= set(metadata.tables)
    assert "web_user_id_seq" in {s.name for s in metadata._sequences.values()}
