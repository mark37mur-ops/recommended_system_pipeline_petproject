"""Training snapshot from the `app` database (first task of `retrain_daily`, spec 4.2).

Returns the same frames the raw-file parsers in `recsys.data.movielens` return, plus a `source`
column on ratings, so training code does not care where the data came from:
- ratings: user_id, movie_id, rating, timestamp (unix seconds of `rated_at`), source;
- movies: movie_id, title, year, genres (list);
- users: MovieLens demographics only. Web visitors have none, so their features are NaN, the
  same as in the API (spec 3.4, 2.3).
Columns are down-cast to the compact parser dtypes only when they hold no NULLs, so that
`validate_ratings` still sees (and reports) a NULL instead of a cast error.
"""

import pandas as pd
from sqlalchemy import Engine, text

RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp", "source"]
_RATINGS_SQL = """
SELECT user_id, movie_id, rating,
       CAST(EXTRACT(EPOCH FROM rated_at) AS BIGINT) AS timestamp, source
FROM ratings {where}
ORDER BY user_id, movie_id
"""
_MOVIES_SQL = "SELECT movie_id, title, year, genres FROM movies ORDER BY movie_id"
_USERS_SQL = """
SELECT user_id, gender, age, occupation, zip FROM users
WHERE source = 'movielens' ORDER BY user_id
"""


def _downcast(frame: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    usable = {c: t for c, t in dtypes.items() if c in frame and not frame[c].isna().any()}
    return frame.astype(usable)


def extract_snapshot(
    engine: Engine, include_web: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    where = "" if include_web else "WHERE source = 'movielens'"
    # One REPEATABLE READ transaction: the three tables come from the same moment.
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        ratings = pd.read_sql(text(_RATINGS_SQL.format(where=where)), conn)
        movies = pd.read_sql(text(_MOVIES_SQL), conn)
        users = pd.read_sql(text(_USERS_SQL), conn)

    ratings = _downcast(
        ratings.reindex(columns=RATING_COLUMNS),
        {"user_id": "int32", "movie_id": "int32", "rating": "int8", "timestamp": "int64"},
    )
    movies = pd.DataFrame(
        {
            "movie_id": movies["movie_id"].astype("int32"),
            "title": movies["title"].astype(str),
            "year": pd.array(movies["year"], dtype="Int16"),
            "genres": [g.split("|") if g else [] for g in movies["genres"]],
        }
    )
    users = _downcast(users, {"user_id": "int32", "age": "int8", "occupation": "int8"})
    return ratings, movies, users
