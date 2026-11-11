"""Small deterministic MovieLens-like data with planted tastes, shared by stage-2 tests.

Every user has a taste (one of TASTES); 80% of their ratings go to movies of that genre and
are high (4-5), the rest are random movies with low ratings (1-3). Inside a taste, movies are
picked with a skewed (Zipf-like) probability, so popularity is a reasonable but beatable
baseline. All timestamps are distinct and increasing per user.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TASTES = ("Comedy", "Drama", "Horror")
N_USERS = 60
N_MOVIES = 90
PER_USER = 25


def taste_of_user(user_id: int) -> str:
    return TASTES[user_id % len(TASTES)]


def taste_of_movie(movie_id: int) -> str:
    return TASTES[movie_id % len(TASTES)]


def make_movies(n_movies: int = N_MOVIES) -> pd.DataFrame:
    ids = np.arange(1, n_movies + 1)
    genres = [[taste_of_movie(m)] + (["Romance"] if m % 5 == 0 else []) for m in ids]
    return pd.DataFrame(
        {
            "movie_id": ids.astype("int32"),
            "title": [f"Movie {m}" for m in ids],
            "year": pd.array([1980 + m % 20 for m in ids], dtype="Int16"),
            "genres": genres,
        }
    )


def make_users(n_users: int = N_USERS) -> pd.DataFrame:
    ids = np.arange(1, n_users + 1)
    return pd.DataFrame(
        {
            "user_id": ids.astype("int32"),
            "gender": ["M" if u % 2 else "F" for u in ids],
            "age": np.array([(1, 18, 25, 35, 45, 50, 56)[u % 7] for u in ids], dtype="int8"),
            "occupation": (ids % 21).astype("int8"),
            "zip": [f"{10000 + u}" for u in ids],
        }
    )


def make_ratings(
    n_users: int = N_USERS, n_movies: int = N_MOVIES, per_user: int = PER_USER, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    movie_ids = np.arange(1, n_movies + 1)
    rows = []
    for user in range(1, n_users + 1):
        own = movie_ids[[taste_of_movie(m) == taste_of_user(user) for m in movie_ids]]
        other = np.setdiff1d(movie_ids, own)
        zipf = 1.0 / np.arange(1, len(own) + 1)
        n_own = int(per_user * 0.8)
        liked = rng.choice(own, size=n_own, replace=False, p=zipf / zipf.sum())
        disliked = rng.choice(other, size=per_user - n_own, replace=False)
        chosen = np.concatenate([liked, disliked])
        ratings = np.concatenate(
            [rng.integers(4, 6, size=n_own), rng.integers(1, 4, size=per_user - n_own)]
        )
        order = rng.permutation(per_user)
        for pos, i in enumerate(order):
            rows.append((user, int(chosen[i]), int(ratings[i]), 1_000_000 + user * 1000 + pos))
    df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    return df.astype(
        {"user_id": "int32", "movie_id": "int32", "rating": "int8", "timestamp": "int64"}
    )


def write_raw_dir(path: Path, seed: int = 0) -> Path:
    """Write ratings/movies/users in the ML-1M `::` format (latin-1) and return `path`."""
    path.mkdir(parents=True, exist_ok=True)
    ratings = make_ratings(seed=seed)
    lines = [f"{r.user_id}::{r.movie_id}::{r.rating}::{r.timestamp}" for r in ratings.itertuples()]
    (path / "ratings.dat").write_text("\n".join(lines) + "\n", encoding="latin-1")
    movies = make_movies()
    lines = [
        f"{m.movie_id}::{m.title} ({m.year})::{'|'.join(m.genres)}" for m in movies.itertuples()
    ]
    (path / "movies.dat").write_text("\n".join(lines) + "\n", encoding="latin-1")
    users = make_users()
    lines = [
        f"{u.user_id}::{u.gender}::{u.age}::{u.occupation}::{u.zip}" for u in users.itertuples()
    ]
    (path / "users.dat").write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path
