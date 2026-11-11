"""Ranker features, shared by training, offline evaluation and the online service.

Every function here is pure: DataFrames in, DataFrames out, no global state and no
training-only shortcuts. The service builds features for a web visitor by calling exactly the
same functions with that visitor's ratings (and no demographics), which is what protects the
model from train/serve skew (spec 3.2).

Feature groups (spec 2.3):
- ALS: score and rank of the candidate;
- movie: popularity (number of ratings), mean rating, year, one-hot genres;
- user: number of ratings, mean rating, age, gender, occupation (NaN for web users);
- pair: the user's share of ratings in the candidate's genres (mean and max over its genres).
"""

import re

import numpy as np
import pandas as pd

GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)


def _slug(genre: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", genre.lower().replace("'", "")).strip("_")


GENRE_COLUMNS = [f"genre_{_slug(g)}" for g in GENRES]
SHARE_COLUMNS = [f"share_{_slug(g)}" for g in GENRES]
DEMOGRAPHIC_COLUMNS = ["user_age", "user_gender", "user_occupation"]
FEATURE_COLUMNS = [
    "als_score",
    "als_rank",
    "movie_n_ratings",
    "movie_mean_rating",
    "movie_year",
    *GENRE_COLUMNS,
    "user_n_ratings",
    "user_mean_rating",
    *DEMOGRAPHIC_COLUMNS,
    "genre_affinity_mean",
    "genre_affinity_max",
]
CATEGORICAL_FEATURES = ["user_occupation"]
GENDER_CODES = {"F": 0.0, "M": 1.0}


def movie_genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    """0/1 matrix indexed by movie_id with one column per genre in GENRE_COLUMNS.

    Genre names outside the ML-1M vocabulary are ignored.
    """
    position = {g: i for i, g in enumerate(GENRES)}
    matrix = np.zeros((len(movies), len(GENRES)), dtype=np.float32)
    for row, genres in enumerate(movies["genres"]):
        for genre in genres:
            if genre in position:
                matrix[row, position[genre]] = 1.0
    index = pd.Index(movies["movie_id"].to_numpy(dtype=np.int64), name="movie_id")
    return pd.DataFrame(matrix, index=index, columns=GENRE_COLUMNS)


def movie_features(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Per-movie features over the given ratings, for every movie in `movies`.

    Unrated movies get movie_n_ratings = 0 and movie_mean_rating = NaN.
    """
    genres = movie_genre_matrix(movies)
    stats = ratings.groupby("movie_id")["rating"].agg(["size", "mean"])
    stats.index = stats.index.astype(np.int64)
    out = pd.DataFrame(index=genres.index)
    out["movie_n_ratings"] = stats["size"].reindex(out.index).fillna(0).astype(np.float32)
    out["movie_mean_rating"] = stats["mean"].reindex(out.index).astype(np.float32)
    year = pd.to_numeric(movies["year"], errors="coerce").astype("float64").to_numpy()
    out["movie_year"] = year.astype(np.float32)
    return pd.concat([out, genres], axis=1)


def user_features(
    ratings: pd.DataFrame, movie_genres: pd.DataFrame, users: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-user features for every user in `ratings`, indexed by user_id.

    `movie_genres` is `movie_genre_matrix(movies)` (or the genre columns of `movie_features`).
    `users` holds demographics (gender, age, occupation); users missing from it, or all users
    when it is None (web visitors), get NaN demographics. share_<genre> = share of the user's
    ratings that fall on movies of that genre (movies unknown to `movie_genres` count as
    genre-less).
    """
    stats = ratings.groupby("user_id")["rating"].agg(["size", "mean"])
    stats.index = stats.index.astype(np.int64)
    index = pd.Index(stats.index, name="user_id")
    out = pd.DataFrame(index=index)
    out["user_n_ratings"] = stats["size"].astype(np.float32)
    out["user_mean_rating"] = stats["mean"].astype(np.float32)

    if users is None:
        for col in DEMOGRAPHIC_COLUMNS:
            out[col] = np.float32("nan")
    else:
        profile = users.assign(user_id=users["user_id"].astype(np.int64)).set_index("user_id")
        profile = profile.reindex(index)
        out["user_age"] = pd.to_numeric(profile["age"], errors="coerce").astype(np.float32)
        out["user_gender"] = profile["gender"].map(GENDER_CODES).astype(np.float32)
        out["user_occupation"] = pd.to_numeric(profile["occupation"], errors="coerce").astype(
            np.float32
        )

    per_rating = movie_genres[GENRE_COLUMNS].reindex(ratings["movie_id"].to_numpy(dtype=np.int64))
    per_rating = per_rating.fillna(0.0).to_numpy(dtype=np.float32)
    shares = pd.DataFrame(per_rating, columns=SHARE_COLUMNS)
    shares["user_id"] = ratings["user_id"].to_numpy(dtype=np.int64)
    shares = shares.groupby("user_id")[SHARE_COLUMNS].mean().reindex(index)
    return pd.concat([out, shares.astype(np.float32)], axis=1)


def drop_demographics(user_feats: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Copy of `user_feats` with demographics set to NaN for round(fraction * n) users.

    Users are chosen by `seed` over the sorted user ids, so the result does not depend on row
    order. Used when training the ranker, so that it learns what to do with web users, whose
    demographics are unknown.
    """
    out = user_feats.copy()
    n = round(fraction * len(out))
    if n == 0:
        return out
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.sort(out.index.to_numpy()), size=n, replace=False)
    out.loc[chosen, DEMOGRAPHIC_COLUMNS] = np.nan
    return out


def build_feature_frame(
    candidates: pd.DataFrame, movie_feats: pd.DataFrame, user_feats: pd.DataFrame
) -> pd.DataFrame:
    """Join candidates (user_id, movie_id, als_score, als_rank) with movie and user features.

    Returns user_id, movie_id and FEATURE_COLUMNS (float32), one row per candidate, in the
    candidates' order, with a fresh RangeIndex.
    """
    user_ids = candidates["user_id"].to_numpy(dtype=np.int64)
    movie_ids = candidates["movie_id"].to_numpy(dtype=np.int64)
    mf = movie_feats.reindex(movie_ids)
    uf = user_feats.reindex(user_ids)

    movie_genres = mf[GENRE_COLUMNS].to_numpy(dtype=np.float32)
    shares = uf[SHARE_COLUMNS].to_numpy(dtype=np.float32)
    n_genres = movie_genres.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        affinity_mean = (shares * movie_genres).sum(axis=1) / n_genres
    masked = np.where(movie_genres > 0, shares, -np.inf)
    affinity_max = np.where(n_genres > 0, masked.max(axis=1, initial=-np.inf), np.nan)
    affinity_mean = np.where(n_genres > 0, affinity_mean, np.nan)

    frame = pd.DataFrame(
        {
            "user_id": user_ids,
            "movie_id": movie_ids,
            "als_score": candidates["als_score"].to_numpy(),
            "als_rank": candidates["als_rank"].to_numpy(),
        }
    )
    movie_cols = ["movie_n_ratings", "movie_mean_rating", "movie_year", *GENRE_COLUMNS]
    user_cols = ["user_n_ratings", "user_mean_rating", *DEMOGRAPHIC_COLUMNS]
    frame[movie_cols] = mf[movie_cols].to_numpy()
    frame[user_cols] = uf[user_cols].to_numpy()
    frame["genre_affinity_mean"] = affinity_mean
    frame["genre_affinity_max"] = affinity_max
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].astype(np.float32)
    return frame[["user_id", "movie_id", *FEATURE_COLUMNS]]
