"""Parsers for the MovieLens 1M raw files."""

from pathlib import Path

import pandas as pd

_YEAR_RE = r"\s*\((\d{4})\)\s*$"


def _read(raw_dir: Path, name: str, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        Path(raw_dir) / name,
        sep="::",
        engine="python",
        encoding="latin-1",
        header=None,
        names=names,
        dtype=str,
        keep_default_na=False,
    )


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    df = _read(raw_dir, "ratings.dat", ["user_id", "movie_id", "rating", "timestamp"])
    return df.astype(
        {"user_id": "int32", "movie_id": "int32", "rating": "int8", "timestamp": "int64"}
    )


def load_movies(raw_dir: Path) -> pd.DataFrame:
    df = _read(raw_dir, "movies.dat", ["movie_id", "title", "genres"])
    year = df["title"].str.extract(_YEAR_RE, expand=False)
    return pd.DataFrame(
        {
            "movie_id": df["movie_id"].astype("int32"),
            "title": df["title"].str.replace(_YEAR_RE, "", regex=True),
            "year": pd.to_numeric(year).astype("Int16"),
            "genres": df["genres"].str.split("|"),
        }
    )


def load_users(raw_dir: Path) -> pd.DataFrame:
    df = _read(raw_dir, "users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    return df.astype({"user_id": "int32", "age": "int8", "occupation": "int8"})
