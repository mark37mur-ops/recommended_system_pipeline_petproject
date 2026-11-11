"""Load MovieLens 1M into the `app` database, idempotently (spec 4.2 `bootstrap_data`).

Movies and users are upserted; ratings are streamed with COPY into a temporary table and
upserted from there (`ON CONFLICT (user_id, movie_id)`), touching only rows that changed, so a
re-run is cheap and leaves the database in the same state. Web data is never touched: web ids
live in their own range and every loaded row has `source = 'movielens'`.

CLI (the Airflow bootstrap DAG of stage 4 calls `load_movielens` directly):
    python -m recsys.db.load_movielens --data-dir data/raw/ml-1m [--download] [--database-url URL]
"""

import argparse
import io
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

from recsys.data.download import download_ml1m
from recsys.data.movielens import load_movies, load_ratings, load_users
from recsys.data.validate import validate_ratings
from recsys.db.schema import create_schema, make_engine, movies, ratings, users


@dataclass(frozen=True)
class LoadStats:
    movies: int
    users: int
    ratings: int


def _int_or_none(value) -> int | None:
    return None if pd.isna(value) else int(value)


def movie_rows(frame: pd.DataFrame) -> list[dict]:
    return [
        {
            "movie_id": int(m.movie_id),
            "title": str(m.title),
            "year": _int_or_none(m.year),
            "genres": "|".join(m.genres),
        }
        for m in frame.itertuples(index=False)
    ]


def user_rows(frame: pd.DataFrame) -> list[dict]:
    return [
        {
            "user_id": int(u.user_id),
            "source": "movielens",
            "gender": str(u.gender),
            "age": int(u.age),
            "occupation": int(u.occupation),
            "zip": str(u.zip),
        }
        for u in frame.itertuples(index=False)
    ]


def ratings_csv(frame: pd.DataFrame) -> str:
    """user_id,movie_id,rating,unix_timestamp lines for COPY (no header)."""
    cols = ["user_id", "movie_id", "rating", "timestamp"]
    return frame[cols].to_csv(index=False, header=False, lineterminator="\n")


_STAGE_DDL = """
CREATE TEMP TABLE ratings_stage (
    user_id integer, movie_id integer, rating smallint, ts bigint
) ON COMMIT DROP
"""
_UPSERT_RATINGS = """
INSERT INTO ratings (user_id, movie_id, rating, source, rated_at)
SELECT user_id, movie_id, rating, 'movielens', to_timestamp(ts) FROM ratings_stage
ON CONFLICT (user_id, movie_id) DO UPDATE
    SET rating = EXCLUDED.rating, rated_at = EXCLUDED.rated_at, updated_at = now()
    WHERE ratings.rating IS DISTINCT FROM EXCLUDED.rating
       OR ratings.rated_at IS DISTINCT FROM EXCLUDED.rated_at
"""


def load_movielens(engine: Engine, raw_dir: Path) -> LoadStats:
    """Create the schema if needed and upsert movies, users and ratings from `raw_dir`."""
    raw_dir = Path(raw_dir)
    movie_frame, user_frame = load_movies(raw_dir), load_users(raw_dir)
    rating_frame = load_ratings(raw_dir)
    validate_ratings(rating_frame)
    create_schema(engine)

    with engine.begin() as conn:
        stmt = insert(movies)
        conn.execute(
            stmt.on_conflict_do_update(
                index_elements=[movies.c.movie_id],
                set_={c: stmt.excluded[c] for c in ("title", "year", "genres")},
            ),
            movie_rows(movie_frame),
        )
        stmt = insert(users)
        conn.execute(
            stmt.on_conflict_do_update(
                index_elements=[users.c.user_id],
                set_={c: stmt.excluded[c] for c in ("gender", "age", "occupation", "zip")},
            ),
            user_rows(user_frame),
        )
        conn.exec_driver_sql(_STAGE_DDL)
        dbapi = conn.connection.driver_connection  # psycopg 3 connection, same transaction
        buffer = io.StringIO(ratings_csv(rating_frame))
        with dbapi.cursor() as cur:
            with cur.copy(
                "COPY ratings_stage (user_id, movie_id, rating, ts) FROM STDIN (FORMAT csv)"
            ) as copy:
                while chunk := buffer.read(1 << 20):
                    copy.write(chunk)
        conn.exec_driver_sql(_UPSERT_RATINGS)

    with engine.connect() as conn:

        def count(table, source: str | None = None) -> int:
            query = select(func.count()).select_from(table)
            if source is not None:
                query = query.where(table.c.source == source)
            return int(conn.execute(query).scalar_one())

        return LoadStats(
            movies=count(movies),
            users=count(users, "movielens"),
            ratings=count(ratings, "movielens"),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load MovieLens 1M into the app database.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "raw" / "ml-1m")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download + MD5-check ml-1m.zip into the parent of --data-dir if needed",
    )
    parser.add_argument("--database-url", default=None, help="default: $DATABASE_URL")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> LoadStats:
    args = parse_args(argv)
    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("set --database-url or $DATABASE_URL")
    raw_dir = args.data_dir
    if args.download:
        raw_dir = download_ml1m(args.data_dir.parent)
    engine = make_engine(url)
    try:
        stats = load_movielens(engine, raw_dir)
    finally:
        engine.dispose()
    print(f"loaded: {stats.movies} movies, {stats.users} users, {stats.ratings} ratings")
    return stats


if __name__ == "__main__":
    main()
