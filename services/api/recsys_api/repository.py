"""Postgres access for the API (`app` database, schema in `recsys.db.schema`)."""

import logging

from sqlalchemy import Engine, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeout

from recsys.db.schema import create_schema, movies, ratings, users, web_user_id_seq
from recsys_api.catalog import Movie
from recsys_api.service import DatabaseUnavailable, RatingQuotaExceeded, UserNotFound

log = logging.getLogger(__name__)


def _unavailable(exc: Exception) -> bool:
    return isinstance(exc, OperationalError) or (
        isinstance(exc, DBAPIError) and exc.connection_invalidated
    )


class PgRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def _guard(self, fn):
        try:
            return fn()
        except PoolTimeout as exc:  # every pooled connection busy for pool_timeout seconds
            raise DatabaseUnavailable(str(exc)) from exc
        except DBAPIError as exc:
            if _unavailable(exc):
                raise DatabaseUnavailable(str(exc.orig or exc)) from exc
            raise

    def ensure_schema(self) -> None:
        self._guard(lambda: create_schema(self.engine))

    def ping(self) -> None:
        def run():
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

        self._guard(run)

    def load_movies(self) -> list[Movie]:
        """All movies with their number of ratings (any source): the popularity signal."""
        counts = (
            select(ratings.c.movie_id, func.count().label("n"))
            .group_by(ratings.c.movie_id)
            .subquery()
        )
        query = select(
            movies.c.movie_id,
            movies.c.title,
            movies.c.year,
            movies.c.genres,
            func.coalesce(counts.c.n, 0),
        ).outerjoin(counts, counts.c.movie_id == movies.c.movie_id)

        def run():
            with self.engine.connect() as conn:
                return [
                    Movie(
                        int(movie_id),
                        title,
                        None if year is None else int(year),
                        tuple(g for g in genres.split("|") if g),
                        int(n),
                    )
                    for movie_id, title, year, genres, n in conn.execute(query)
                ]

        return self._guard(run)

    def user_ratings(self, user_id: int) -> list[tuple[int, int]]:
        query = (
            select(ratings.c.movie_id, ratings.c.rating)
            .where(ratings.c.user_id == user_id)
            .order_by(ratings.c.rated_at, ratings.c.movie_id)
        )

        def run():
            with self.engine.connect() as conn:
                return [(int(m), int(r)) for m, r in conn.execute(query)]

        return self._guard(run)

    def create_web_user(self) -> int:
        def run():
            with self.engine.begin() as conn:
                user_id = conn.execute(select(web_user_id_seq.next_value())).scalar_one()
                conn.execute(users.insert().values(user_id=user_id, source="web"))
                return int(user_id)

        return self._guard(run)

    def upsert_rating(self, user_id: int, movie_id: int, rating: int, max_ratings: int) -> int:
        """Upsert under a row lock on the user, so concurrent ratings of one user cannot both
        pass the quota check (spec 3.4: at most `max_ratings` ratings per user). Only web users
        are locked: a MovieLens id raises UserNotFound, so training ratings are never
        overwritten from the web."""

        def run():
            with self.engine.begin() as conn:
                locked = conn.execute(
                    select(users.c.user_id)
                    .where(users.c.user_id == user_id, users.c.source == "web")
                    .with_for_update()
                ).first()
                if locked is None:
                    raise UserNotFound(user_id)
                own = select(func.count()).where(ratings.c.user_id == user_id)
                n_before = conn.execute(own).scalar_one()
                exists = conn.execute(
                    select(ratings.c.movie_id).where(
                        ratings.c.user_id == user_id, ratings.c.movie_id == movie_id
                    )
                ).first()
                if exists is None and n_before >= max_ratings:
                    raise RatingQuotaExceeded(user_id)
                stmt = insert(ratings).values(
                    user_id=user_id,
                    movie_id=movie_id,
                    rating=rating,
                    source="web",
                    rated_at=func.now(),
                )
                conn.execute(
                    stmt.on_conflict_do_update(
                        index_elements=[ratings.c.user_id, ratings.c.movie_id],
                        set_={
                            "rating": stmt.excluded.rating,
                            "rated_at": func.now(),
                            "updated_at": func.now(),
                        },
                    )
                )
                return int(n_before + (exists is None))

        return self._guard(run)
