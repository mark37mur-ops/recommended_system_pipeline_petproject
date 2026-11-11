"""In-memory stand-ins for Postgres, Redis and the model, shared by the API unit tests."""

import itertools
from collections.abc import Sequence

from recsys.db.schema import WEB_USER_ID_START
from recsys_api.cache import StoreUnavailable
from recsys_api.catalog import Movie
from recsys_api.service import DatabaseUnavailable, RatingQuotaExceeded, UserNotFound

GENRE_CYCLE = ("Comedy", "Drama", "Action", "Horror", "Romance", "Sci-Fi")


def make_catalog_movies(n: int = 60) -> list[Movie]:
    """Movie i has genre GENRE_CYCLE[i % 6] (+ Drama for every 4th) and n_ratings = 1000 - i,
    so popularity order is the id order."""
    movies = []
    for i in range(1, n + 1):
        genres = (GENRE_CYCLE[i % len(GENRE_CYCLE)],)
        if i % 4 == 0 and "Drama" not in genres:
            genres = (*genres, "Drama")
        movies.append(Movie(i, f"Movie {i}", 1990 + i % 10, tuple(sorted(genres)), 1000 - i))
    return movies


class FakeRepository:
    def __init__(self, movies: Sequence[Movie] | None = None) -> None:
        self.movies = list(movies if movies is not None else make_catalog_movies())
        self.ratings: dict[int, dict[int, int]] = {}
        self._ids = itertools.count(WEB_USER_ID_START)
        self.down = False
        self.calls: list[str] = []

    def _check(self, name: str) -> None:
        self.calls.append(name)
        if self.down:
            raise DatabaseUnavailable("fake postgres is down")

    def ping(self) -> None:
        self._check("ping")

    def load_movies(self) -> list[Movie]:
        self._check("load_movies")
        return list(self.movies)

    def user_ratings(self, user_id: int) -> list[tuple[int, int]]:
        self._check("user_ratings")
        return list(self.ratings.get(user_id, {}).items())

    def create_web_user(self) -> int:
        self._check("create_web_user")
        user_id = next(self._ids)
        self.ratings[user_id] = {}
        return user_id

    def upsert_rating(self, user_id: int, movie_id: int, rating: int, max_ratings: int) -> int:
        self._check("upsert_rating")
        if user_id not in self.ratings:
            raise UserNotFound(user_id)
        own = self.ratings[user_id]
        if movie_id not in own and len(own) >= max_ratings:
            raise RatingQuotaExceeded(user_id)
        own[movie_id] = rating
        return len(own)


class FakeStore:
    """Redis-like store: values, TTLs, counters; `down=True` makes every call fail."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.ttl: dict[str, int] = {}
        self.down = False
        self.calls = 0

    def _check(self) -> None:
        self.calls += 1
        if self.down:
            raise StoreUnavailable("fake redis is down")

    def get_many(self, keys: Sequence[str]) -> list[bytes | None]:
        self._check()
        return [self.data.get(k) for k in keys]

    def set(self, key: str, value: bytes, ttl: int) -> None:
        self._check()
        self.data[key] = value
        self.ttl[key] = ttl

    def delete(self, *keys: str) -> None:
        self._check()
        for key in keys:
            self.data.pop(key, None)

    def incr_many(self, keys: Sequence[str], ttl: int) -> list[int]:
        self._check()
        out = []
        for key in keys:
            value = int(self.data.get(key, b"0")) + 1
            self.data[key] = str(value).encode()
            self.ttl.setdefault(key, ttl)
            out.append(value)
        return out

    def ping(self) -> bool:
        self._check()
        return True


class StubRecommender:
    """Recommends movies in descending id order, skipping rated ones; records every call."""

    variant = "stub"

    def __init__(self, fail: bool = False, known: Sequence[int] | None = None) -> None:
        self.fail = fail
        self.known = set(known) if known is not None else None
        self.calls: list[list[tuple[int, int]]] = []

    def recommend_new_user(self, ratings, k: int = 10, profile=None) -> list[int]:
        self.calls.append(list(ratings))
        if self.fail:
            raise RuntimeError("model exploded")
        rated = {m for m, _ in ratings}
        if self.known is not None and not rated & self.known:
            return []
        return [m for m in range(60, 0, -1) if m not in rated][:k]


class FakeChampion:
    def __init__(self, recommender, version: str, run_id: str | None = "run") -> None:
        self.recommender = recommender
        self.version = version
        self.run_id = run_id
