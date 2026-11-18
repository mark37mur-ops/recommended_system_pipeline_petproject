"""Recommendation and rating logic: cold start (spec 2.4), cache (3.3), data protection (3.4)
and degradation (3.5). Knows nothing about HTTP, SQL or Redis: it talks to a `Repository`,
the cache/limiter wrappers and the `ModelHolder`.

Paths (the `path` label of spec 5.1):
- `cold_start`: 0 ratings -> popular; 1-4 -> popular in the genres rated high;
- `full`: >= 5 ratings -> fold-in ALS + ranker on the visitor's own ratings
  (`recommend_new_user`; stored ALS factors are never used by id, so web ids are irrelevant
  to the model), padded with popular movies if the model returns fewer than MAX_K;
- `fallback`: no model, the model failed or returned nothing -> eligible popular. If an
  identified visitor's exclusions cannot be read, return no ids. Fallbacks are not cached.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from recsys.db.schema import WEB_USER_ID_START
from recsys_api.cache import CachedRecs, RateLimiter, RecommendationCache
from recsys_api.catalog import Catalog, Movie
from recsys_api.model import LoadedModel, ModelHolder
from recsys_api.observer import Observer

log = logging.getLogger(__name__)

MIN_PERSONAL_RATINGS = 5
MAX_K = 50  # computed and cached once per user; the response is cut to the requested k
MAX_RATINGS_PER_USER = 300
NO_MODEL_VERSION = "none"
FeedbackState = Literal["watchlist", "dismissed"]


class DatabaseUnavailable(Exception):
    pass


class UserNotFound(Exception):
    pass


class RatingQuotaExceeded(Exception):
    pass


class RateLimited(Exception):
    pass


class UnknownMovie(Exception):
    pass


class MovieAlreadyRated(Exception):
    pass


class Repository(Protocol):
    """Access to the `app` database. Connection problems raise DatabaseUnavailable."""

    def ping(self) -> None: ...
    def load_movies(self) -> list[Movie]: ...
    def user_ratings(self, user_id: int) -> list[tuple[int, int]]: ...
    def create_web_user(self) -> int: ...
    def user_feedback(self, user_id: int) -> dict[int, FeedbackState]: ...
    def upsert_feedback(self, user_id: int, movie_id: int, state: FeedbackState) -> None: ...
    def delete_watchlist(self, user_id: int, movie_id: int) -> None: ...
    def upsert_rating(self, user_id: int, movie_id: int, rating: int, max_ratings: int) -> int:
        """Insert or update one rating, return the user's rating count. Raises UserNotFound,
        RatingQuotaExceeded (new movie while the user already has `max_ratings`)."""
        ...


@dataclass(frozen=True)
class Recommendation:
    movie_ids: list[int]
    path: str
    model_version: str | None
    n_ratings: int | None  # None when the database could not be asked
    cached: bool


@dataclass(frozen=True)
class RatingResult:
    user_id: int
    created_user: bool
    movie_id: int
    rating: int
    n_ratings: int


@dataclass(frozen=True)
class FeedbackResult:
    user_id: int
    created_user: bool
    movie_id: int
    state: FeedbackState


def _web_user(user_id: int | None) -> int | None:
    """Ids below the web range belong to MovieLens users: never act as one of them."""
    return user_id if user_id is not None and user_id >= WEB_USER_ID_START else None


class RecommendationService:
    def __init__(
        self,
        repo: Repository,
        catalog: Callable[[], Catalog],
        models: ModelHolder,
        cache: RecommendationCache,
        limiter: RateLimiter,
        observer: Observer | None = None,
        max_ratings: int = MAX_RATINGS_PER_USER,
        min_personal: int = MIN_PERSONAL_RATINGS,
    ) -> None:
        self.repo = repo
        self.catalog = catalog
        self.models = models
        self.cache = cache
        self.limiter = limiter
        self.observer = observer or Observer()
        self.max_ratings = max_ratings
        self.min_personal = min_personal
        # One inference at a time: predictable latency, no CPU oversubscription, no reliance on
        # implicit/LightGBM being safe for concurrent calls on one model object.
        self._inference = threading.Lock()

    # ----- recommendations ---------------------------------------------------------------
    def recommend(self, user_id: int | None, k: int) -> Recommendation:
        user_id = _web_user(user_id)
        model = self.models.current
        version = model.version if model is not None else None
        catalog = self.catalog()
        if user_id is None:  # no cookie yet: nothing rated
            return self._serve(None, catalog.popular(k), "cold_start", version, 0, False)

        cache_version = version or NO_MODEL_VERSION
        cached = self.cache.get(user_id, cache_version)
        self.observer.cache_lookup(cached is not None)
        generation = self.cache.generation(user_id)
        try:
            ratings = self.repo.user_ratings(user_id)
            feedback = self.repo.user_feedback(user_id)
        except DatabaseUnavailable as exc:
            log.warning("exclusions of user %s unavailable: %s", user_id, exc)
            return self._serve(user_id, [], "fallback", version, None, False)

        excluded = {m for m, _ in ratings} | set(feedback)
        if cached is not None:
            return self._serve(
                user_id,
                self._eligible(cached.movie_ids, catalog, excluded, k),
                cached.path,
                version,
                len(ratings),
                True,
            )

        movie_ids, path = self._compute(ratings, model, catalog)
        movie_ids = self._eligible(movie_ids, catalog, excluded, MAX_K)
        if path != "fallback":
            self.cache.put(
                user_id, cache_version, CachedRecs(movie_ids, path, len(ratings)), generation
            )
        return self._serve(user_id, movie_ids[:k], path, version, len(ratings), False)

    @staticmethod
    def _eligible(movie_ids: list[int], catalog: Catalog, excluded: set[int], k: int) -> list[int]:
        """Filter every serving path and refill from the entire popularity snapshot."""
        out = list(dict.fromkeys(m for m in movie_ids if m in catalog and m not in excluded))[:k]
        return out + catalog.popular(k - len(out), exclude=excluded | set(out))

    def _compute(
        self, ratings: list[tuple[int, int]], model: LoadedModel | None, catalog: Catalog
    ) -> tuple[list[int], str]:
        rated = {movie_id for movie_id, _ in ratings}
        if not ratings:
            return catalog.popular(MAX_K), "cold_start"
        if len(ratings) < self.min_personal:
            return catalog.genre_popular(ratings, MAX_K), "cold_start"
        if model is None:
            return catalog.popular(MAX_K, exclude=rated), "fallback"
        visitor = [(int(movie_id), int(rating)) for movie_id, rating in ratings]
        try:
            with self._inference:
                predicted = model.recommender.recommend_new_user(visitor, k=MAX_K)
        except Exception as exc:  # noqa: BLE001 - spec 3.5: model error -> popularity
            log.exception("model %s failed, serving popular", model.version)
            self.observer.model_error(exc)
            return catalog.popular(MAX_K, exclude=rated), "fallback"
        movie_ids = [int(m) for m in predicted if int(m) in catalog and int(m) not in rated]
        if not movie_ids:
            return catalog.popular(MAX_K, exclude=rated), "fallback"
        if len(movie_ids) < MAX_K:
            movie_ids += catalog.popular(MAX_K - len(movie_ids), exclude=rated | set(movie_ids))
        return movie_ids[:MAX_K], "full"

    def _serve(
        self,
        user_id: int | None,
        movie_ids: list[int],
        path: str,
        version: str | None,
        n_ratings: int | None,
        cached: bool,
    ) -> Recommendation:
        self.observer.recommendations_served(user_id, path, movie_ids, version, cached)
        return Recommendation(movie_ids, path, version, n_ratings, cached)

    # ----- ratings ----------------------------------------------------------------------
    def rate(self, user_id: int | None, ip: str, movie_id: int, rating: int) -> RatingResult:
        """Validate, rate-limit and upsert one rating; creates the web user on first rating."""
        user_id = _web_user(user_id)
        self._check_mutation(user_id, ip, movie_id)

        created = user_id is None
        if user_id is None:
            user_id = self.repo.create_web_user()
        try:
            n_ratings = self.repo.upsert_rating(user_id, movie_id, rating, self.max_ratings)
        except UserNotFound:  # signed cookie of a user the database no longer has
            user_id, created = self.repo.create_web_user(), True
            n_ratings = self.repo.upsert_rating(user_id, movie_id, rating, self.max_ratings)

        self._invalidate(user_id)
        self.observer.rating_received("web")
        return RatingResult(user_id, created, movie_id, rating, n_ratings)

    def user_ratings(self, user_id: int | None) -> dict[int, int]:
        """The visitor's ratings for display; empty when unknown or the database is down."""
        user_id = _web_user(user_id)
        if user_id is None:
            return {}
        try:
            return dict(self.repo.user_ratings(user_id))
        except DatabaseUnavailable:
            return {}

    def rating_history(self, user_id: int | None) -> dict[int, int]:
        """Strict history: a failed read must not masquerade as a new visitor."""
        user_id = _web_user(user_id)
        return {} if user_id is None else dict(self.repo.user_ratings(user_id))

    def feedback(self, user_id: int | None) -> dict[int, FeedbackState]:
        user_id = _web_user(user_id)
        return {} if user_id is None else self.repo.user_feedback(user_id)

    def _check_mutation(self, user_id: int | None, ip: str, movie_id: int) -> None:
        catalog = self.catalog()
        if not len(catalog):
            raise DatabaseUnavailable("movie catalog is not loaded")
        if movie_id not in catalog:
            raise UnknownMovie(movie_id)
        keys = [f"ip:{ip}"] + ([f"user:{user_id}"] if user_id is not None else [])
        if not self.limiter.allow(keys):
            raise RateLimited()

    def _invalidate(self, user_id: int) -> None:
        model = self.models.current
        self.cache.invalidate(user_id, model.version if model else NO_MODEL_VERSION)

    def set_feedback(
        self, user_id: int | None, ip: str, movie_id: int, state: FeedbackState
    ) -> FeedbackResult:
        if state not in ("watchlist", "dismissed"):
            raise ValueError("invalid feedback state")
        user_id = _web_user(user_id)
        self._check_mutation(user_id, ip, movie_id)
        created = user_id is None
        if user_id is None:
            user_id = self.repo.create_web_user()
        try:
            self.repo.upsert_feedback(user_id, movie_id, state)
        except UserNotFound:
            user_id, created = self.repo.create_web_user(), True
            self.repo.upsert_feedback(user_id, movie_id, state)
        self._invalidate(user_id)
        return FeedbackResult(user_id, created, movie_id, state)

    def remove_watchlist(self, user_id: int | None, ip: str, movie_id: int) -> None:
        user_id = _web_user(user_id)
        if user_id is None:
            return
        self._check_mutation(user_id, ip, movie_id)
        try:
            self.repo.delete_watchlist(user_id, movie_id)
        except UserNotFound:
            return  # an absent user has no watchlist, and deletion never creates one
        self._invalidate(user_id)
