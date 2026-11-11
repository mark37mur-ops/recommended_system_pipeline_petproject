"""Redis: recommendation cache and rate-limit counters (spec 3.3), with graceful degradation.

Every Redis problem surfaces as `StoreUnavailable`; the cache then reports a miss and the rate
limiter lets requests through (spec 3.5: "work without cache, rate limiting temporarily off").
`RedisStore` uses short socket timeouts and, after a failure, skips Redis for `cooldown`
seconds, so a dead Redis costs one timeout every few seconds instead of one per request.
"""

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import redis

log = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 600
GENERATION_TTL_SECONDS = 24 * 3600


class StoreUnavailable(Exception):
    pass


class KeyValueStore(Protocol):
    def get_many(self, keys: Sequence[str]) -> list[bytes | None]: ...
    def set(self, key: str, value: bytes, ttl: int) -> None: ...
    def delete(self, *keys: str) -> None: ...
    def incr_many(self, keys: Sequence[str], ttl: int) -> list[int]: ...
    def ping(self) -> bool: ...


class RedisStore:
    def __init__(
        self,
        url: str | None = None,
        *,
        client: "redis.Redis | None" = None,
        timeout: float = 0.2,
        cooldown: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if client is None:
            if url is None:
                raise ValueError("url or client is required")
            client = redis.Redis.from_url(
                url,
                socket_timeout=timeout,
                socket_connect_timeout=timeout,
                health_check_interval=30,
            )
        self._client = client
        self._cooldown = cooldown
        self._clock = clock
        self._down_until = 0.0

    def _call(self, fn: Callable[[], object]):
        if self._clock() < self._down_until:
            raise StoreUnavailable("redis marked down")
        try:
            return fn()
        except redis.RedisError as exc:
            if self._down_until == 0.0 or self._clock() >= self._down_until:
                log.warning("redis unavailable, skipping it for %.0fs: %s", self._cooldown, exc)
            self._down_until = self._clock() + self._cooldown
            raise StoreUnavailable(str(exc)) from exc

    def get_many(self, keys: Sequence[str]) -> list[bytes | None]:
        return self._call(lambda: self._client.mget(list(keys)))

    def set(self, key: str, value: bytes, ttl: int) -> None:
        self._call(lambda: self._client.set(key, value, ex=ttl))

    def delete(self, *keys: str) -> None:
        self._call(lambda: self._client.delete(*keys))

    def incr_many(self, keys: Sequence[str], ttl: int) -> list[int]:
        def run():
            pipe = self._client.pipeline(transaction=False)
            for key in keys:
                pipe.incr(key)
                pipe.expire(key, ttl, nx=True)
            return [int(v) for v in pipe.execute()[::2]]

        return self._call(run)

    def ping(self) -> bool:
        try:
            return bool(self._call(self._client.ping))
        except StoreUnavailable:
            return False

    def close(self) -> None:
        self._client.close()


@dataclass(frozen=True)
class CachedRecs:
    movie_ids: list[int]
    path: str
    n_ratings: int


class RecommendationCache:
    """`recs:{user_id}:{model_version}` -> top-MAX_K list (TTL 10 min).

    Each entry also stores the user's rating generation (`rv:{user_id}`, incremented by every
    new rating). An entry whose generation is not the current one is a miss: this closes the
    race where a request computed recommendations from the old ratings and wrote them after a
    new rating had already invalidated the key. A new model version is a miss by construction.
    """

    def __init__(self, store: KeyValueStore, ttl: int = CACHE_TTL_SECONDS) -> None:
        self.store = store
        self.ttl = ttl

    @staticmethod
    def key(user_id: int, version: str) -> str:
        return f"recs:{user_id}:{version}"

    @staticmethod
    def generation_key(user_id: int) -> str:
        return f"rv:{user_id}"

    def generation(self, user_id: int) -> int | None:
        """Current rating generation (read it before reading the ratings); None = no Redis."""
        try:
            (value,) = self.store.get_many([self.generation_key(user_id)])
        except StoreUnavailable:
            return None
        return int(value) if value is not None else 0

    def get(self, user_id: int, version: str) -> CachedRecs | None:
        try:
            entry, generation = self.store.get_many(
                [self.key(user_id, version), self.generation_key(user_id)]
            )
        except StoreUnavailable:
            return None
        if entry is None:
            return None
        payload = json.loads(entry)
        if payload["generation"] != (int(generation) if generation is not None else 0):
            return None
        return CachedRecs(payload["movie_ids"], payload["path"], payload["n_ratings"])

    def put(self, user_id: int, version: str, recs: CachedRecs, generation: int | None) -> None:
        if generation is None:
            return
        payload = {
            "movie_ids": recs.movie_ids,
            "path": recs.path,
            "n_ratings": recs.n_ratings,
            "generation": generation,
        }
        try:
            self.store.set(self.key(user_id, version), json.dumps(payload).encode(), self.ttl)
        except StoreUnavailable:
            pass

    def invalidate(self, user_id: int, version: str) -> None:
        try:
            self.store.incr_many([self.generation_key(user_id)], GENERATION_TTL_SECONDS)
            self.store.delete(self.key(user_id, version))
        except StoreUnavailable:
            pass


class RateLimiter:
    """Fixed-window counter per key (`rl:{key}:{window}`): at most `limit` hits per window."""

    def __init__(
        self,
        store: KeyValueStore,
        limit: int = 30,
        window: int = 60,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.limit = limit
        self.window = window
        self._clock = clock

    def allow(self, keys: Sequence[str]) -> bool:
        bucket = int(self._clock() // self.window)
        try:
            counts = self.store.incr_many([f"rl:{k}:{bucket}" for k in keys], self.window * 2)
        except StoreUnavailable:
            return True
        return all(count <= self.limit for count in counts)
