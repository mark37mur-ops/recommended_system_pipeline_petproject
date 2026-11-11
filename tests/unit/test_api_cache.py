import pytest
import redis
from api_fakes import FakeStore

from recsys_api.cache import (
    CachedRecs,
    RateLimiter,
    RecommendationCache,
    RedisStore,
    StoreUnavailable,
)


def test_cache_round_trip_uses_spec_key_and_ttl():
    store = FakeStore()
    cache = RecommendationCache(store, ttl=600)
    generation = cache.generation(7)
    cache.put(7, "3", CachedRecs([5, 4, 3], "full", 6), generation)
    assert "recs:7:3" in store.data and store.ttl["recs:7:3"] == 600
    assert cache.get(7, "3") == CachedRecs([5, 4, 3], "full", 6)
    assert cache.get(7, "4") is None  # new model version -> miss
    assert cache.get(8, "3") is None


def test_invalidate_drops_entry_and_beats_in_flight_writes():
    store = FakeStore()
    cache = RecommendationCache(store)
    generation = cache.generation(7)  # request A reads the generation ...
    cache.invalidate(7, "3")  # ... a new rating lands ...
    cache.put(7, "3", CachedRecs([1], "full", 5), generation)  # ... A writes its stale result
    assert cache.get(7, "3") is None
    cache.put(7, "3", CachedRecs([2], "full", 6), cache.generation(7))
    assert cache.get(7, "3").movie_ids == [2]
    cache.invalidate(7, "3")
    assert "recs:7:3" not in store.data


def test_cache_without_store_is_a_miss_not_an_error():
    store = FakeStore()
    store.down = True
    cache = RecommendationCache(store)
    assert cache.generation(7) is None and cache.get(7, "3") is None
    cache.put(7, "3", CachedRecs([1], "full", 5), None)
    cache.invalidate(7, "3")


def test_put_without_generation_is_skipped():
    store = FakeStore()
    RecommendationCache(store).put(7, "3", CachedRecs([1], "full", 5), None)
    assert store.data == {}


def test_rate_limiter_allows_limit_then_rejects_per_key():
    store = FakeStore()
    now = [1000.0]
    limiter = RateLimiter(store, limit=3, window=60, clock=lambda: now[0])
    assert all(limiter.allow(["user:1", "ip:a"]) for _ in range(3))
    assert not limiter.allow(["user:1", "ip:a"])
    assert limiter.allow(["user:2", "ip:b"])  # other user, other IP: own counters
    assert not limiter.allow(["user:3", "ip:a"])  # same IP, new user: the IP limit holds
    now[0] += 60  # next window
    assert limiter.allow(["user:1", "ip:a"])


def test_rate_limiter_without_store_allows():
    store = FakeStore()
    store.down = True
    assert RateLimiter(store, limit=1).allow(["user:1"])


class _BrokenRedis:
    def __init__(self):
        self.calls = 0

    def mget(self, keys):
        self.calls += 1
        raise redis.ConnectionError("down")

    def ping(self):
        self.calls += 1
        raise redis.ConnectionError("down")


def test_redis_store_short_circuits_after_failure():
    now = [0.0]
    client = _BrokenRedis()
    store = RedisStore(client=client, cooldown=5.0, clock=lambda: now[0])
    with pytest.raises(StoreUnavailable):
        store.get_many(["a"])
    with pytest.raises(StoreUnavailable):
        store.get_many(["a"])
    assert client.calls == 1  # second call did not wait for a timeout
    assert store.ping() is False
    now[0] = 6.0
    with pytest.raises(StoreUnavailable):
        store.get_many(["a"])
    assert client.calls == 2
