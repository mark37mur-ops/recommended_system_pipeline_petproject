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


def test_failed_invalidation_makes_the_user_a_miss_until_retried():
    """Review I4: a rating whose invalidation failed must not leave a stale entry to be served
    once Redis is reachable again; the delete is retried on the next read."""
    store = FakeStore()
    cache = RecommendationCache(store)
    cache.put(7, "3", CachedRecs([1], "cold_start", 4), cache.generation(7))
    cache.put(8, "3", CachedRecs([2], "full", 9), cache.generation(8))
    store.down = True
    cache.invalidate(7, "3")  # the 5th rating lands while Redis is unreachable
    store.down = False
    assert "recs:7:3" in store.data  # Redis still holds the stale entry ...
    cache.put(7, "3", CachedRecs([3], "full", 5), 0)  # ... an in-flight stale write is dropped
    assert cache.get(7, "3") is None  # ... and it is not served: retried and dropped instead
    assert "recs:7:3" not in store.data and store.data["rv:7"] == b"1"
    assert cache.get(8, "3") is not None  # other users are not affected
    cache.put(7, "3", CachedRecs([4], "full", 5), cache.generation(7))
    assert cache.get(7, "3").movie_ids == [4]  # back to normal after the successful retry


def test_failed_retry_keeps_the_user_a_miss():
    store = FakeStore()
    cache = RecommendationCache(store)
    cache.put(7, "3", CachedRecs([1], "full", 5), cache.generation(7))
    store.down = True
    cache.invalidate(7, "3")
    assert cache.get(7, "3") is None
    store.down = False
    assert cache.get(7, "3") is None and "recs:7:3" not in store.data


def test_failed_invalidations_are_bounded_in_time_and_size():
    """A stale entry lives at most `ttl` after the failed invalidation, so the mark can expire
    then; too many marks at once make every user a miss for one `ttl` instead."""
    now = [0.0]
    store = FakeStore()
    cache = RecommendationCache(store, ttl=600, clock=lambda: now[0], max_dirty=2)
    store.down = True
    cache.invalidate(1, "3")
    store.down = False
    for user in (1, 2, 3, 4):
        cache.put(user, "3", CachedRecs([user], "full", 5), cache.generation(user))
    assert cache.get(1, "3") is None and cache.get(2, "3") is not None
    store.down = True
    cache.invalidate(1, "3")
    now[0] = 601.0
    store.down = False
    cache.put(1, "3", CachedRecs([1], "full", 5), cache.generation(1))
    assert cache.get(1, "3") is not None  # mark expired with the TTL of anything it guarded
    store.down = True
    for user in (2, 3, 4):  # one more than max_dirty
        cache.invalidate(user, "3")
    store.down = False
    assert cache.get(1, "3") is None  # overflow: everybody is a miss ...
    now[0] = 1202.0
    assert cache.get(1, "3") is not None  # ... for one TTL


class _FlakyRedis:
    """Minimal redis-py stand-in; `slow = True` makes the next call time out."""

    def __init__(self):
        self.data, self.slow = {}, False

    def _maybe_time_out(self):
        if self.slow:
            self.slow = False
            raise redis.TimeoutError("slow")

    def mget(self, keys):
        self._maybe_time_out()
        return [self.data.get(k) for k in keys]

    def set(self, key, value, ex=None):
        self._maybe_time_out()
        self.data[key] = value

    def delete(self, *keys):
        self._maybe_time_out()
        for key in keys:
            self.data.pop(key, None)

    def ping(self):
        self._maybe_time_out()
        return True

    def pipeline(self, transaction=False):
        return _FlakyPipeline(self)


class _FlakyPipeline:
    def __init__(self, client):
        self.client, self.keys = client, []

    def incr(self, key):
        self.keys.append(key)

    def expire(self, key, ttl, nx=False):
        pass

    def execute(self):
        self.client._maybe_time_out()
        out = []
        for key in self.keys:
            value = int(self.client.data.get(key, b"0")) + 1
            self.client.data[key] = str(value).encode()
            out += [value, True]
        return out


def test_rating_during_redis_cooldown_does_not_serve_stale_recommendations():
    """The reviewer's scenario: one Redis timeout (here from /ready's ping) opens a 5 s
    cooldown; the 5th rating lands inside it; afterwards the 4-rating cold-start entry is
    still in Redis with a matching generation and used to be served for 10 minutes."""
    from api_fakes import FakeChampion, FakeRepository, StubRecommender

    from recsys_api.catalog import Catalog
    from recsys_api.model import ModelHolder
    from recsys_api.service import RecommendationService

    now = [0.0]
    client = _FlakyRedis()
    store = RedisStore(client=client, clock=lambda: now[0])
    repo = FakeRepository()
    catalog = Catalog(repo.load_movies())
    holder = ModelHolder(lambda: FakeChampion(StubRecommender(), "7"), lambda: "7")
    holder.reload()
    service = RecommendationService(
        repo, lambda: catalog, holder, RecommendationCache(store), RateLimiter(store)
    )
    user = None
    for movie_id in (1, 2, 3, 4):
        user = service.rate(user, "ip", movie_id, 5).user_id
    assert service.recommend(user, 10).path == "cold_start"
    client.slow = True
    assert store.ping() is False  # one slow answer -> Redis skipped for 5 s
    service.rate(user, "ip", 5, 5)  # 5th rating inside the cooldown
    now[0] += 6  # cooldown over, Redis healthy, the stale entry is still there
    after = service.recommend(user, 10)
    assert (after.path, after.n_ratings, after.cached) == ("full", 5, False)
    assert service.recommend(user, 10).cached is True  # caching works again afterwards
