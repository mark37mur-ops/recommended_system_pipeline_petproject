import pytest
from api_fakes import FakeChampion, FakeRepository, FakeStore, StubRecommender

from recsys.db.schema import WEB_USER_ID_START
from recsys_api.cache import RateLimiter, RecommendationCache
from recsys_api.catalog import Catalog
from recsys_api.model import ModelHolder
from recsys_api.observer import Observer
from recsys_api.service import (
    MAX_K,
    DatabaseUnavailable,
    RateLimited,
    RatingQuotaExceeded,
    RecommendationService,
    UnknownMovie,
)


class Recorder(Observer):
    def __init__(self):
        self.served, self.cache, self.errors, self.ratings = [], [], [], []

    def recommendations_served(self, user_id, path, movie_ids, model_version, cached):
        self.served.append((path, cached))

    def cache_lookup(self, hit):
        self.cache.append(hit)

    def model_error(self, exc):
        self.errors.append(exc)

    def rating_received(self, source):
        self.ratings.append(source)


def make_service(recommender=None, max_ratings=300, limit=30):
    repo = FakeRepository()
    catalog = Catalog(repo.load_movies())
    repo.calls.clear()
    store = FakeStore()
    holder = ModelHolder(lambda: FakeChampion(recommender, "7"), lambda: "7")
    if recommender is not None:
        holder.reload()
    observer = Recorder()
    service = RecommendationService(
        repo=repo,
        catalog=lambda: catalog,
        models=holder,
        cache=RecommendationCache(store),
        limiter=RateLimiter(store, limit=limit),
        observer=observer,
        max_ratings=max_ratings,
    )
    return service, repo, store, observer


def rate_many(service, user_id, movies, rating=5, ip="1.2.3.4"):
    for movie in movies:
        user_id = service.rate(user_id, ip, movie, rating).user_id
    return user_id


def test_anonymous_visitor_gets_popular_without_db_access():
    service, repo, _, observer = make_service(StubRecommender())
    rec = service.recommend(None, k=5)
    assert rec.movie_ids == [1, 2, 3, 4, 5] and rec.path == "cold_start" and rec.n_ratings == 0
    assert repo.calls == [] and observer.served == [("cold_start", False)]


def test_first_rating_creates_web_user_in_web_range():
    service, repo, _, observer = make_service()
    result = service.rate(None, "1.2.3.4", 10, 4)
    assert result.created_user and result.user_id >= WEB_USER_ID_START and result.n_ratings == 1
    assert repo.ratings[result.user_id] == {10: 4} and observer.ratings == ["web"]
    again = service.rate(result.user_id, "1.2.3.4", 10, 2)
    assert not again.created_user and again.n_ratings == 1  # upsert, not a second row


def test_unknown_user_id_from_cookie_gets_a_new_user():
    service, _, _, _ = make_service()
    result = service.rate(WEB_USER_ID_START + 999, "1.2.3.4", 10, 4)
    assert result.created_user and result.user_id != WEB_USER_ID_START + 999


def test_one_to_four_ratings_are_genre_popular_cold_start():
    recommender = StubRecommender()
    service, _, _, _ = make_service(recommender)
    user = rate_many(service, None, [3, 9])  # two Horror movies rated 5
    rec = service.recommend(user, k=4)
    assert rec.path == "cold_start" and rec.n_ratings == 2
    assert rec.movie_ids == [15, 21, 27, 33]
    assert recommender.calls == []  # the model is not used below 5 ratings


def test_five_ratings_go_through_fold_in_and_exclude_rated():
    recommender = StubRecommender()
    service, _, _, _ = make_service(recommender)
    user = rate_many(service, None, [60, 59, 1, 2, 3])
    rec = service.recommend(user, k=10)
    assert rec.path == "full" and rec.model_version == "7" and rec.n_ratings == 5
    assert rec.movie_ids == list(range(58, 48, -1))
    assert sorted(recommender.calls[0]) == sorted([(60, 5), (59, 5), (1, 5), (2, 5), (3, 5)])


def test_ratings_passed_to_the_model_are_python_ints():
    recommender = StubRecommender()
    service, _, _, _ = make_service(recommender)
    user = rate_many(service, None, [1, 2, 3, 4, 5], rating=4)
    service.recommend(user, k=3)
    assert all(type(m) is int and type(r) is int for m, r in recommender.calls[0])


def test_repeat_request_is_served_from_cache_until_next_rating():
    recommender = StubRecommender()
    service, _, store, observer = make_service(recommender)
    user = rate_many(service, None, [1, 2, 3, 4, 5])
    first = service.recommend(user, k=10)
    second = service.recommend(user, k=3)
    assert not first.cached and second.cached and second.movie_ids == first.movie_ids[:3]
    assert len(recommender.calls) == 1 and f"recs:{user}:7" in store.data
    service.rate(user, "1.2.3.4", 60, 1)
    third = service.recommend(user, k=10)
    assert not third.cached and 60 not in third.movie_ids and len(recommender.calls) == 2
    assert observer.cache == [False, True, False]


def test_cache_holds_max_k_so_any_k_is_served():
    service, _, _, _ = make_service(StubRecommender())
    user = rate_many(service, None, [1, 2, 3, 4, 5])
    service.recommend(user, k=1)
    assert len(service.recommend(user, k=MAX_K).movie_ids) == MAX_K


def test_model_error_falls_back_to_popular_and_is_not_cached():
    service, _, store, observer = make_service(StubRecommender(fail=True))
    user = rate_many(service, None, [1, 2, 3, 4, 5])
    rec = service.recommend(user, k=3)
    assert rec.path == "fallback" and rec.movie_ids == [6, 7, 8]
    assert len(observer.errors) == 1 and not any(k.startswith("recs:") for k in store.data)


def test_no_model_loaded_falls_back_to_popular():
    service, _, _, _ = make_service(recommender=None)
    user = rate_many(service, None, [1, 2, 3, 4, 5])
    rec = service.recommend(user, k=3)
    assert rec.path == "fallback" and rec.model_version is None and rec.movie_ids == [6, 7, 8]


def test_model_returning_nothing_is_fallback():
    service, _, _, _ = make_service(StubRecommender(known=[999]))
    user = rate_many(service, None, [1, 2, 3, 4, 5])
    assert service.recommend(user, k=3).path == "fallback"


def test_short_model_list_is_padded_with_popular():
    class Short(StubRecommender):
        def recommend_new_user(self, ratings, k=10, profile=None):
            return [42]

    service, _, _, _ = make_service(Short())
    user = rate_many(service, None, [1, 2, 3, 4, 5])
    rec = service.recommend(user, k=4)
    assert rec.path == "full" and rec.movie_ids == [42, 6, 7, 8]


def test_postgres_down_serves_cache_then_popular_and_rejects_writes():
    service, repo, _, _ = make_service(StubRecommender())
    user = rate_many(service, None, [1, 2, 3, 4, 5])
    warm = service.recommend(user, k=5)
    repo.down = True
    cached = service.recommend(user, k=5)
    assert cached.cached and cached.movie_ids == warm.movie_ids
    other = service.recommend(user + 1, k=3)
    assert other.path == "fallback" and other.movie_ids == [1, 2, 3] and other.n_ratings is None
    with pytest.raises(DatabaseUnavailable):
        service.rate(user, "1.2.3.4", 9, 5)


def test_redis_down_still_recommends_and_rates_without_limits():
    service, _, store, _ = make_service(StubRecommender(), limit=2)
    store.down = True
    user = rate_many(service, None, [1, 2, 3, 4, 5])  # 5 > limit, but limiting is off
    rec = service.recommend(user, k=3)
    assert rec.path == "full" and not rec.cached


def test_rate_limit_per_user_and_per_ip():
    service, _, _, _ = make_service(limit=3)
    user = service.rate(None, "10.0.0.1", 1, 5).user_id  # creates the user: IP counter only
    service.rate(user, "10.0.0.2", 2, 5)
    service.rate(user, "10.0.0.2", 3, 5)
    service.rate(user, "10.0.0.3", 4, 5)
    with pytest.raises(RateLimited):
        service.rate(user, "10.0.0.4", 5, 5)  # 4th hit of this user, from a fresh IP
    service.rate(None, "10.0.0.2", 6, 5)  # 3rd hit of this IP, as a new visitor
    with pytest.raises(RateLimited):
        service.rate(None, "10.0.0.2", 7, 5)  # 4th hit of the IP


def test_rating_quota():
    service, _, _, _ = make_service(max_ratings=3)
    user = rate_many(service, None, [1, 2, 3])
    service.rate(user, "1.2.3.4", 3, 1)  # re-rating an already rated movie is fine
    with pytest.raises(RatingQuotaExceeded):
        service.rate(user, "1.2.3.4", 4, 5)


def test_unknown_movie_is_rejected_before_any_write():
    service, repo, _, _ = make_service()
    with pytest.raises(UnknownMovie):
        service.rate(None, "1.2.3.4", 999, 5)
    assert "create_web_user" not in repo.calls


def test_empty_catalog_means_database_problem():
    service, _, _, _ = make_service()
    service.catalog = lambda: Catalog([])
    with pytest.raises(DatabaseUnavailable):
        service.rate(None, "1.2.3.4", 1, 5)


def test_user_ratings_map_and_db_down():
    service, repo, _, _ = make_service()
    user = rate_many(service, None, [4, 5], rating=3)
    assert service.user_ratings(user) == {4: 3, 5: 3}
    assert service.user_ratings(None) == {}
    repo.down = True
    assert service.user_ratings(user) == {}
