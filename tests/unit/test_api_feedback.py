import pytest
from api_fakes import FakeRepository, StubRecommender, make_catalog_movies
from test_api_service import make_service, rate_many

from recsys_api.catalog import Catalog
from recsys_api.service import DatabaseUnavailable, RateLimited, UnknownMovie


def setup(recommender=None, limit=100):
    service, _, store, observer = make_service(recommender, limit=limit)
    repo = FakeRepository()
    service.repo = repo
    return service, repo, store, observer


def test_intent_never_changes_ratings_or_model_input():
    model = StubRecommender()
    service, repo, _, _ = setup(model)
    user = rate_many(service, None, range(1, 6))
    before = dict(repo.ratings[user])
    result = service.set_feedback(user, "ip", 60, "watchlist")
    assert (result.user_id, result.created_user, result.movie_id, result.state) == (
        user,
        False,
        60,
        "watchlist",
    )
    service.recommend(user, 10)
    assert repo.ratings[user] == before
    assert model.calls == [list(before.items())]


def test_feedback_upsert_switches_state():
    service, repo, _, _ = setup()
    result = service.set_feedback(None, "ip", 10, "watchlist")
    assert result.created_user
    service.set_feedback(result.user_id, "ip", 10, "dismissed")
    assert service.feedback(result.user_id) == {10: "dismissed"}
    assert repo.ratings[result.user_id] == {}


def test_rating_clears_intent():
    service, _, _, _ = setup()
    user = service.set_feedback(None, "ip", 10, "watchlist").user_id
    service.rate(user, "ip", 10, 4)
    assert service.feedback(user) == {}
    assert service.rating_history(user) == {10: 4}


def test_remove_watchlist_only():
    service, repo, _, _ = setup()
    user = service.set_feedback(None, "ip", 10, "watchlist").user_id
    service.set_feedback(user, "ip", 11, "dismissed")
    service.remove_watchlist(user, "ip", 11)
    service.remove_watchlist(user, "ip", 10)
    service.remove_watchlist(user, "ip", 10)
    assert service.feedback(user) == {11: "dismissed"}
    repo.calls.clear()
    service.remove_watchlist(None, "ip", 10)
    assert repo.calls == []


def test_rated_movie_rejects_intent():
    from recsys_api.service import MovieAlreadyRated

    service, _, _, _ = setup()
    user = service.rate(None, "ip", 10, 4).user_id
    with pytest.raises(MovieAlreadyRated):
        service.set_feedback(user, "ip", 10, "dismissed")


def test_history_failure_is_not_empty():
    service, repo, _, _ = setup()
    user = service.rate(None, "ip", 10, 4).user_id
    repo.down = True
    with pytest.raises(DatabaseUnavailable):
        service.rating_history(user)
    with pytest.raises(DatabaseUnavailable):
        service.feedback(user)
    assert service.rating_history(None) == {} and service.feedback(None) == {}


def test_feedback_limits_and_web_id_guard():
    service, repo, _, _ = setup(limit=2)
    with pytest.raises(UnknownMovie):
        service.set_feedback(None, "ip", 999, "watchlist")
    assert "create_web_user" not in repo.calls
    repo.ratings[42] = {1: 5}
    user = service.set_feedback(42, "ip", 10, "watchlist").user_id
    assert user != 42 and repo.ratings[42] == {1: 5}
    service.rate(user, "ip", 2, 5)
    with pytest.raises(RateLimited):
        service.remove_watchlist(user, "ip", 10)


@pytest.mark.parametrize(
    "path", ["cache-hit", "cold-start", "full", "model-missing", "model-error"]
)
def test_exclusions_on_every_path(path):
    model = None if path == "model-missing" else StubRecommender(fail=path == "model-error")
    service, repo, _, observer = setup(model)
    user = rate_many(service, None, range(1, 3 if path == "cold-start" else 6))
    if path == "cache-hit":
        service.recommend(user, 20)
    repo.feedback_data[user] = {
        60: "watchlist",
        59: "dismissed",
        6: "dismissed",
        7: "watchlist",
        8: "dismissed",
    }
    final = []
    observer.recommendations_served = lambda uid, p, ids, v, c: final.append(list(ids))
    rec = service.recommend(user, 20)
    assert len(rec.movie_ids) == len(set(rec.movie_ids)) == 20
    assert not set(rec.movie_ids) & (set(repo.ratings[user]) | set(repo.feedback_data[user]))
    assert final == [rec.movie_ids]
    assert rec.cached == (path == "cache-hit")


def test_more_than_max_k_exclusions():
    service, repo, _, _ = setup()
    service.catalog = lambda: Catalog(make_catalog_movies(140))
    user = repo.create_web_user()
    repo.feedback_data[user] = {m: "dismissed" for m in range(1, 81)}
    assert service.recommend(user, 50).movie_ids == list(range(81, 131))
    repo.feedback_data[user] = {m: "dismissed" for m in range(1, 141)}
    assert service.recommend(user, 50).movie_ids == []


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("failure", ["ratings", "feedback"])
def test_db_down_does_not_serve_unknown_exclusions(warm, failure):
    service, repo, store, _ = setup()
    user = repo.create_web_user()
    if warm:
        service.recommend(user, 10)
    if failure == "ratings":
        repo.down = True
    else:

        def fail(uid):
            raise DatabaseUnavailable("feedback unavailable")

        repo.user_feedback = fail
    before = dict(store.data)
    rec = service.recommend(user, 10)
    assert (rec.movie_ids, rec.n_ratings, rec.path, rec.cached) == ([], None, "fallback", False)
    assert store.data == before
    assert service.recommend(None, 3).movie_ids == [1, 2, 3]


def test_feedback_dirty_cache_recovers_and_rejects_stale_put():
    from recsys_api.cache import CachedRecs

    service, repo, store, _ = setup()
    user = repo.create_web_user()
    service.recommend(user, 10)
    generation = service.cache.generation(user)
    store.down = True
    service.set_feedback(user, "ip", 1, "dismissed")
    store.down = False
    service.cache.put(user, "none", CachedRecs([1], "cold_start", 0), generation)
    rec = service.recommend(user, 10)
    assert not rec.cached and 1 not in rec.movie_ids
    assert service.recommend(user, 10).cached
    service.remove_watchlist(user, "ip", 1)
    assert not service.recommend(user, 10).cached


def test_watchlist_removal_returns_unrated_movie_to_pool():
    service, _, _, _ = setup()
    user = service.set_feedback(None, "ip", 1, "watchlist").user_id
    assert 1 not in service.recommend(user, 50).movie_ids
    service.remove_watchlist(user, "ip", 1)
    assert service.recommend(user, 1).movie_ids == [1]


def test_duplicate_and_unknown_model_ids_are_filtered_before_observer():
    class Duplicates(StubRecommender):
        def recommend_new_user(self, ratings, k=10, profile=None):
            return [60, 60, 999, 1, 59, 59]

    service, _, _, observer = setup(Duplicates())
    user = rate_many(service, None, range(1, 6))
    served = []
    observer.recommendations_served = lambda uid, p, ids, v, c: served.append(list(ids))
    rec = service.recommend(user, 50)
    assert len(rec.movie_ids) == len(set(rec.movie_ids)) == 50
    assert rec.movie_ids[:4] == [60, 59, 6, 7]
    assert served == [rec.movie_ids]


def test_cache_hit_reads_current_ratings_as_well_as_feedback():
    service, repo, _, _ = setup()
    user = repo.create_web_user()
    assert service.recommend(user, 3).movie_ids == [1, 2, 3]
    repo.ratings[user] = {1: 4}
    repo.feedback_data[user] = {2: "watchlist"}
    rec = service.recommend(user, 3)
    assert rec.cached and rec.movie_ids == [3, 4, 5] and rec.n_ratings == 1


def test_feedback_with_stale_cookie_recreates_user():
    from recsys.db.schema import WEB_USER_ID_START

    service, _, _, _ = setup()
    result = service.set_feedback(WEB_USER_ID_START + 999, "ip", 1, "dismissed")
    assert result.created_user and result.user_id != WEB_USER_ID_START + 999
    assert service.feedback(result.user_id) == {1: "dismissed"}
