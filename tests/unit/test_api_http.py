"""HTTP contract of the API (spec 3.1) with in-memory Postgres/Redis/MLflow stand-ins."""

import time

import pandas as pd
import pytest
from api_fakes import FakeChampion, FakeRepository, FakeStore, StubRecommender
from fastapi.testclient import TestClient
from pydantic import SecretStr

from recsys.db.schema import WEB_USER_ID_START
from recsys_api.identity import COOKIE_NAME, CookieSigner
from recsys_api.main import create_app
from recsys_api.settings import Settings

TOKEN = "unit-test-admin-token-0123456789abcdef"
SESSION_SECRET = "unit-test-session-secret-0123456789abcdef"


class Registry:
    def __init__(self, version: str | None = "7", fail: bool = False, recommender=None) -> None:
        self.version = version
        self.fail = fail
        self.recommender = recommender or StubRecommender()

    def probe(self):
        return self.version

    def load(self):
        if self.fail:
            raise ConnectionError("mlflow unreachable")
        return FakeChampion(self.recommender, self.version)


def make_client(registry=None, repo=None, store=None, background=False, **overrides):
    settings = Settings(
        session_secret=SESSION_SECRET,
        admin_token=TOKEN,
        model_retry_seconds=0.05,
        **overrides,
    )
    registry = registry or Registry()
    app = create_app(
        settings,
        repo=repo or FakeRepository(),
        store=store or FakeStore(),
        loader=registry.load,
        probe=registry.probe,
        background=background,
    )
    return TestClient(app)


@pytest.fixture
def client():
    with make_client() as c:
        yield c


def rate(client, movie_id, rating=5):
    return client.post("/ratings", json={"movie_id": movie_id, "rating": rating})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_page_and_static_assets(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert 'lang="ru"' in page.text and "/static/app.js" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_onboarding_lists_twenty_movies_with_own_ratings(client):
    items = client.get("/movies/onboarding").json()["items"]
    assert len(items) == 20 and all(item["user_rating"] is None for item in items)
    assert set(items[0]) == {"movie_id", "title", "year", "genres", "user_rating"}
    rate(client, items[0]["movie_id"], 4)
    again = client.get("/movies/onboarding").json()["items"]
    assert again[0]["user_rating"] == 4


def test_search(client):
    found = client.get("/movies/search", params={"q": "movie 1"}).json()["items"]
    assert found and all("movie 1" in item["title"].lower() for item in found)
    assert len(found) <= 20
    assert client.get("/movies/search", params={"q": "m"}).status_code == 422
    assert client.get("/movies/search", params={"q": "x" * 101}).status_code == 422


def test_first_rating_sets_a_signed_cookie_once(client):
    first = rate(client, 1)
    assert first.status_code == 200 and first.json() == {"movie_id": 1, "rating": 5, "n_ratings": 1}
    cookie = first.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE_NAME}=") and "HttpOnly" in cookie
    assert "samesite=lax" in cookie.lower()
    second = rate(client, 2)
    assert second.json()["n_ratings"] == 2 and "set-cookie" not in second.headers


def test_forged_cookie_cannot_act_as_another_user():
    repo = FakeRepository()
    with make_client(repo=repo) as alice, make_client(repo=repo) as mallory:
        rate(alice, 1, 5)
        alice_id = next(iter(repo.ratings))
        mallory.cookies.set(COOKIE_NAME, f"{alice_id}.forged")
        response = rate(mallory, 2, 1)
        assert response.status_code == 200 and response.json()["n_ratings"] == 1
        assert repo.ratings[alice_id] == {1: 5}
        assert len(repo.ratings) == 2


@pytest.mark.parametrize(
    "body",
    [
        {"movie_id": 1, "rating": 6},
        {"movie_id": 1, "rating": 0},
        {"movie_id": 1, "rating": 4.5},
        {"movie_id": 1, "rating": 4.0},
        {"movie_id": 1, "rating": "4"},
        {"movie_id": 1, "rating": True},
        {"movie_id": 1},
        {"movie_id": 1, "rating": 4, "user_id": 3},
        {"movie_id": 999, "rating": 4},
        {"movie_id": -1, "rating": 4},
    ],
)
def test_rating_validation(client, body):
    assert client.post("/ratings", json=body).status_code == 422


def test_cold_start_then_personal_then_cache(client):
    anonymous = client.get("/recommendations").json()
    assert anonymous["path"] == "cold_start" and anonymous["n_ratings"] == 0
    assert len(anonymous["items"]) == 10 and anonymous["min_ratings"] == 5
    for movie_id in (1, 2, 3):
        rate(client, movie_id)
    assert client.get("/recommendations").json()["path"] == "cold_start"
    for movie_id in (4, 5):
        rate(client, movie_id)
    personal = client.get("/recommendations", params={"k": 5}).json()
    assert personal["path"] == "full" and personal["model_version"] == "7"
    assert [i["movie_id"] for i in personal["items"]] == [60, 59, 58, 57, 56]
    assert personal["cached"] is False
    assert client.get("/recommendations", params={"k": 5}).json()["cached"] is True
    rate(client, 60, 1)
    fresh = client.get("/recommendations", params={"k": 5}).json()
    assert fresh["cached"] is False and 60 not in [i["movie_id"] for i in fresh["items"]]


def test_k_bounds(client):
    assert client.get("/recommendations", params={"k": 0}).status_code == 422
    assert client.get("/recommendations", params={"k": 51}).status_code == 422
    assert len(client.get("/recommendations", params={"k": 50}).json()["items"]) == 50


def test_rate_limit_returns_429():
    with make_client(rate_limit_per_minute=3) as client:
        assert [rate(client, m).status_code for m in (1, 2, 3)] == [200, 200, 200]
        limited = rate(client, 4)
        assert limited.status_code == 429 and limited.headers["retry-after"] == "60"


def test_rating_quota_returns_403():
    with make_client(max_ratings_per_user=2) as client:
        rate(client, 1)
        rate(client, 2)
        assert rate(client, 2, 3).status_code == 200  # re-rating is not a new rating
        assert rate(client, 3).status_code == 403


def test_postgres_down():
    repo = FakeRepository()
    with make_client(repo=repo) as client:
        rate(client, 1)
        repo.down = True
        assert rate(client, 2).status_code == 503
        recs = client.get("/recommendations").json()
        assert recs["path"] == "fallback" and len(recs["items"]) == 10
        ready = client.get("/ready")
        assert ready.status_code == 503 and ready.json()["database"] == "unavailable"


def test_redis_down_keeps_service_working():
    store = FakeStore()
    store.down = True
    with make_client(store=store, rate_limit_per_minute=1) as client:
        assert [rate(client, m).status_code for m in (1, 2, 3, 4, 5)] == [200] * 5
        recs = client.get("/recommendations").json()
        assert recs["path"] == "full" and recs["cached"] is False
        ready = client.get("/ready").json()
        assert ready["status"] == "ready" and ready["redis"] == "unavailable"


def test_ready_reports_model_db_and_catalog(client):
    ready = client.get("/ready")
    body = ready.json()
    assert ready.status_code == 200 and body["status"] == "ready"
    assert body["model"]["version"] == "7" and body["database"] == "ok"
    assert body["catalog_size"] == 60


def test_mlflow_down_at_start_degrades_to_popularity():
    registry = Registry(fail=True)
    with make_client(registry) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503 and ready.json()["status"] == "degraded"
        assert "mlflow unreachable" in ready.json()["model"]["last_error"]
        for movie_id in (1, 2, 3, 4, 5):
            rate(client, movie_id)
        recs = client.get("/recommendations").json()
        assert recs["path"] == "fallback" and recs["model_version"] is None
        registry.fail = False
        reload = client.post("/admin/reload-model", headers={"Authorization": f"Bearer {TOKEN}"})
        assert reload.status_code == 200 and reload.json()["status"] == "loaded"
        assert client.get("/recommendations").json()["path"] == "full"


def test_background_loader_retries_until_mlflow_is_up():
    registry = Registry(fail=True)
    with make_client(registry, background=True) as client:
        assert client.get("/ready").status_code == 503
        registry.fail = False
        deadline = time.monotonic() + 5
        while client.get("/ready").status_code != 200 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert client.get("/ready").json()["model"]["version"] == "7"


def test_admin_reload_contract():
    registry = Registry()
    with make_client(registry) as client:
        url = "/admin/reload-model"
        assert client.post(url).status_code == 401
        assert client.post(url, headers={"Authorization": "Bearer nope"}).status_code == 401
        auth = {"Authorization": f"Bearer {TOKEN}"}
        same = client.post(url, headers=auth).json()
        assert same["status"] == "unchanged" and same["version"] == "7"
        registry.version = "8"
        new = client.post(url, headers=auth).json()
        assert new == {
            "status": "loaded",
            "version": "8",
            "previous_version": "7",
            "error": None,
            "catalog_size": 60,
        }
        registry.version, registry.fail = "9", True
        failed = client.post(url, headers=auth)
        assert failed.status_code == 502 and failed.json()["version"] == "8"
        assert client.get("/ready").json()["model"]["version"] == "8"


def app_with(settings: Settings) -> TestClient:
    registry = Registry()
    app = create_app(
        settings,
        repo=FakeRepository(),
        store=FakeStore(),
        loader=registry.load,
        probe=registry.probe,
        background=False,
    )
    return TestClient(app)


def test_admin_disabled_without_token():
    with app_with(Settings(session_secret=SESSION_SECRET, admin_token=None)) as client:
        assert client.post("/admin/reload-model").status_code == 503


@pytest.mark.parametrize(
    "settings",
    [
        Settings(session_secret=SESSION_SECRET, admin_token=""),
        # Even if validation is bypassed, an empty token never authorises anybody.
        Settings.model_construct(session_secret=None, admin_token=SecretStr("")),
    ],
    ids=["validated", "constructed"],
)
@pytest.mark.parametrize("header", ["Bearer ", "Bearer", "bearer  ", ""])
def test_empty_admin_token_disables_reload(settings, header):
    with app_with(settings) as client:
        response = client.post("/admin/reload-model?force=true", headers={"Authorization": header})
        assert response.status_code == 503


def test_cookie_for_a_movielens_user_id_is_a_new_visitor():
    """A validly signed cookie for an id below the web range (a MovieLens user) must not let
    anybody write into that user's training data: it is ignored like a forged one."""
    repo = FakeRepository()
    repo.ratings[42] = {1: 3}
    with make_client(repo=repo) as client:
        client.cookies.set(COOKIE_NAME, CookieSigner(SESSION_SECRET).sign(42))
        assert client.get("/movies/onboarding").json()["items"][0]["user_rating"] is None
        response = rate(client, 1, 5)
        assert response.status_code == 200 and response.json()["n_ratings"] == 1
        new_id = int(response.headers["set-cookie"].split("=", 1)[1].split(".", 1)[0])
        assert new_id >= WEB_USER_ID_START
    assert repo.ratings[42] == {1: 3} and repo.ratings[new_id] == {1: 5}


class ModelWithMovies(StubRecommender):
    """Stub with the `movie_feats` table of a real TwoStageRecommender (popularity column)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.movie_feats = pd.DataFrame(
            {"movie_n_ratings": [1.0, 50.0, 40.0, 30.0, 20.0, 10.0]},
            index=pd.Index([9, 1, 2, 3, 4, 5], name="movie_id"),
        )


def test_model_returning_nothing_is_not_put_live():
    registry = Registry(recommender=StubRecommender(known=[999]))  # [] for every visitor
    with make_client(registry) as client:
        ready = client.get("/ready").json()
        assert ready["model"]["loaded"] is False
        assert "no recommendations" in ready["model"]["last_error"]


def test_model_is_validated_on_its_own_movies_when_the_catalog_is_empty():
    repo = FakeRepository()
    repo.down = True  # catalog cannot be loaded at start-up
    recommender = ModelWithMovies()
    with make_client(Registry(recommender=recommender), repo=repo) as client:
        assert client.get("/ready").json()["model"]["version"] == "7"
        assert sorted(m for m, _ in recommender.calls[0]) == [1, 2, 3, 4, 5]


def test_model_without_any_movie_to_validate_on_is_not_put_live():
    repo = FakeRepository()
    repo.down = True
    with make_client(repo=repo) as client:  # plain stub: no catalog, no movie_feats
        ready = client.get("/ready").json()
        assert ready["model"]["loaded"] is False
        assert "validate" in ready["model"]["last_error"]


def test_recommendations_are_rate_limited_per_visitor():
    """s5-review I-4: every response with a cookie writes impressions; a script must not be
    able to call /recommendations without bound. Separate (generous) budget from ratings."""
    with make_client(recommendations_rate_limit_per_minute=3, rate_limit_per_minute=100) as c:
        assert [c.get("/recommendations").status_code for _ in range(3)] == [200] * 3
        limited = c.get("/recommendations")
        assert limited.status_code == 429 and limited.headers["retry-after"] == "60"
        assert rate(c, 1).status_code == 200  # ratings have their own counter


def test_recommendations_limit_is_off_while_redis_is_down():
    store = FakeStore()
    store.down = True
    with make_client(store=store, recommendations_rate_limit_per_minute=1) as client:
        assert [client.get("/recommendations").status_code for _ in range(5)] == [200] * 5


def test_recommendations_limit_default_is_generous():
    assert Settings(session_secret=SESSION_SECRET).recommendations_rate_limit_per_minute == 120
