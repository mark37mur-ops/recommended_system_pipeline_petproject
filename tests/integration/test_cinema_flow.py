"""Cinema contracts against additive SQL schema and a real Redis cache."""

from datetime import UTC, datetime

from api_fakes import FakeChampion, StubRecommender
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from synthetic import write_raw_dir

from recsys.db.load_movielens import load_movielens
from recsys.db.schema import (
    WEB_USER_ID_START,
    create_schema,
    movie_feedback,
    movie_metadata,
    movies,
    ratings,
    users,
)
from recsys_api.cache import RedisStore
from recsys_api.identity import COOKIE_NAME, CookieSigner
from recsys_api.main import create_app
from recsys_api.repository import PgRepository
from recsys_api.settings import Settings

from .conftest import REDIS_URL

SECRET = "cinema-test-session-secret-0123456789abcdef"


def client_for(engine, store=None):
    return TestClient(
        create_app(
            Settings(
                session_secret=SECRET,
                impressions_enabled=False,
                rate_limit_per_minute=1000,
                recommendations_rate_limit_per_minute=1000,
            ),
            repo=PgRepository(engine),
            store=store or RedisStore(REDIS_URL),
            loader=lambda: FakeChampion(StubRecommender(), "cinema-test"),
            probe=lambda: "cinema-test",
            background=False,
        )
    )


def test_existing_cookie_survives_additive_upgrade(db):
    # Reproduce an old populated database: only the newly introduced empty tables are absent.
    movie_feedback.drop(db)
    movie_metadata.drop(db)
    now = datetime.now(UTC)
    with db.begin() as conn:
        conn.execute(
            movies.insert().values(movie_id=1, title="Original title", year=1999, genres="Drama")
        )
        conn.execute(users.insert().values(user_id=WEB_USER_ID_START, source="web"))
        conn.execute(
            ratings.insert().values(
                user_id=WEB_USER_ID_START, movie_id=1, rating=4, source="web", rated_at=now
            )
        )
    cookie = CookieSigner(SECRET).sign(WEB_USER_ID_START)
    create_schema(db)
    create_schema(db)
    for _ in range(2):
        with client_for(db) as c:
            c.cookies.set(COOKIE_NAME, cookie)
            response = c.get("/ratings")
            assert response.status_code == 200
            assert response.json()["n_ratings"] == 1
            assert response.json()["items"][0]["user_rating"] == 4
            assert not response.headers.get("set-cookie")
    with db.connect() as conn:
        assert conn.scalar(select(movies.c.title)) == "Original title"
        assert conn.scalar(select(ratings.c.rating)) == 4


def test_five_unique_ratings_and_intent_flow(db, tmp_path, redis_client):
    load_movielens(db, write_raw_dir(tmp_path / "raw"))
    store = RedisStore(REDIS_URL, cooldown=0)
    with client_for(db, store) as alice, client_for(db) as bob:
        for mid in (1, 2, 3, 4, 1):
            response = alice.post("/ratings", json={"movie_id": mid, "rating": 4})
            assert response.status_code == 200
        assert response.json()["n_ratings"] == 4
        assert alice.post("/ratings", json={"movie_id": 5, "rating": 5}).json()["n_ratings"] == 5

        def recs():
            response = alice.get("/recommendations?k=50")
            assert response.status_code == 200
            return response.json()

        before = recs()
        assert recs()["cached"]
        mid = before["items"][0]["movie_id"]
        assert (
            alice.post("/feedback", json={"movie_id": mid, "state": "watchlist"}).status_code == 200
        )
        assert alice.get("/ratings").json()["n_ratings"] == 5
        assert mid not in {m["movie_id"] for m in recs()["items"]}
        assert alice.get("/watchlist").json()["items"][0]["movie_id"] == mid
        assert bob.get("/watchlist").json()["items"] == []
        assert alice.delete(f"/watchlist/{mid}").status_code == 204
        assert mid in {m["movie_id"] for m in recs()["items"]}
        alice.post("/feedback", json={"movie_id": mid, "state": "watchlist"})
        alice.post("/ratings", json={"movie_id": mid, "rating": 3})
        assert alice.get("/watchlist").json()["items"] == []
        rejected = next(m["movie_id"] for m in recs()["items"])
        # Use an unavailable Redis transport, preserving the same service's recovery state.
        real_client = store._client
        import redis

        store._client = redis.Redis.from_url("redis://127.0.0.1:1/15", socket_connect_timeout=0.1)
        try:
            assert (
                alice.post(
                    "/feedback", json={"movie_id": rejected, "state": "dismissed"}
                ).status_code
                == 200
            )
            assert rejected not in {m["movie_id"] for m in recs()["items"]}
        finally:
            store._client.close()
            store._client = real_client
        assert rejected not in {m["movie_id"] for m in recs()["items"]}
        assert rejected not in {m["movie_id"] for m in recs()["items"]}
        cookie = alice.cookies.get(COOKIE_NAME)
    with client_for(db) as restored:
        restored.cookies.set(COOKIE_NAME, cookie)
        assert restored.get("/ratings").json()["n_ratings"] == 6
        assert restored.get("/watchlist").json()["items"] == []
    with db.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(movie_feedback)) == 1
