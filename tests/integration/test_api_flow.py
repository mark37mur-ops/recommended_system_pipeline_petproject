"""The API against real Postgres and Redis (spec 6.5), with a real two-stage model trained on
synthetic data (fold-in path), injected instead of MLflow."""

import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from synthetic import make_movies, make_ratings, make_users, write_raw_dir

from recsys.db.load_movielens import load_movielens
from recsys.db.schema import WEB_USER_ID_START, ratings, users
from recsys.features import movie_features, movie_genre_matrix, user_features
from recsys.models.als import ALSCandidateGenerator, ALSConfig
from recsys.models.two_stage import TwoStageRecommender
from recsys_api.cache import RedisStore
from recsys_api.identity import COOKIE_NAME, CookieSigner
from recsys_api.main import create_app
from recsys_api.repository import PgRepository
from recsys_api.service import RatingQuotaExceeded, UserNotFound
from recsys_api.settings import Settings

from .conftest import REDIS_URL

VERSION = "it-1"
SESSION_SECRET = "integration-session-secret-0123456789abcdef"
ADMIN_TOKEN = "integration-admin-token-0123456789abcdef"


class Champion:
    def __init__(self, recommender):
        self.recommender, self.version, self.run_id = recommender, VERSION, "run-it"


@pytest.fixture(scope="module")
def recommender() -> TwoStageRecommender:
    data, movies = make_ratings(), make_movies()
    als = ALSCandidateGenerator(ALSConfig(factors=8, iterations=10, num_threads=1)).fit(data)
    genres = movie_genre_matrix(movies)
    return TwoStageRecommender(
        als, movie_features(data, movies), user_features(data, genres, make_users()), None, 30
    )


@pytest.fixture
def loaded_db(db, tmp_path):
    load_movielens(db, write_raw_dir(tmp_path / "raw"))
    return db


def make_client(engine, recommender, redis_url=REDIS_URL, **overrides) -> TestClient:
    settings = Settings(session_secret=SESSION_SECRET, admin_token=ADMIN_TOKEN, **overrides)
    app = create_app(
        settings,
        repo=PgRepository(engine),
        store=RedisStore(redis_url),
        loader=lambda: Champion(recommender),
        probe=lambda: VERSION,
        background=False,
    )
    return TestClient(app)


def web_user_ids(engine) -> list[int]:
    with engine.connect() as conn:
        query = select(users.c.user_id).where(users.c.source == "web").order_by(users.c.user_id)
        return list(conn.execute(query).scalars())


def test_new_visitor_gets_fold_in_recommendations_and_cache(loaded_db, recommender, redis_client):
    with make_client(loaded_db, recommender) as client:
        assert client.get("/ready").json()["status"] == "ready"
        rated = [3, 6, 9, 12, 15]  # all "Comedy" in the synthetic catalog (id % 3 == 0)
        for movie_id in rated:
            assert (
                client.post("/ratings", json={"movie_id": movie_id, "rating": 5}).status_code == 200
            )
        (user_id,) = web_user_ids(loaded_db)
        assert user_id >= WEB_USER_ID_START

        first = client.get("/recommendations").json()
        assert first["path"] == "full" and first["model_version"] == VERSION
        assert first["cached"] is False and first["n_ratings"] == 5
        ids = [item["movie_id"] for item in first["items"]]
        assert len(ids) == 10 and not set(ids) & set(rated)
        assert ids == recommender.recommend_new_user([(m, 5) for m in rated], k=10)
        key = f"recs:{user_id}:{VERSION}"
        assert 0 < redis_client.ttl(key) <= 600

        assert client.get("/recommendations").json()["cached"] is True
        client.post("/ratings", json={"movie_id": ids[0], "rating": 1})
        assert not redis_client.exists(key)
        fresh = client.get("/recommendations").json()
        assert fresh["cached"] is False and ids[0] not in [i["movie_id"] for i in fresh["items"]]


def test_rerating_updates_the_same_row(loaded_db, recommender, redis_client):
    with make_client(loaded_db, recommender) as client:
        client.post("/ratings", json={"movie_id": 7, "rating": 2})
        response = client.post("/ratings", json={"movie_id": 7, "rating": 4})
        assert response.json()["n_ratings"] == 1
    (user_id,) = web_user_ids(loaded_db)
    with loaded_db.connect() as conn:
        rows = conn.execute(
            select(ratings.c.rating, ratings.c.source).where(ratings.c.user_id == user_id)
        ).all()
    assert [tuple(r) for r in rows] == [(4, "web")]


def test_rate_limit_with_real_redis(loaded_db, recommender, redis_client):
    with make_client(loaded_db, recommender, rate_limit_per_minute=3) as client:
        codes = [
            client.post("/ratings", json={"movie_id": m, "rating": 4}).status_code
            for m in (1, 2, 3, 4)
        ]
    assert codes == [200, 200, 200, 429]


def test_quota_holds_under_concurrent_ratings(loaded_db):
    repo = PgRepository(loaded_db)
    user_id = repo.create_web_user()
    accepted, rejected = [], []

    def rate(movie_id):
        try:
            repo.upsert_rating(user_id, movie_id, 5, max_ratings=5)
            accepted.append(movie_id)
        except RatingQuotaExceeded:
            rejected.append(movie_id)

    threads = [threading.Thread(target=rate, args=(m,)) for m in range(1, 13)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(accepted) == 5 and len(rejected) == 7
    with loaded_db.connect() as conn:
        count = select(func.count()).where(ratings.c.user_id == user_id)
        assert conn.execute(count).scalar_one() == 5


def test_quota_returns_403(loaded_db, recommender, redis_client):
    with make_client(loaded_db, recommender, max_ratings_per_user=2) as client:
        codes = [
            client.post("/ratings", json={"movie_id": m, "rating": 4}).status_code
            for m in (1, 2, 3)
        ]
    assert codes == [200, 200, 403]


def test_service_works_without_redis(loaded_db, recommender):
    dead_redis = "redis://127.0.0.1:1/0"
    with make_client(loaded_db, recommender, redis_url=dead_redis, rate_limit_per_minute=1) as c:
        codes = [
            c.post("/ratings", json={"movie_id": m, "rating": 5}).status_code
            for m in (3, 6, 9, 12, 15)
        ]
        assert codes == [200] * 5  # rate limiting is off, not failing
        recs = c.get("/recommendations").json()
        assert recs["path"] == "full" and recs["cached"] is False
        ready = c.get("/ready").json()
        assert ready["status"] == "ready" and ready["redis"] == "unavailable"


def test_catalog_popularity_counts_all_ratings(loaded_db):
    movies = {m.movie_id: m for m in PgRepository(loaded_db).load_movies()}
    expected = make_ratings().groupby("movie_id").size()
    assert len(movies) == len(make_movies())
    assert all(movies[int(m)].n_ratings == int(n) for m, n in expected.items())
    assert movies[5].genres == ("Horror", "Romance")


def movielens_rows(engine, user_id: int) -> list[tuple]:
    query = (
        select(ratings.c.movie_id, ratings.c.rating, ratings.c.source)
        .where(ratings.c.user_id == user_id)
        .order_by(ratings.c.movie_id)
    )
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(query)]


def test_upsert_never_writes_into_a_movielens_user(loaded_db):
    """Review I3: the row lock is taken on web users only, so even a MovieLens id that got
    past the cookie check cannot overwrite (or add to) the training ratings."""
    repo = PgRepository(loaded_db)
    before = movielens_rows(loaded_db, 1)
    movie_id, rating, source = before[0]
    assert source == "movielens"
    rated = {r[0] for r in before}
    unrated = next(int(m) for m in make_movies()["movie_id"] if int(m) not in rated)
    for target in (movie_id, unrated):
        with pytest.raises(UserNotFound):
            repo.upsert_rating(1, target, 1 if rating != 1 else 2, max_ratings=300)
    assert movielens_rows(loaded_db, 1) == before


def test_signed_cookie_for_a_movielens_user_creates_a_web_user(
    loaded_db, recommender, redis_client
):
    before = movielens_rows(loaded_db, 1)
    movie_id, rating, _ = before[0]
    with make_client(loaded_db, recommender) as client:
        client.cookies.set(COOKIE_NAME, CookieSigner(SESSION_SECRET).sign(1))
        new = 1 if rating != 1 else 2
        response = client.post("/ratings", json={"movie_id": movie_id, "rating": new})
        assert response.status_code == 200 and response.json()["n_ratings"] == 1
    assert movielens_rows(loaded_db, 1) == before
    (web_user,) = web_user_ids(loaded_db)
    assert movielens_rows(loaded_db, web_user) == [(movie_id, new, "web")]
