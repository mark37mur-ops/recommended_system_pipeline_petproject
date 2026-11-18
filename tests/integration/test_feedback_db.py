import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import IntegrityError

from recsys.db.schema import movie_feedback, movies, ratings, users
from recsys_api.repository import PgRepository
from recsys_api.service import MovieAlreadyRated, RatingQuotaExceeded, UserNotFound


@pytest.fixture
def repo(db):
    with db.begin() as conn:
        conn.execute(movies.insert(), [{"movie_id": i, "title": f"Movie {i}"} for i in range(1, 5)])
    return PgRepository(db)


@pytest.mark.parametrize("first", ["rating", "feedback"])
def test_concurrent_rating_feedback_rating_wins(repo, first):
    user = repo.create_web_user()
    locked, release, second_started = threading.Event(), threading.Event(), threading.Event()

    def after_execute(conn, cursor, statement, parameters, context, executemany):
        if "FOR UPDATE" in statement and not locked.is_set():
            locked.set()
            assert release.wait(10)

    def before_execute(conn, cursor, statement, parameters, context, executemany):
        if "FOR UPDATE" in statement and locked.is_set():
            second_started.set()

    def rating():
        repo.upsert_rating(user, 1, 4, 300)

    def feedback():
        try:
            repo.upsert_feedback(user, 1, "watchlist")
        except MovieAlreadyRated:
            return "rated"
        return "saved"

    def second():
        return feedback() if first == "rating" else rating()

    event.listen(repo.engine, "after_cursor_execute", after_execute)
    event.listen(repo.engine, "before_cursor_execute", before_execute)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(rating if first == "rating" else feedback)
            try:
                assert locked.wait(10)
                b = pool.submit(second)
                assert second_started.wait(10)
            finally:
                release.set()
            result_a, result_b = a.result(timeout=10), b.result(timeout=10)
    finally:
        release.set()
        event.remove(repo.engine, "after_cursor_execute", after_execute)
        event.remove(repo.engine, "before_cursor_execute", before_execute)
    assert (result_b if first == "rating" else result_a) == (
        "rated" if first == "rating" else "saved"
    )
    assert repo.user_ratings(user) == [(1, 4)]
    assert repo.user_feedback(user) == {}


def test_failed_rating_rolls_back_feedback_delete(repo):
    user = repo.create_web_user()
    repo.upsert_feedback(user, 1, "watchlist")
    with pytest.raises(IntegrityError):
        repo.upsert_rating(user, 1, 9, 300)
    assert repo.user_feedback(user) == {1: "watchlist"}
    assert repo.user_ratings(user) == []
    repo.upsert_rating(user, 2, 5, 1)
    with pytest.raises(RatingQuotaExceeded):
        repo.upsert_rating(user, 1, 4, 1)
    assert repo.user_feedback(user) == {1: "watchlist"}


def test_user_isolation_and_fk(repo):
    one, two = repo.create_web_user(), repo.create_web_user()
    repo.upsert_feedback(one, 1, "watchlist")
    repo.delete_watchlist(two, 1)
    assert repo.user_feedback(one) == {1: "watchlist"}
    assert repo.user_feedback(two) == {}
    with pytest.raises(IntegrityError):
        repo.upsert_feedback(two, 999, "watchlist")
    with repo.engine.begin() as conn:
        conn.execute(users.insert().values(user_id=42, source="movielens"))
    for user in [42, two + 100]:
        with pytest.raises(UserNotFound):
            repo.upsert_feedback(user, 1, "dismissed")
        with pytest.raises(UserNotFound):
            repo.delete_watchlist(user, 1)
        with pytest.raises(UserNotFound):
            repo.upsert_rating(user, 1, 4, 300)
    with repo.engine.begin() as conn:
        conn.execute(users.delete().where(users.c.user_id == one))
    assert repo.user_feedback(one) == {}


def test_retry_upsert_idempotent(repo):
    user = repo.create_web_user()
    repo.upsert_feedback(user, 1, "watchlist")
    repo.upsert_feedback(user, 1, "watchlist")
    repo.upsert_feedback(user, 1, "dismissed")
    repo.delete_watchlist(user, 1)
    assert repo.user_feedback(user) == {1: "dismissed"}
    repo.upsert_feedback(user, 1, "watchlist")
    repo.delete_watchlist(user, 1)
    repo.delete_watchlist(user, 1)
    assert repo.user_feedback(user) == {}
    repo.upsert_feedback(user, 1, "dismissed")
    repo.upsert_rating(user, 1, 4, 300)
    with pytest.raises(MovieAlreadyRated):
        repo.upsert_feedback(user, 1, "dismissed")
    assert repo.user_feedback(user) == {}
    with repo.engine.connect() as conn:
        assert (
            conn.execute(select(ratings.c.rating).where(ratings.c.user_id == user)).scalar_one()
            == 4
        )


def test_feedback_order_newest_then_movie_id(repo):
    user = repo.create_web_user()
    now = datetime.now(UTC)
    with repo.engine.begin() as conn:
        conn.execute(
            movie_feedback.insert(),
            [
                {"user_id": user, "movie_id": 3, "state": "watchlist", "updated_at": now},
                {
                    "user_id": user,
                    "movie_id": 1,
                    "state": "dismissed",
                    "updated_at": now - timedelta(days=1),
                },
                {"user_id": user, "movie_id": 2, "state": "watchlist", "updated_at": now},
            ],
        )
    assert list(repo.user_feedback(user)) == [2, 3, 1]
