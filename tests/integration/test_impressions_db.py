"""recommendation_impressions, the online metric views and the dashboard grants on Postgres."""

from datetime import UTC, datetime, timedelta

import pytest
from api_fakes import StubRecommender
from sqlalchemy import func, select, text
from synthetic import write_raw_dir

from recsys.db.load_movielens import load_movielens
from recsys.db.schema import (
    DASHBOARD_OBJECTS,
    DASHBOARD_READER,
    ONLINE_METRIC_WINDOW,
    WEB_USER_ID_START,
    create_schema,
    ratings,
    recommendation_impressions,
    users,
)
from recsys_api.impressions import Impression, ImpressionLogger
from recsys_api.metrics import ApiMetrics
from recsys_api.repository import PgRepository

from .test_api_flow import make_client

# Relative to today: the views only look at the last ONLINE_METRIC_WINDOW (and at impressions
# older than an hour). Two days ago at 10:00 keeps every test row within one calendar day.
T0 = (datetime.now(UTC) - timedelta(days=2)).replace(hour=10, minute=0, second=0, microsecond=0)
U1, U2 = WEB_USER_ID_START + 500, WEB_USER_ID_START + 501  # clear of the web-id sequence


@pytest.fixture
def loaded_db(db, tmp_path):
    load_movielens(db, write_raw_dir(tmp_path / "raw"))
    with db.begin() as conn:
        conn.execute(
            users.insert(), [{"user_id": U1, "source": "web"}, {"user_id": U2, "source": "web"}]
        )
    return db


def rate_at(engine, user_id, movie_id, rating, at):
    with engine.begin() as conn:
        conn.execute(
            ratings.insert().values(
                user_id=user_id, movie_id=movie_id, rating=rating, source="web", rated_at=at
            )
        )


def shown(user_id, movie_ids, at, path="full", version="6"):
    return [Impression(user_id, m, i, version, path, at) for i, m in enumerate(movie_ids, 1)]


def test_batch_insert_through_the_logger(loaded_db):
    repo, metrics = PgRepository(loaded_db), ApiMetrics()
    logger = ImpressionLogger(repo.insert_impressions, metrics=metrics)
    logger.record(U1, [5, 6, 7], "full", "6", shown_at=T0)
    logger.record(U1, [8], "cold_start", None, shown_at=T0)
    logger.drain()
    with loaded_db.connect() as conn:
        rows = conn.execute(
            select(
                recommendation_impressions.c.movie_id,
                recommendation_impressions.c.position,
                recommendation_impressions.c.model_version,
                recommendation_impressions.c.path,
                recommendation_impressions.c.shown_at,
            ).order_by(recommendation_impressions.c.impression_id)
        ).all()
    assert [tuple(r) for r in rows] == [
        (5, 1, "6", "full", T0),
        (6, 2, "6", "full", T0),
        (7, 3, "6", "full", T0),
        (8, 1, None, "cold_start", T0),
    ]
    assert (
        metrics.registry.get_sample_value("recommendation_impressions_total", {"result": "written"})
        == 4
    )


def test_online_metric_counts_first_impressions_rated_within_an_hour(loaded_db):
    repo = PgRepository(loaded_db)
    repo.insert_impressions(shown(U1, [1, 2, 3, 4], T0))
    repo.insert_impressions(shown(U1, [1, 2, 3, 4], T0 + timedelta(minutes=10)))  # repeat
    repo.insert_impressions(shown(U2, [5, 6], T0, path="cold_start", version=None))
    rate_at(loaded_db, U1, 1, 5, T0 + timedelta(minutes=30))  # within the hour of the 1st show
    rate_at(loaded_db, U1, 2, 4, T0 + timedelta(hours=1))  # exactly 1 h: outside (half-open)
    rate_at(loaded_db, U1, 3, 2, T0)  # at the very moment it was shown: inside
    rate_at(loaded_db, U2, 5, 1, T0 + timedelta(hours=3))  # far too late
    # U1 / movie 4 never rated, U2 / movie 6 never rated

    with loaded_db.connect() as conn:
        outcomes = conn.execute(
            text(
                "SELECT user_id, movie_id, shown_at, rated_within_hour FROM recommendation_outcomes"
                " ORDER BY user_id, movie_id"
            )
        ).all()
        overall = conn.execute(
            text(
                "SELECT avg(rated_within_hour::int), avg(rating) FILTER (WHERE rated_within_hour)"
                " FROM recommendation_outcomes WHERE path = 'full'"
            )
        ).one()
        daily = conn.execute(
            text(
                "SELECT path, model_version, n_shown, n_rated_within_hour,"
                " share_rated_within_hour, avg_rating FROM online_quality_daily ORDER BY path"
            )
        ).all()

    # one row per (user, movie), at the first impression; repeats do not count twice
    assert [(u, m, s) for u, m, s, _ in outcomes] == [
        (U1, 1, T0),
        (U1, 2, T0),
        (U1, 3, T0),
        (U1, 4, T0),
        (U2, 5, T0),
        (U2, 6, T0),
    ]
    assert [r for *_, r in outcomes] == [True, False, True, False, False, False]
    assert float(overall[0]) == pytest.approx(0.5)  # 2 of 4 "full" impressions
    assert float(overall[1]) == pytest.approx((5 + 2) / 2)
    assert [tuple(r) for r in daily] == [
        ("cold_start", None, 2, 0, 0.0, None),
        ("full", "6", 4, 2, 0.5, 3.5),
    ]


def test_api_writes_impressions_to_postgres(loaded_db, redis_client):
    with make_client(loaded_db, StubRecommender()) as client:
        for movie in (3, 6, 9, 12, 15):
            assert client.post("/ratings", json={"movie_id": movie, "rating": 5}).status_code == 200
        body = client.get("/recommendations", params={"k": 5}).json()
        assert client.app.state.impressions.flush(timeout=10)
    with loaded_db.connect() as conn:
        rows = conn.execute(
            select(
                recommendation_impressions.c.movie_id, recommendation_impressions.c.path
            ).order_by(recommendation_impressions.c.position)
        ).all()
        n = conn.execute(select(func.count()).select_from(recommendation_impressions)).scalar()
    assert n == 5
    assert [m for m, _ in rows] == [item["movie_id"] for item in body["items"]]
    assert {p for _, p in rows} == {body["path"]} == {"full"}


def test_dashboard_role_reads_monitoring_objects_only(db):
    with db.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": DASHBOARD_READER}
        ).first()
        if exists is None:  # CI: no init script; compose: created by infra/postgres/init
            conn.execute(text(f"CREATE ROLE {DASHBOARD_READER} NOLOGIN"))
        # granted by an older schema version (stage 5 before review): must be taken back
        conn.execute(text(f"GRANT SELECT ON recommendation_outcomes TO {DASHBOARD_READER}"))
    create_schema(db)  # grants on every run, also when the role appeared after the tables
    with db.connect() as conn:

        def can_read(obj):
            return conn.execute(
                text("SELECT has_table_privilege(:r, :o, 'SELECT')"),
                {"r": DASHBOARD_READER, "o": obj},
            ).scalar()

        assert all(can_read(obj) for obj in DASHBOARD_OBJECTS)
        private = ("ratings", "users", "recommendation_impressions", "recommendation_outcomes")
        assert not any(can_read(obj) for obj in private)


# ----- review fixes (s5-review I-3, I-4, M-1) -------------------------------------------------


def outcomes(engine, where="true"):
    with engine.connect() as conn:
        return [
            tuple(r)
            for r in conn.execute(
                text(
                    "SELECT user_id, movie_id, path, model_version, shown_at, rated_within_hour"
                    f" FROM recommendation_outcomes WHERE {where}"
                    " ORDER BY user_id, movie_id, path, model_version"
                )
            )
        ]


def daily(engine):
    with engine.connect() as conn:
        return [
            tuple(r)
            for r in conn.execute(
                text(
                    "SELECT path, model_version, n_shown, n_rated_within_hour"
                    " FROM online_quality_daily ORDER BY path, model_version"
                )
            )
        ]


def test_each_path_and_model_version_gets_its_own_first_impression(loaded_db):
    """A pair first shown in cold_start and later by the model counts for the model too; a
    pair already shown under v6 counts for v7 when v7 shows it."""
    repo = PgRepository(loaded_db)
    minute = timedelta(minutes=1)
    repo.insert_impressions(shown(U1, [1, 2], T0, path="cold_start", version=None))
    repo.insert_impressions(shown(U1, [1, 2], T0 + 5 * minute, path="full", version="6"))
    repo.insert_impressions(shown(U1, [1], T0 + 7 * minute, path="full", version="6"))  # repeat
    repo.insert_impressions(shown(U1, [2], T0 + 9 * minute, path="full", version="7"))
    rate_at(loaded_db, U1, 1, 5, T0 + 6 * minute)  # after the cold-start and the v6 showing
    rate_at(loaded_db, U1, 2, 4, T0 + 65 * minute)  # >= 1 h after cold start / v6, < 1 h after v7

    assert outcomes(loaded_db) == [
        (U1, 1, "cold_start", None, T0, True),
        (U1, 1, "full", "6", T0 + 5 * minute, True),
        (U1, 2, "cold_start", None, T0, False),
        (U1, 2, "full", "6", T0 + 5 * minute, False),
        (U1, 2, "full", "7", T0 + 9 * minute, True),
    ]
    assert daily(loaded_db) == [
        ("cold_start", None, 2, 1),
        ("full", "6", 2, 1),
        ("full", "7", 1, 1),
    ]


def test_views_only_read_the_recent_window(loaded_db):
    """The view sorts only the last ONLINE_METRIC_WINDOW of impressions (s5-review I-4)."""
    repo = PgRepository(loaded_db)
    old = datetime.now(UTC) - ONLINE_METRIC_WINDOW - timedelta(days=1)
    repo.insert_impressions(shown(U1, [1], old))
    repo.insert_impressions(shown(U1, [1, 2], T0))
    assert [(m, at) for _, m, _, _, at, _ in outcomes(loaded_db)] == [(1, T0), (2, T0)]


def test_daily_quality_counts_only_matured_impressions(loaded_db):
    """An impression younger than an hour has no final outcome yet (s5-review M-1)."""
    repo = PgRepository(loaded_db)
    repo.insert_impressions(shown(U1, [1, 2], T0))
    repo.insert_impressions(shown(U2, [3], datetime.now(UTC) - timedelta(minutes=10)))
    assert daily(loaded_db) == [("full", "6", 2, 0)]
    assert len(outcomes(loaded_db)) == 3  # the raw view still has it


def test_old_impressions_are_cleaned_up_on_postgres(loaded_db):
    from recsys.db.monitoring import IMPRESSIONS_RETENTION, delete_old_impressions

    repo = PgRepository(loaded_db)
    expired = datetime.now(UTC) - IMPRESSIONS_RETENTION - timedelta(days=1)
    repo.insert_impressions(shown(U1, [1], expired))
    repo.insert_impressions(shown(U1, [2], T0))
    assert delete_old_impressions(loaded_db) == 1
    assert delete_old_impressions(loaded_db) == 0
    with loaded_db.connect() as conn:
        left = conn.execute(select(recommendation_impressions.c.movie_id)).scalars().all()
    assert left == [2]
