"""recsys.db.monitoring on SQLite: impressions for prediction drift and the retention clean-up."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, func, select

from recsys.db.monitoring import IMPRESSIONS_RETENTION, delete_old_impressions, recent_impressions
from recsys.db.schema import ONLINE_METRIC_WINDOW, WEB_USER_ID_START, recommendation_impressions
from recsys.drift import CURRENT_WINDOW

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
U = WEB_USER_ID_START


def engine_with(rows):
    engine = create_engine("sqlite://")
    recommendation_impressions.create(engine)
    with engine.begin() as conn:
        conn.execute(
            recommendation_impressions.insert(),
            [
                {
                    "user_id": user,
                    "movie_id": movie,
                    "position": 1,
                    "model_version": "6",
                    "path": path,
                    "shown_at": at,
                }
                for user, movie, path, at in rows
            ],
        )
    return engine


def count(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(recommendation_impressions)).scalar()


def test_recent_impressions_count_each_pair_once():
    """s5-review M-5: one visitor refreshing the page must not dominate prediction drift."""
    hour = timedelta(hours=1)
    engine = engine_with(
        [
            (U, 1, "full", NOW - 3 * hour),
            (U, 1, "full", NOW - 2 * hour),  # repeat of the same pair
            (U, 2, "full", NOW - 2 * hour),
            (U + 1, 1, "full", NOW - hour),  # another visitor, same movie: counts
            (U, 3, "cold_start", NOW - hour),  # not the model's prediction
            (U, 4, "full", NOW - 10 * 24 * hour),  # outside the window
        ]
    )
    assert recent_impressions(engine, since=NOW - CURRENT_WINDOW, until=NOW) == [1, 2, 1]


def test_old_impressions_are_deleted_idempotently():
    day = timedelta(days=1)
    engine = engine_with(
        [
            (U, 1, "full", NOW - IMPRESSIONS_RETENTION - day),
            (U, 2, "full", NOW - IMPRESSIONS_RETENTION - timedelta(seconds=1)),
            (U, 3, "full", NOW - IMPRESSIONS_RETENTION + day),
            (U, 4, "full", NOW),
        ]
    )
    assert delete_old_impressions(engine, now=NOW) == 2
    assert count(engine) == 2
    assert delete_old_impressions(engine, now=NOW) == 0  # a retry deletes nothing more
    assert count(engine) == 2


def test_cleanup_without_the_table_is_a_no_op():
    assert delete_old_impressions(create_engine("sqlite://"), now=NOW) == 0


def test_retention_outlives_every_reader():
    """The online metric views (ONLINE_METRIC_WINDOW) and drift (CURRENT_WINDOW) must never
    lose rows to the clean-up."""
    assert IMPRESSIONS_RETENTION > ONLINE_METRIC_WINDOW > CURRENT_WINDOW
