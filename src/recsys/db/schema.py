"""Schema of the `app` database (spec 1, 3.4): movies, users, ratings; retrain_runs (stage 4);
recommendation_impressions, drift_metrics, run_metrics and two views for Grafana (stage 5).

SQLAlchemy Core, created idempotently with `create_schema` (`metadata.create_all`): the schema
is small and only gains new tables before v2, which `create_all` adds; Alembic becomes worth it
at the first ALTER of an existing table.

Web visitors get ids from their own sequence starting at WEB_USER_ID_START, far above the
MovieLens ids (1..6040), and a CHECK constraint ties `source` to the id range, so a web user can
never collide with (or be mistaken for) a MovieLens user.

The monitoring views are plain Postgres DDL hooked to `metadata` events, so `create_all`
creates (or replaces) them after the tables and `drop_all` drops them first. Grafana reads the
database as the read-only role DASHBOARD_READER (created by infra/postgres/init), which is
granted SELECT on DASHBOARD_OBJECTS only: aggregates and metrics, not raw ratings and users.
"""

from sqlalchemy import (
    DDL,
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Sequence,
    SmallInteger,
    String,
    Table,
    Text,
    create_engine,
    event,
    func,
)

WEB_USER_ID_START = 1_000_000
SOURCES = ("movielens", "web")
_SOURCE_CHECK = "source IN ('movielens', 'web')"

metadata = MetaData()

web_user_id_seq = Sequence("web_user_id_seq", start=WEB_USER_ID_START, metadata=metadata)

movies = Table(
    "movies",
    metadata,
    Column("movie_id", Integer, primary_key=True, autoincrement=False),
    Column("title", Text, nullable=False),
    Column("year", SmallInteger, nullable=True),
    Column("genres", Text, nullable=False, server_default=""),  # "Action|Comedy", ML-1M order
)

users = Table(
    "users",
    metadata,
    Column("user_id", Integer, primary_key=True, autoincrement=False),
    Column("source", Text, nullable=False),
    Column("gender", String(1), nullable=True),
    Column("age", SmallInteger, nullable=True),
    Column("occupation", SmallInteger, nullable=True),
    Column("zip", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint(_SOURCE_CHECK, name="users_source_valid"),
    CheckConstraint(
        f"(source = 'web') = (user_id >= {WEB_USER_ID_START})", name="users_web_id_range"
    ),
)

ratings = Table(
    "ratings",
    metadata,
    Column("user_id", Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
    Column("movie_id", Integer, ForeignKey("movies.movie_id"), nullable=False),
    Column("rating", SmallInteger, nullable=False),
    Column("source", Text, nullable=False),
    Column("rated_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    PrimaryKeyConstraint("user_id", "movie_id", name="ratings_pkey"),  # one rating per pair
    CheckConstraint("rating BETWEEN 1 AND 5", name="ratings_rating_range"),
    CheckConstraint(_SOURCE_CHECK, name="ratings_source_valid"),
    Index("ratings_movie_id_idx", "movie_id"),
)

# One row per `retrain_daily` run whose snapshot passed validation: the "previous run" of the
# next one (the -5% row-count rule, drift against the previous run). Written by validate_data.
retrain_runs = Table(
    "retrain_runs",
    metadata,
    Column("run_id", Text, primary_key=True),
    Column("include_web", Boolean, nullable=False),
    Column("n_ratings", Integer, nullable=False),
    Column("data_snapshot", Text, nullable=False),
    Column("summary", JSON, nullable=False),  # recsys.drift.summarize_ratings
    Column("recorded_at", DateTime(timezone=True), nullable=False),
)


# Stage 5 (spec 5.2): what was shown to whom. Written in batches off the request path by the API
# (recsys_api.impressions). No foreign keys on purpose: it is a log, and one row for a user
# deleted meanwhile must not fail the whole batch. Only visitors with an id (cookie) are logged:
# an anonymous impression cannot be joined to a later rating.
recommendation_impressions = Table(
    "recommendation_impressions",
    metadata,
    Column(
        "impression_id",
        BigInteger().with_variant(Integer, "sqlite"),  # SQLite autoincrements INTEGER only
        primary_key=True,
        autoincrement=True,
    ),
    Column("user_id", Integer, nullable=False),
    Column("movie_id", Integer, nullable=False),
    Column("position", SmallInteger, nullable=False),  # 1 = first in the list
    Column("model_version", Text, nullable=True),  # None: no model loaded
    Column("path", Text, nullable=False),  # full / cold_start / fallback (spec 5.1)
    Column("shown_at", DateTime(timezone=True), nullable=False),
    Index("recommendation_impressions_user_movie_idx", "user_id", "movie_id", "shown_at"),
    Index("recommendation_impressions_shown_at_idx", "shown_at"),
)

# Stage 5 (spec 5.3): one row per retrain_daily run, written by drift_report. NULL shares mean
# "not enough data to say" (no or too few web ratings / impressions), not "no drift".
drift_metrics = Table(
    "drift_metrics",
    metadata,
    Column("run_id", Text, primary_key=True),
    Column("computed_at", DateTime(timezone=True), nullable=False),
    Column("include_web", Boolean, nullable=False),
    Column("drift_share", Float, nullable=True),  # share of drifted rating features
    Column("prediction_drift_share", Float, nullable=True),  # ... of recommendation features
    Column("ratings_status", Text, nullable=False),  # ok / insufficient_data / error
    Column("recommendations_status", Text, nullable=False),
    Column("n_reference", Integer, nullable=False),
    Column("n_current", Integer, nullable=False),
    Column("n_current_recommendations", Integer, nullable=False),
    Column("details", JSON, nullable=False),  # per feature: method, threshold, score, drifted
)

# Stage 5 (spec 5.4 "NDCG@10 of the last runs"): evaluate's numbers in long format, so Grafana
# reads them from `app` instead of MLflow's internal schema (MLflow stays the source of truth).
run_metrics = Table(
    "run_metrics",
    metadata,
    Column("run_id", Text, nullable=False),
    Column("metric", Text, nullable=False),
    Column("value", Float, nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    PrimaryKeyConstraint("run_id", "metric", name="run_metrics_pkey"),
)

# Online quality (spec 5.2). The page refreshes recommendations after every rating, so a visitor
# sees one movie many times: only the FIRST impression of a (user, movie) pair counts, otherwise
# repeats would inflate the denominator. "Rated within an hour" means
# shown_at <= rated_at < shown_at + 1 hour. A later re-rating moves rated_at and can push the
# pair out of the window (conservative).
RECOMMENDATION_OUTCOMES_SQL = """
CREATE OR REPLACE VIEW recommendation_outcomes AS
WITH first_shown AS (
    SELECT DISTINCT ON (user_id, movie_id)
           user_id, movie_id, position, model_version, path, shown_at
    FROM recommendation_impressions
    ORDER BY user_id, movie_id, shown_at, impression_id
)
SELECT f.user_id, f.movie_id, f.position, f.model_version, f.path, f.shown_at,
       r.rating, r.rated_at,
       COALESCE(r.rated_at >= f.shown_at
                AND r.rated_at < f.shown_at + INTERVAL '1 hour', false) AS rated_within_hour
FROM first_shown f
LEFT JOIN ratings r ON r.user_id = f.user_id AND r.movie_id = f.movie_id
"""
ONLINE_QUALITY_DAILY_SQL = """
CREATE OR REPLACE VIEW online_quality_daily AS
SELECT date_trunc('day', shown_at) AS day, path, model_version,
       count(*) AS n_shown,
       count(*) FILTER (WHERE rated_within_hour) AS n_rated_within_hour,
       avg(rated_within_hour::int)::double precision AS share_rated_within_hour,
       (avg(rating) FILTER (WHERE rated_within_hour))::double precision AS avg_rating
FROM recommendation_outcomes
GROUP BY 1, 2, 3
"""
MONITORING_VIEWS = ("online_quality_daily", "recommendation_outcomes")  # in drop order

DASHBOARD_READER = "grafana_reader"
DASHBOARD_OBJECTS = (
    "recommendation_outcomes",
    "online_quality_daily",
    "drift_metrics",
    "run_metrics",
)
# Only if the role exists: infra/postgres/init creates it in compose; tests have no such role.
GRANT_DASHBOARD_SQL = f"""
DO $grant$
BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname = '{DASHBOARD_READER}') THEN
        GRANT SELECT ON {", ".join(DASHBOARD_OBJECTS)} TO {DASHBOARD_READER};
    END IF;
END
$grant$
"""

for _sql in (RECOMMENDATION_OUTCOMES_SQL, ONLINE_QUALITY_DAILY_SQL, GRANT_DASHBOARD_SQL):
    event.listen(metadata, "after_create", DDL(_sql).execute_if(dialect="postgresql"))
for _view in MONITORING_VIEWS:
    event.listen(
        metadata,
        "before_drop",
        DDL(f"DROP VIEW IF EXISTS {_view}").execute_if(dialect="postgresql"),
    )


def make_engine(url: str, **kwargs) -> Engine:
    """Engine with liveness checks and a short connect timeout (Postgres down -> fast error)."""
    kwargs.setdefault("pool_pre_ping", True)
    if url.startswith("postgresql"):
        kwargs.setdefault("connect_args", {"connect_timeout": 3})
    return create_engine(url, **kwargs)


def create_schema(engine: Engine) -> None:
    """Create missing tables, indexes and the web-id sequence; existing ones are left alone.
    On Postgres also (re)create the monitoring views and grant them to the dashboard role."""
    metadata.create_all(engine)
