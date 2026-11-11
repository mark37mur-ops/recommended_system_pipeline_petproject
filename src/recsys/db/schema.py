"""Schema of the `app` database (spec 1, 3.4): movies, users, ratings; retrain_runs (stage 4).

SQLAlchemy Core, created idempotently with `create_schema` (`metadata.create_all`): the schema
is small and only gains new tables before v2, which `create_all` adds; Alembic becomes worth it
at the first ALTER of an existing table.

Web visitors get ids from their own sequence starting at WEB_USER_ID_START, far above the
MovieLens ids (1..6040), and a CHECK constraint ties `source` to the id range, so a web user can
never collide with (or be mistaken for) a MovieLens user.
"""

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Engine,
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


def make_engine(url: str, **kwargs) -> Engine:
    """Engine with liveness checks and a short connect timeout (Postgres down -> fast error)."""
    kwargs.setdefault("pool_pre_ping", True)
    if url.startswith("postgresql"):
        kwargs.setdefault("connect_args", {"connect_timeout": 3})
    return create_engine(url, **kwargs)


def create_schema(engine: Engine) -> None:
    """Create missing tables, indexes and the web-id sequence; existing ones are left alone."""
    metadata.create_all(engine)
