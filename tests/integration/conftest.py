"""Integration tests: real Postgres and Redis, pointed to by TEST_DATABASE_URL / TEST_REDIS_URL.

Without both variables every test here is skipped. The tests drop and recreate the schema, so
they refuse to run against a database whose name does not contain "test".
"""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from recsys.db.schema import create_schema, make_engine, metadata

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
REDIS_URL = os.environ.get("TEST_REDIS_URL")


def pytest_collection_modifyitems(config, items):
    here = os.path.dirname(__file__)
    skip = pytest.mark.skip(reason="TEST_DATABASE_URL and TEST_REDIS_URL are not set")
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.integration)
            if not (DATABASE_URL and REDIS_URL):
                item.add_marker(skip)


@pytest.fixture(scope="session")
def engine():
    url = make_url(DATABASE_URL)
    if "test" not in (url.database or ""):
        pytest.exit(f"refusing to drop tables in database {url.database!r} (name lacks 'test')")
    engine = make_engine(DATABASE_URL)
    metadata.drop_all(engine)
    create_schema(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db(engine):
    """Empty tables (and a fresh web-id sequence) for every test."""
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE ratings, users, movies"))
        conn.execute(text("ALTER SEQUENCE web_user_id_seq RESTART"))
    return engine


@pytest.fixture
def redis_client():
    import redis

    client = redis.Redis.from_url(REDIS_URL)
    client.flushdb()
    yield client
    client.flushdb()
    client.close()
