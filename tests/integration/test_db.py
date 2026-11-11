import os

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from synthetic import N_MOVIES, N_USERS, PER_USER, write_raw_dir

from recsys.db.load_movielens import load_movielens, main
from recsys.db.schema import WEB_USER_ID_START, create_schema, ratings, users


def test_create_schema_is_idempotent(db):
    create_schema(db)
    create_schema(db)
    with db.connect() as conn:
        tables = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        ).scalars()
        assert {"movies", "users", "ratings"} <= set(tables)


def test_web_user_sequence_starts_in_web_range(db):
    with db.begin() as conn:
        first = conn.execute(text("SELECT nextval('web_user_id_seq')")).scalar_one()
        second = conn.execute(text("SELECT nextval('web_user_id_seq')")).scalar_one()
    assert first == WEB_USER_ID_START and second == WEB_USER_ID_START + 1


@pytest.mark.parametrize(
    ("user_id", "source"), [(42, "web"), (WEB_USER_ID_START, "movielens"), (1, "other")]
)
def test_check_constraints_tie_source_to_id_range(db, user_id, source):
    with pytest.raises(IntegrityError), db.begin() as conn:
        conn.execute(users.insert().values(user_id=user_id, source=source))


def test_loader_is_idempotent_and_marks_movielens(db, tmp_path):
    raw = write_raw_dir(tmp_path / "raw")
    first = load_movielens(db, raw)
    second = load_movielens(db, raw)
    assert first == second
    assert (first.movies, first.users, first.ratings) == (N_MOVIES, N_USERS, N_USERS * PER_USER)
    with db.connect() as conn:
        sources = conn.execute(select(ratings.c.source, func.count()).group_by(ratings.c.source))
        assert dict(sources.all()) == {"movielens": N_USERS * PER_USER}
        row = conn.execute(text("SELECT title, year, genres FROM movies WHERE movie_id = 5")).one()
        assert row.title == "Movie 5" and row.year == 1985 and row.genres == "Horror|Romance"
        assert conn.execute(select(func.max(users.c.user_id))).scalar_one() < WEB_USER_ID_START


def test_loader_keeps_web_ratings(db, tmp_path):
    raw = write_raw_dir(tmp_path / "raw")
    load_movielens(db, raw)
    with db.begin() as conn:
        conn.execute(users.insert().values(user_id=WEB_USER_ID_START, source="web"))
        conn.execute(
            ratings.insert().values(
                user_id=WEB_USER_ID_START, movie_id=1, rating=5, source="web", rated_at=func.now()
            )
        )
    load_movielens(db, raw)
    with db.connect() as conn:
        n_web = conn.execute(
            select(func.count()).select_from(ratings).where(ratings.c.source == "web")
        ).scalar_one()
    assert n_web == 1


def test_cli_reads_database_url_from_env(db, tmp_path, monkeypatch, capsys):
    raw = write_raw_dir(tmp_path / "raw")
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    stats = main(["--data-dir", str(raw)])
    assert stats.ratings == N_USERS * PER_USER
    assert f"{stats.ratings} ratings" in capsys.readouterr().out
