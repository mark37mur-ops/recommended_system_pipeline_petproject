import pytest
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError

from recsys.db.schema import movies
from recsys.movie_metadata import (
    MovieMetadata,
    load_enrichment_catalog,
    load_metadata,
    upsert_metadata,
)
from recsys_api.metadata import MetadataHolder


def test_empty_metadata(db):
    assert load_metadata(db) == {}
    assert MetadataHolder().refresh(db) == 0


def test_source_title_unchanged_and_repeated_upsert(db):
    with db.begin() as conn:
        conn.execute(
            insert(movies).values(movie_id=1, title="Thing, The", year=1982, genres="Horror")
        )
    upsert_metadata(db, MovieMetadata(1, 1091, "/first.jpg", "Нечто"))
    upsert_metadata(db, MovieMetadata(1, 1091, "/second.jpg", "Нечто"))
    assert load_metadata(db) == {1: MovieMetadata(1, 1091, "/second.jpg", "Нечто")}
    with db.connect() as conn:
        assert conn.scalar(select(movies.c.title)) == "Thing, The"
    catalog = load_enrichment_catalog(db)
    assert [(m.movie_id, m.title, m.year, m.n_ratings) for m in catalog] == [
        (1, "Thing, The", 1982, 0)
    ]


def test_metadata_fk(db):
    with pytest.raises(IntegrityError):
        upsert_metadata(db, MovieMetadata(999, 1, None, None))


def test_refresh_error_preserves_snapshot(db, monkeypatch):
    import recsys_api.metadata as module

    with db.begin() as conn:
        conn.execute(insert(movies).values(movie_id=1, title="Thing, The", year=1982, genres=""))
    upsert_metadata(db, MovieMetadata(1, 1091, None, "Нечто"))
    holder = MetadataHolder()
    assert holder.refresh(db) == 1
    previous = holder.current

    def fail(_):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(module, "load_metadata", fail)
    with pytest.raises(RuntimeError):
        holder.refresh(db)
    assert holder.current is previous
