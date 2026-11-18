from dataclasses import FrozenInstanceError

import pytest

from recsys.movie_metadata import MovieMetadata, match_movie, normalize_title, poster_url
from recsys_api.metadata import MetadataHolder


def candidate(tmdb_id=1, title="The Thing", year="1982", **kwargs):
    return dict(id=tmdb_id, title=title, release_date=f"{year}-06-25", **kwargs)


def test_match_rejects_remake_year():
    assert match_movie("Thing, The", 1982, [candidate(year="2011")]) is None
    right = candidate()
    assert match_movie("Thing, The", 1982, [candidate(year="2011"), right]) == right


def test_ambiguous_or_missing_year_unmatched():
    assert match_movie("The Thing", None, [candidate()]) is None
    assert match_movie("The Thing", 1982, [candidate(), candidate(2)]) is None
    assert match_movie("The Thing", 1982, [dict(id=1, title="The Thing")]) is None
    assert match_movie("The Thing", 1982, [candidate(title="Other")]) is None


def test_article_and_alternative_title_match():
    item = candidate(title="Нечто", original_title="The Thing")
    assert match_movie("Thing, The (1982)", 1982, [item]) == item
    item = candidate(title="La vita è bella", year="1997")
    assert match_movie("Life Is Beautiful (La vita è bella)", 1997, [item]) == item
    item = candidate(title="Other", alternative_titles={"titles": [{"title": "The Thing"}]})
    assert match_movie("Thing, The", 1982, [item]) == item
    assert normalize_title("  Matrix, The! ") == normalize_title("THE MATRIX")


def test_numeric_title_is_not_confused_with_release_year():
    item = candidate(title="1984", year="1984")
    assert match_movie("1984 (1984)", 1984, [item]) == item
    assert match_movie("Another Film (1984)", 1984, [item]) is None


@pytest.mark.parametrize(
    "title,year,aliases",
    [
        ("Hard 8 (a.k.a. Sydney, a.k.a. Hard Eight) (1996)", 1996, ["Sydney", "Hard Eight"]),
        (
            "Strike! (a.k.a. All I Wanna Do, The Hairy Bird) (1998)",
            1998,
            ["All I Wanna Do", "The Hairy Bird"],
        ),
        (
            "Other (a.k.a. Thing, The, a.k.a. Story, A, Example, An) (1982)",
            1982,
            ["The Thing", "A Story", "An Example"],
        ),
    ],
)
def test_multiple_aka_aliases_remain_exact_and_year_strict(title, year, aliases):
    for alias in aliases:
        item = candidate(title="Unrelated", original_title=alias, year=str(year))
        assert (
            match_movie(title, year, [candidate(99, title="Unrelated", year=str(year)), item])
            == item
        )
        assert match_movie(title, year + 1, [item]) is None
    assert (
        match_movie(
            title,
            year,
            [candidate(i, title=alias, year=str(year)) for i, alias in enumerate(aliases, start=1)],
        )
        is None
    )


def test_plain_comma_title_is_not_split_into_unverified_aliases():
    assert (
        match_movie("Hello, Goodbye (1998)", 1998, [candidate(title="Hello", year="1998")]) is None
    )


@pytest.mark.parametrize(
    "path",
    [
        None,
        "",
        "//evil/a.jpg",
        "https://evil/a.jpg",
        "/../a.jpg",
        "/a.jpg?x=1",
        "/a%2ejpg",
        "/a/b.jpg",
        "/a.svg",
        "/a.jpg\n",
    ],
)
def test_unsafe_poster_path_rejected(path):
    assert poster_url(path) is None


def test_poster_url_and_frozen_metadata():
    assert poster_url("/a_B-12.jpg") == "https://image.tmdb.org/t/p/w500/a_B-12.jpg"
    with pytest.raises(FrozenInstanceError):
        MovieMetadata(1, 2, None, None).tmdb_id = 3


def test_holder_atomic_refresh_and_local_search(monkeypatch):
    import recsys_api.metadata as module

    holder = MetadataHolder()
    assert holder.current == {} and holder.get(1) is None
    snapshot = {1: MovieMetadata(1, 2, None, "Матрица"), 2: MovieMetadata(2, 3, None, None)}
    monkeypatch.setattr(module, "load_metadata", lambda engine: snapshot)
    assert holder.refresh(None) == 2
    assert holder.get(1) == snapshot[1]
    assert holder.search("МАТР") == [1]
    assert holder.search("") == []
    assert holder.search("матр", limit=0) == []

    def fail(engine):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(module, "load_metadata", fail)
    with pytest.raises(RuntimeError):
        holder.refresh(None)
    assert holder.current == snapshot
