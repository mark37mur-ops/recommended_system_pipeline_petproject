import pytest
from api_fakes import make_catalog_movies

from recsys_api.catalog import Catalog, CatalogHolder, Movie, display_title


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(make_catalog_movies())


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("Matrix, The", "The Matrix"),
        ("Christmas Story, A", "A Christmas Story"),
        (
            "City of Lost Children, The (Cité des enfants perdus, La)",
            "The City of Lost Children (Cité des enfants perdus, La)",
        ),
        ("Toy Story", "Toy Story"),
        ("Theory, Practice", "Theory, Practice"),
    ],
)
def test_display_title_moves_trailing_article(raw, shown):
    assert display_title(raw) == shown


def test_popularity_order_and_exclusion(catalog):
    assert catalog.popular(3) == [1, 2, 3]
    assert catalog.popular(3, exclude={1, 3}) == [2, 4, 5]
    assert len(catalog) == 60 and 5 in catalog and 61 not in catalog
    assert catalog.get(5).title == "Movie 5" and catalog.get(999) is None


def test_search_is_case_insensitive_prefix_first_then_popular():
    movies = [
        Movie(1, "Matrix, The", 1999, ("Action",), 500),
        Movie(2, "Animatrix", 2003, ("Animation",), 900),
        Movie(3, "Matrix Reloaded, The", 2003, ("Action",), 100),
        Movie(4, "Toy Story", 1995, ("Comedy",), 1000),
    ]
    catalog = Catalog(movies)
    assert [m.movie_id for m in catalog.search("MATRIX")] == [1, 3, 2]
    assert [m.movie_id for m in catalog.search("the matrix")] == [1, 3]
    assert [m.movie_id for m in catalog.search("  toy   story ")] == [4]
    assert catalog.search("zzz") == [] and catalog.search("   ") == []
    assert len(catalog.search("matrix", limit=1)) == 1


def test_onboarding_spreads_over_genres_and_is_deterministic(catalog):
    picks = catalog.onboarding(n=12)
    assert len(picks) == 12 and len({m.movie_id for m in picks}) == 12
    first_round = picks[:6]
    covered = {g for m in first_round for g in m.genres}
    assert set(make_genres()) <= covered  # one movie per genre before any genre repeats
    assert [m.movie_id for m in picks] == [m.movie_id for m in catalog.onboarding(n=12)]


def make_genres():
    return ("Comedy", "Drama", "Action", "Horror", "Romance", "Sci-Fi")


def test_onboarding_on_small_catalog_returns_everything():
    catalog = Catalog(make_catalog_movies(5))
    assert sorted(m.movie_id for m in catalog.onboarding(n=20)) == [1, 2, 3, 4, 5]
    assert Catalog([]).onboarding() == []


def test_genre_popular_prefers_genres_rated_high(catalog):
    # movie 3 is Horror (3 % 6 = 3), movie 9 is Horror too; both rated 5 -> Horror weight 2
    ratings = [(3, 5), (9, 5), (2, 1)]
    recs = catalog.genre_popular(ratings, k=5)
    assert len(recs) == 5
    assert all("Horror" in catalog.get(m).genres for m in recs)
    assert not {3, 9, 2} & set(recs)
    assert recs == sorted(recs)  # same weight -> popularity order


def test_genre_popular_without_high_ratings_is_plain_popularity(catalog):
    assert catalog.genre_popular([(1, 2), (2, 3)], k=3) == [3, 4, 5]


def test_genre_popular_pads_with_popular_when_genre_is_small():
    movies = make_catalog_movies(6)  # one movie per genre
    catalog = Catalog(movies)
    recs = catalog.genre_popular([(3, 5)], k=4)  # only Horror movie is the rated one
    assert len(recs) == 4 and 3 not in recs


def test_catalog_holder_swaps_whole_catalog():
    holder = CatalogHolder()
    assert len(holder.current) == 0

    class Repo:
        def load_movies(self):
            return make_catalog_movies(7)

    assert holder.refresh(Repo()) == 7 and len(holder.current) == 7
