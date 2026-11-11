import pytest

from recsys.data.movielens import load_movies, load_ratings, load_users

MOVIES = (
    "1::Toy Story (1995)::Animation|Children's|Comedy\n"
    "2::City of Lost Children, The (Cité des enfants perdus, La) (1995)::Adventure|Sci-Fi\n"
    "3::Misérables, Les (1995)::Drama\n"
)
RATINGS = "1::1::5::978300760\n1::2::3::978302109\n2::3::4::978301968\n"
USERS = "1::F::1::10::48067\n2::M::56::16::70072\n"


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "movies.dat").write_bytes(MOVIES.encode("latin-1"))
    (tmp_path / "ratings.dat").write_bytes(RATINGS.encode("latin-1"))
    (tmp_path / "users.dat").write_bytes(USERS.encode("latin-1"))
    return tmp_path


def test_load_movies_parses_year_title_genres(raw_dir):
    m = load_movies(raw_dir).set_index("movie_id")
    assert m.loc[1, "title"] == "Toy Story"
    assert m.loc[1, "year"] == 1995
    assert m.loc[1, "genres"] == ["Animation", "Children's", "Comedy"]
    assert m.loc[2, "title"] == "City of Lost Children, The (Cité des enfants perdus, La)"
    assert m.loc[3, "title"] == "Misérables, Les"


def test_load_movies_missing_year_is_na(tmp_path):
    (tmp_path / "movies.dat").write_bytes(b"1::No Year Movie::Drama\n")
    m = load_movies(tmp_path)
    assert m.loc[0, "title"] == "No Year Movie"
    assert m["year"].isna().all() and str(m["year"].dtype) == "Int16"


def test_load_ratings_columns_and_dtypes(raw_dir):
    r = load_ratings(raw_dir)
    assert list(r.columns) == ["user_id", "movie_id", "rating", "timestamp"]
    assert len(r) == 3 and r["rating"].dtype == "int8"
    assert r["user_id"].dtype == "int32" and r["timestamp"].dtype == "int64"


def test_load_users(raw_dir):
    u = load_users(raw_dir).set_index("user_id")
    assert u.loc[2, "age"] == 56 and u.loc[1, "zip"] == "48067"
