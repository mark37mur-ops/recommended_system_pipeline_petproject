from pathlib import Path

import pytest

from recsys.data.movielens import load_ratings
from recsys.data.split import temporal_split
from recsys.pipelines.baseline import build_history, recommend_for_test_users, run_baseline

N_USERS = 30
N_RATINGS = 20
N_MOVIES = 40


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    ratings_lines = []
    for user in range(1, N_USERS + 1):
        # movie 1 is rated by everyone; the rest is a user-dependent window of movies
        movies = [1] + [2 + (user + i) % (N_MOVIES - 1) for i in range(N_RATINGS - 1)]
        for i, movie in enumerate(movies):
            ratings_lines.append(f"{user}::{movie}::{1 + (user + i) % 5}::{1_000_000 + i * 100}")
    (tmp_path / "ratings.dat").write_text("\n".join(ratings_lines) + "\n", encoding="latin-1")
    movies_lines = [f"{m}::Movie {m} (1999)::Drama|Comedy" for m in range(1, N_MOVIES + 1)]
    (tmp_path / "movies.dat").write_text("\n".join(movies_lines) + "\n", encoding="latin-1")
    return tmp_path


def test_run_baseline_returns_metrics_in_unit_range(raw_dir):
    result = run_baseline(raw_dir)
    assert set(result) == {"ndcg_at_10", "recall_at_10", "coverage"}
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_run_baseline_matches_golden_values(raw_dir):
    # Golden values for this synthetic fixture, computed after the F1+F2 fixes and cross-checked
    # with an independent pure-python implementation of the same protocol.
    result = run_baseline(raw_dir)
    assert result["ndcg_at_10"] == pytest.approx(0.5292149112368794)
    assert result["recall_at_10"] == pytest.approx(0.5666666666666667)
    assert result["coverage"] == pytest.approx(0.6)


def test_recommendations_exclude_every_seen_movie(raw_dir):
    split = temporal_split(load_ratings(raw_dir))
    _, seen = build_history(split)
    recommendations = recommend_for_test_users(split, k=10)
    assert recommendations
    for user, recs in recommendations.items():
        assert len(recs) == 10
        assert not set(recs) & seen[user]


def test_history_contains_no_test_ratings(raw_dir):
    split = temporal_split(load_ratings(raw_dir))
    history, seen = build_history(split)
    keys = lambda df: set(zip(df["user_id"], df["movie_id"], strict=True))  # noqa: E731
    assert not keys(history) & keys(split.test)
    assert len(history) == len(split.als_train) + len(split.ranker_train)
    for user, movies in seen.items():
        test_movies = set(split.test.loc[split.test["user_id"] == user, "movie_id"])
        assert not movies & test_movies


def test_run_baseline_is_deterministic(raw_dir):
    assert run_baseline(raw_dir) == run_baseline(raw_dir)
