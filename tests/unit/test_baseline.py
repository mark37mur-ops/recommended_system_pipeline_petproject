from pathlib import Path

import pytest

from recsys.pipelines.baseline import run_baseline

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


def test_run_baseline_is_deterministic(raw_dir):
    assert run_baseline(raw_dir) == run_baseline(raw_dir)
