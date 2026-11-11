import numpy as np
import pandas as pd
import pytest
from synthetic import make_ratings, taste_of_movie, taste_of_user
from threadpoolctl import threadpool_info

from recsys.models.als import (
    CANDIDATE_COLUMNS,
    ALSCandidateGenerator,
    ALSConfig,
    confidence_weights,
)

CONFIG = ALSConfig(factors=8, regularization=0.1, iterations=10, num_threads=1)


@pytest.fixture(scope="module")
def ratings() -> pd.DataFrame:
    return make_ratings()


@pytest.fixture(scope="module")
def als(ratings) -> ALSCandidateGenerator:
    return ALSCandidateGenerator(CONFIG).fit(ratings)


def _seen(ratings: pd.DataFrame, user: int) -> set[int]:
    return set(ratings.loc[ratings["user_id"] == user, "movie_id"])


def test_confidence_weights_linear_and_binary():
    r = np.array([1, 3, 5])
    assert confidence_weights(r, "linear").tolist() == [1.0, 3.0, 5.0]
    assert confidence_weights(r, "binary").tolist() == [1.0, 1.0, 1.0]
    assert confidence_weights(r).dtype == np.float32


def test_unknown_weight_scheme_raises():
    with pytest.raises(ValueError, match="rating_weight"):
        confidence_weights(np.array([1]), "cubic")


def test_candidates_schema_ranks_and_sorted_scores(als):
    cands = als.candidates([1, 2], n=20)
    assert list(cands.columns) == CANDIDATE_COLUMNS
    for _, group in cands.groupby("user_id"):
        assert group["als_rank"].tolist() == list(range(1, 21))
        assert group["als_score"].is_monotonic_decreasing
        assert group["movie_id"].is_unique


def test_candidates_exclude_training_items(als, ratings):
    cands = als.candidates(list(range(1, 61)), n=30)
    for user, group in cands.groupby("user_id"):
        assert not set(group["movie_id"]) & _seen(ratings, user)


def test_candidates_capped_by_unseen_catalog(als, ratings):
    cands = als.candidates([5], n=1000)
    n_catalog = ratings["movie_id"].nunique()
    assert len(cands) == n_catalog - len(_seen(ratings, 5))
    assert set(cands["movie_id"]) <= set(ratings["movie_id"])
    assert np.isfinite(cands["als_score"]).all()


def test_unknown_user_raises_key_error(als):
    with pytest.raises(KeyError):
        als.candidates([999])


def test_fold_in_excludes_rated_and_ignores_unknown_movies(als):
    new = pd.DataFrame({"user_id": [-1, -1, -1], "movie_id": [3, 6, 10_000], "rating": [5, 4, 5]})
    cands = als.candidates_for_ratings(new, n=20)
    assert len(cands) == 20 and set(cands["user_id"]) == {-1}
    assert not set(cands["movie_id"]) & {3, 6, 10_000}


def test_fold_in_user_with_only_unknown_movies_gets_no_candidates(als):
    new = pd.DataFrame({"user_id": [-1], "movie_id": [10_000], "rating": [5]})
    cands = als.candidates_for_ratings(new, n=20)
    assert cands.empty and list(cands.columns) == CANDIDATE_COLUMNS


def test_fold_in_close_to_fitted_factors_for_known_user(als, ratings):
    overlaps = []
    for user in range(1, 21):
        fitted = set(als.candidates([user], n=10)["movie_id"])
        folded = set(
            als.candidates_for_ratings(ratings[ratings["user_id"] == user], n=10)["movie_id"]
        )
        overlaps.append(len(fitted & folded) / 10)
    assert np.mean(overlaps) >= 0.7


def test_als_prefers_users_taste(als):
    # Each user already rated 20 of the 30 movies of their taste, so only 10 unseen ones are
    # left (several of them almost never rated); check the top-5 against the 1/3 chance level.
    cands = als.candidates(list(range(1, 61)), n=5)
    hit = [taste_of_movie(m) == taste_of_user(u) for u, m in cands[["user_id", "movie_id"]].values]
    assert np.mean(hit) > 0.75


def test_fit_is_deterministic(ratings):
    a = ALSCandidateGenerator(CONFIG).fit(ratings).candidates([1, 2, 3], n=15)
    b = ALSCandidateGenerator(CONFIG).fit(ratings).candidates([1, 2, 3], n=15)
    pd.testing.assert_frame_equal(a, b)


def test_save_load_roundtrip_same_candidates(als, tmp_path, ratings):
    als.save(tmp_path / "als")
    loaded = ALSCandidateGenerator.load(tmp_path / "als")
    assert loaded.config == als.config
    assert loaded.n_train_ratings == len(ratings)
    pd.testing.assert_frame_equal(loaded.candidates([1, 7]), als.candidates([1, 7]))
    new = pd.DataFrame({"user_id": [0, 0], "movie_id": [3, 6], "rating": [5, 4]})
    pd.testing.assert_frame_equal(
        loaded.candidates_for_ratings(new), als.candidates_for_ratings(new)
    )


def _with_stale_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Every row preceded by an older, different rating of the same (user, movie)."""
    stale = frame.copy()
    stale["rating"] = (6 - stale["rating"]).astype(frame["rating"].dtype)
    return pd.concat([stale, frame], ignore_index=True)


def test_fold_in_duplicate_rows_keep_last_instead_of_summing(als, ratings):
    own = ratings[ratings["user_id"] == 3][["user_id", "movie_id", "rating"]]
    clean = als.candidates_for_ratings(own, n=10)
    duplicated = als.candidates_for_ratings(_with_stale_duplicates(own), n=10)
    pd.testing.assert_frame_equal(duplicated, clean)


def test_fit_duplicate_rows_keep_last_instead_of_summing(ratings):
    cols = ["user_id", "movie_id", "rating"]
    clean = ALSCandidateGenerator(CONFIG).fit(ratings[cols])
    duplicated = ALSCandidateGenerator(CONFIG).fit(_with_stale_duplicates(ratings[cols]))
    assert duplicated._user_items is not None and clean._user_items is not None
    assert (duplicated._user_items != clean._user_items).nnz == 0


def test_pin_blas_threads_skips_per_call_limits(monkeypatch):
    """Per-call threadpool_limits changes a process-wide setting and races between request
    threads; once BLAS is pinned for the process, ALS calls must not touch it again."""
    import recsys.models.als as als_module

    ratings = make_ratings()
    model = ALSCandidateGenerator(ALSConfig(factors=4, iterations=3, num_threads=1)).fit(ratings)
    visitor = pd.DataFrame({"user_id": [0, 0], "movie_id": [3, 6], "rating": [5, 4]})
    calls = []
    real = als_module.threadpool_limits

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    before = {p["filepath"]: p["num_threads"] for p in threadpool_info()}
    monkeypatch.setattr(als_module, "_BLAS_PINNED", False)
    monkeypatch.setattr(als_module, "_BLAS_LIMITER", None)
    monkeypatch.setattr(als_module, "threadpool_limits", spy)
    model.candidates_for_ratings(visitor, n=5)
    assert calls, "without pinning every call limits BLAS itself"

    calls.clear()
    als_module.pin_blas_threads()
    try:
        model.candidates_for_ratings(visitor, n=5)
        model.candidates([1, 2], n=5)
        assert calls == []
        blas = [p for p in threadpool_info() if p["user_api"] == "blas"]
        assert all(p["num_threads"] == 1 for p in blas)
    finally:
        # The pin is process-wide: undo it so later tests see the thread pools they had.
        als_module._BLAS_LIMITER.restore_original_limits()
    assert {p["filepath"]: p["num_threads"] for p in threadpool_info()} == before
