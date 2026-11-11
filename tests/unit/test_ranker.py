import numpy as np
import pandas as pd
import pytest
from synthetic import make_movies, make_ratings, make_users

from recsys.features import (
    DEMOGRAPHIC_COLUMNS,
    FEATURE_COLUMNS,
    movie_features,
    movie_genre_matrix,
    user_features,
)
from recsys.metrics import ndcg_at_k
from recsys.models.ranker import (
    LambdaRanker,
    RankerConfig,
    build_training_frame,
    label_candidates,
)

SMALL = RankerConfig(n_estimators=40, min_child_samples=5, early_stopping_rounds=10, n_jobs=1)


def _planted_frame(n_users: int = 150, per_user: int = 20, seed: int = 0) -> pd.DataFrame:
    """Random features; the label is 1 exactly when movie_mean_rating is high."""
    rng = np.random.default_rng(seed)
    n = n_users * per_user
    frame = pd.DataFrame(
        rng.random((n, len(FEATURE_COLUMNS)), dtype=np.float32), columns=FEATURE_COLUMNS
    )
    frame["user_occupation"] = rng.integers(0, 21, n).astype(np.float32)
    frame.insert(0, "movie_id", np.tile(np.arange(per_user), n_users))
    frame.insert(0, "user_id", np.repeat(np.arange(n_users), per_user))
    frame["label"] = (frame["movie_mean_rating"] > 0.8).astype(np.int8)
    return frame


def _mean_ndcg(frame: pd.DataFrame, scores: np.ndarray) -> float:
    frame = frame.assign(score=scores)
    out = []
    for _, g in frame.groupby("user_id"):
        ranked = g.sort_values("score", ascending=False)["movie_id"].tolist()
        out.append(ndcg_at_k(ranked, set(g.loc[g["label"] == 1, "movie_id"]), k=10))
    return float(np.mean(out))


@pytest.fixture(scope="module")
def fitted() -> LambdaRanker:
    return LambdaRanker(SMALL).fit(_planted_frame())


def test_labels_mark_target_pairs_only():
    cands = pd.DataFrame({"user_id": [1, 1, 2, 2], "movie_id": [10, 11, 10, 12]})
    targets = pd.DataFrame({"user_id": [1, 2, 3], "movie_id": [11, 12, 10]})
    assert label_candidates(cands, targets).tolist() == [0, 1, 0, 1]
    assert label_candidates(cands, targets).dtype == np.int8


def _training_inputs():
    ratings = make_ratings(n_users=6)
    movies = make_movies()
    mf = movie_features(ratings, movies)
    uf = user_features(ratings, movie_genre_matrix(movies), make_users(6))
    # users 1..3 get 4 candidates each (given out of order), user 3 has no positive
    cands = pd.DataFrame(
        {
            "user_id": [2, 2, 1, 1, 3, 3, 1, 1, 2, 2, 3, 3],
            "movie_id": [5, 6, 1, 2, 7, 8, 3, 4, 7, 8, 9, 10],
            "als_score": np.linspace(1, 0, 12),
            "als_rank": [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4],
        }
    )
    targets = pd.DataFrame({"user_id": [1, 2, 3], "movie_id": [3, 6, 50]})
    return cands, targets, mf, uf


def test_training_frame_drops_groups_without_positives():
    frame = build_training_frame(*_training_inputs())
    assert set(frame["user_id"]) == {1, 2}
    assert frame["label"].sum() == 2 and len(frame) == 8
    assert list(frame.columns) == ["user_id", "movie_id", *FEATURE_COLUMNS, "label"]


def test_training_frame_groups_are_contiguous():
    frame = build_training_frame(*_training_inputs())
    assert frame["user_id"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert frame.loc[frame["user_id"] == 1, "als_rank"].tolist() == [1, 2, 3, 4]


def test_ranker_learns_planted_signal(fitted):
    importance = fitted.feature_importance()
    assert max(importance, key=importance.get) == "movie_mean_rating"
    holdout = _planted_frame(n_users=50, seed=1)
    random_scores = np.random.default_rng(2).random(len(holdout))
    assert _mean_ndcg(holdout, fitted.predict(holdout)) > 0.95
    assert _mean_ndcg(holdout, random_scores) < 0.7


def test_ranker_is_deterministic(fitted):
    again = LambdaRanker(SMALL).fit(_planted_frame())
    holdout = _planted_frame(n_users=20, seed=3)
    np.testing.assert_array_equal(again.predict(holdout), fitted.predict(holdout))
    assert again.best_iteration == fitted.best_iteration


def test_ranker_save_load_same_predictions(fitted, tmp_path):
    fitted.save(tmp_path / "ranker")
    loaded = LambdaRanker.load(tmp_path / "ranker")
    holdout = _planted_frame(n_users=20, seed=4)
    np.testing.assert_allclose(loaded.predict(holdout), fitted.predict(holdout))
    assert loaded.best_iteration == fitted.best_iteration and loaded.config == fitted.config


def test_ranker_handles_nan_demographics(fitted):
    holdout = _planted_frame(n_users=10, seed=5)
    holdout[DEMOGRAPHIC_COLUMNS] = np.nan
    assert np.isfinite(fitted.predict(holdout)).all()


def test_predict_ignores_extra_columns_and_uses_feature_order(fitted):
    holdout = _planted_frame(n_users=5, seed=6)
    shuffled = holdout[["label", *reversed(FEATURE_COLUMNS), "movie_id", "user_id"]]
    np.testing.assert_array_equal(fitted.predict(shuffled), fitted.predict(holdout))
