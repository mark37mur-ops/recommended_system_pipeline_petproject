import numpy as np
import pandas as pd
import pytest
from synthetic import make_movies, make_ratings, make_users

from recsys.data.split import temporal_split
from recsys.evaluation import build_history
from recsys.features import (
    build_feature_frame,
    drop_demographics,
    movie_features,
    movie_genre_matrix,
    user_features,
)
from recsys.models.als import ALSCandidateGenerator, ALSConfig
from recsys.models.ranker import LambdaRanker, RankerConfig, build_training_frame
from recsys.models.two_stage import RECOMMENDATION_COLUMNS, TwoStageRecommender

ALS = ALSConfig(factors=8, regularization=0.1, iterations=10, num_threads=1)
RANKER = RankerConfig(n_estimators=30, min_child_samples=5, early_stopping_rounds=10, n_jobs=1)


@pytest.fixture(scope="module")
def data():
    ratings, movies, users = make_ratings(), make_movies(), make_users()
    split = temporal_split(ratings)
    history, seen = build_history(split)
    return ratings, movies, users, split, history, seen


@pytest.fixture(scope="module")
def recommender(data) -> TwoStageRecommender:
    _, movies, users, split, history, _ = data
    genres = movie_genre_matrix(movies)
    # ranker trained on the stage-A protocol (ALS on als_train, labels from ranker_train)
    als_a = ALSCandidateGenerator(ALS).fit(split.als_train)
    cands = als_a.candidates(sorted(split.ranker_train["user_id"].unique()), n=30)
    uf_a = drop_demographics(user_features(split.als_train, genres, users), 0.1, seed=0)
    frame = build_training_frame(
        cands, split.ranker_train, movie_features(split.als_train, movies), uf_a
    )
    ranker = LambdaRanker(RANKER).fit(frame)
    als_b = ALSCandidateGenerator(ALS).fit(history)
    return TwoStageRecommender(
        als_b,
        movie_features(history, movies),
        user_features(history, genres, users),
        ranker=ranker,
        n_candidates=30,
    )


def test_variant_names(recommender):
    assert recommender.variant == "als_ranker"
    assert recommender.without_ranker().variant == "als_only"


def test_known_users_get_k_unseen_items(recommender, data):
    seen = data[5]
    recs = recommender.recommend_known_users(list(range(1, 61)), k=10)
    assert set(recs) == set(range(1, 61))
    for user, items in recs.items():
        assert len(items) == 10 and len(set(items)) == 10
        assert not set(items) & seen[user]


def test_als_only_equals_top_k_als_candidates(recommender):
    als_only = recommender.without_ranker().recommend_known_users([1, 2, 3], k=10)
    cands = recommender.als.candidates([1, 2, 3], n=30)
    for user in (1, 2, 3):
        top = cands[cands["user_id"] == user].nsmallest(10, "als_rank")["movie_id"].tolist()
        assert als_only[user] == top


def test_ranker_changes_order_within_candidates(recommender):
    users = list(range(1, 61))
    ranked = recommender.recommend_known_users(users, k=10)
    als_only = recommender.without_ranker().recommend_known_users(users, k=10)
    cands = recommender.als.candidates(users, n=30)
    for user in users:
        assert set(ranked[user]) <= set(cands.loc[cands["user_id"] == user, "movie_id"])
    assert any(ranked[u] != als_only[u] for u in users)


def test_new_user_fold_in_excludes_rated(recommender):
    rated = [(3, 5), (6, 5), (9, 4), (12, 5), (15, 4)]
    recs = recommender.recommend_new_user(rated, k=10, profile={"gender": "F", "age": 25})
    assert len(recs) == 10 and len(set(recs)) == 10
    assert not set(recs) & {m for m, _ in rated}


def test_new_user_without_profile_works(recommender):
    recs = recommender.recommend_new_user([(3, 5), (6, 4), (9, 5)], k=5)
    assert len(recs) == 5


def test_new_user_with_only_unknown_movies_gets_empty_list(recommender):
    assert recommender.recommend_new_user([(10_000, 5)], k=5) == []


def test_recommend_for_ratings_output_schema(recommender, data):
    history = data[4]
    ratings = history[history["user_id"].isin([4, 5])]
    out = recommender.recommend_for_ratings(ratings, k=7)
    assert list(out.columns) == RECOMMENDATION_COLUMNS
    assert set(out["user_id"]) == {4, 5}
    for _, group in out.groupby("user_id"):
        assert group["rank"].tolist() == list(range(1, 8))
        assert group["score"].is_monotonic_decreasing


def test_serving_user_features_reproduce_batch_features(recommender, data):
    _, _, users, _, history, _ = data
    user = 7
    own = history[history["user_id"] == user]
    serving = recommender.user_features_for(own, users[users["user_id"] == user])
    pd.testing.assert_frame_equal(serving, recommender.user_feats.loc[[user]])
    cands = recommender.als.candidates([user], n=30)
    pd.testing.assert_frame_equal(
        build_feature_frame(cands, recommender.movie_feats, serving),
        build_feature_frame(cands, recommender.movie_feats, recommender.user_feats),
    )


@pytest.mark.parametrize("with_ranker", [True, False])
def test_save_load_roundtrip_same_recommendations(recommender, tmp_path, with_ranker):
    model = recommender if with_ranker else recommender.without_ranker()
    model.save(tmp_path / "bundle")
    loaded = TwoStageRecommender.load(tmp_path / "bundle")
    assert loaded.variant == model.variant and loaded.n_candidates == model.n_candidates
    users = [1, 2, 30]
    assert loaded.recommend_known_users(users) == model.recommend_known_users(users)
    rated = [(3, 5), (6, 4)]
    assert loaded.recommend_new_user(rated) == model.recommend_new_user(rated)
    pd.testing.assert_frame_equal(loaded.movie_feats, model.movie_feats)
    np.testing.assert_array_equal(loaded.user_feats.index, model.user_feats.index)


def test_fold_in_matches_fitted_path_for_training_users(recommender, data):
    """Serving a known user through fold-in on their own history must give (nearly) the
    recommendations of the fitted-factor path used in offline evaluation."""
    _, _, users, _, history, _ = data
    user_ids = list(range(1, 61))
    fitted = recommender.recommend_known_users(user_ids, k=10)
    own = history[history["user_id"].isin(user_ids)]
    folded = TwoStageRecommender._as_dict(
        recommender.recommend_for_ratings(own, k=10, profiles=users), user_ids
    )
    overlaps = [len(set(fitted[u]) & set(folded[u])) / 10 for u in user_ids]
    assert float(np.mean(overlaps)) >= 0.7, overlaps
    for u in user_ids:
        assert not set(folded[u]) & set(history.loc[history["user_id"] == u, "movie_id"])
