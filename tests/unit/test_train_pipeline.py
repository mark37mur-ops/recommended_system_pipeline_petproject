import pandas as pd
import pytest
from synthetic import make_movies, make_ratings, make_users

from recsys.data.split import temporal_split
from recsys.evaluation import build_history
from recsys.models import als as als_module
from recsys.models.als import ALSConfig
from recsys.models.ranker import RankerConfig
from recsys.pipelines.train import (
    MODEL_NAMES,
    TrainConfig,
    select_variant,
    train_and_evaluate,
    validate_als,
)

CONFIG = TrainConfig(
    als=ALSConfig(factors=8, regularization=0.1, iterations=10, num_threads=1),
    ranker=RankerConfig(n_estimators=30, min_child_samples=5, early_stopping_rounds=10, n_jobs=1),
    n_candidates=30,
)


@pytest.fixture(scope="module")
def inputs():
    return make_ratings(), make_movies(), make_users()


@pytest.fixture(scope="module")
def result(inputs):
    return train_and_evaluate(*inputs, config=CONFIG)


def _pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(df["user_id"], df["movie_id"], strict=True))


def test_train_and_evaluate_reports_three_models(result):
    assert tuple(result.metrics) == MODEL_NAMES
    for metrics in result.metrics.values():
        assert set(metrics) == {"ndcg_at_10", "recall_at_10", "coverage"}
        assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert result.variant in {"als_only", "als_ranker"}
    assert result.recommender.variant == result.variant


def test_als_beats_popularity_on_structured_synthetic(result):
    assert result.metrics["als_only"]["ndcg_at_10"] > result.metrics["popularity"]["ndcg_at_10"]


def test_ranker_trained_on_als_train_only_and_eval_on_history(inputs, monkeypatch):
    ratings = inputs[0]
    split = temporal_split(ratings)
    fitted_on = []
    original_fit = als_module.ALSCandidateGenerator.fit

    def spy(self, frame):
        fitted_on.append(frame)
        return original_fit(self, frame)

    monkeypatch.setattr(als_module.ALSCandidateGenerator, "fit", spy)
    result = train_and_evaluate(*inputs, config=CONFIG)

    assert len(fitted_on) == 2
    stage_a, stage_b = (_pairs(f) for f in fitted_on)
    assert stage_a == _pairs(split.als_train)  # ALS behind the ranker never sees its labels
    assert stage_b == _pairs(split.als_train) | _pairs(split.ranker_train)
    assert not stage_b & _pairs(split.test)
    assert result.diagnostics["ranker_als_n_ratings"] == len(split.als_train)
    assert result.diagnostics["eval_als_n_ratings"] == len(split.als_train) + len(
        split.ranker_train
    )


def test_registered_recommender_never_recommends_history(inputs, result):
    split = temporal_split(inputs[0])
    _, seen = build_history(split)
    users = sorted(split.test["user_id"].unique())
    recs = result.recommender.recommend_known_users(users, k=10)
    for user in users:
        assert len(recs[user]) == 10 and not set(recs[user]) & seen[user]


def test_pipeline_is_deterministic(inputs, result):
    again = train_and_evaluate(*inputs, config=CONFIG)
    assert again.metrics == result.metrics
    assert again.variant == result.variant


def test_diagnostics_describe_ranker_dataset(result):
    d = result.diagnostics
    assert d["ranker_groups"] > 0 and d["ranker_rows"] >= d["ranker_groups"]
    assert 0 < d["ranker_positive_rate"] <= 1
    assert 0 < d["ranker_candidate_recall"] <= 1
    assert d["ranker_best_iteration"] >= 1
    assert set(result.feature_importance) >= {"als_score", "genre_affinity_max"}


@pytest.mark.parametrize(
    ("ranker_ndcg", "expected"), [(0.21, "als_ranker"), (0.20, "als_only"), (0.19, "als_only")]
)
def test_select_variant_prefers_als_only_unless_ranker_strictly_better(ranker_ndcg, expected):
    metrics = {
        "popularity": {"ndcg_at_10": 0.1},
        "als_only": {"ndcg_at_10": 0.20},
        "als_ranker": {"ndcg_at_10": ranker_ndcg},
    }
    assert select_variant(metrics) == expected


def test_validate_als_ignores_test_part(inputs):
    ratings, movies, _ = inputs
    split = temporal_split(ratings)
    clean = validate_als(split, n_items=len(movies), als_config=CONFIG.als, n_candidates=30)
    corrupted_test = split.test.assign(movie_id=1)
    corrupted = type(split)(split.als_train, split.ranker_train, corrupted_test)
    assert validate_als(corrupted, len(movies), CONFIG.als, n_candidates=30) == clean
    assert set(clean) == {"ndcg_at_10", "recall_at_10", "coverage", "candidate_recall_at_30"}
