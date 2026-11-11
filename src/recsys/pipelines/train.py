"""Training pipeline: popularity -> ALS-only -> ALS + ranker on one leakage-free protocol.

Stage A (ranker training data): ALS is fit on `als_train` only; its candidates for the
ranker-train users, with features computed from `als_train`, are labelled by `ranker_train`.
Stage B (evaluation): ALS is refit on `als_train ∪ ranker_train` (the history), features are
recomputed from the history, and the three models are evaluated on `test` through the shared
`EvalContext`. The ranker never sees `test`, and the ALS whose score it learns from never saw
its labels.
"""

from dataclasses import dataclass, field

import pandas as pd

from recsys.data.split import Split, temporal_split
from recsys.data.validate import validate_ratings
from recsys.evaluation import (
    EvalContext,
    compare_models,
    evaluate_recommendations,
    make_test_context,
    make_validation_context,
)
from recsys.features import drop_demographics, movie_features, movie_genre_matrix, user_features
from recsys.metrics import evaluate
from recsys.models.als import ALSCandidateGenerator, ALSConfig
from recsys.models.ranker import LambdaRanker, RankerConfig, build_training_frame
from recsys.models.two_stage import TwoStageRecommender
from recsys.pipelines.baseline import popularity_model

MODEL_NAMES = ("popularity", "als_only", "als_ranker")


@dataclass(frozen=True)
class TrainConfig:
    als: ALSConfig = field(default_factory=ALSConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    n_candidates: int = 100
    k: int = 10


@dataclass
class TrainResult:
    metrics: dict[str, dict[str, float]]
    variant: str
    recommender: TwoStageRecommender
    diagnostics: dict[str, float]
    feature_importance: dict[str, float]


def select_variant(metrics: dict[str, dict[str, float]], k: int = 10) -> str:
    """Serve the ranker only if it strictly beats ALS-only on NDCG@k (spec 2.5)."""
    key = f"ndcg_at_{k}"
    return "als_ranker" if metrics["als_ranker"][key] > metrics["als_only"][key] else "als_only"


def _candidate_lists(candidates: pd.DataFrame) -> dict[int, list[int]]:
    ordered = candidates.sort_values(["user_id", "als_rank"], kind="stable")
    return {
        int(u): [int(m) for m in g["movie_id"]] for u, g in ordered.groupby("user_id", sort=False)
    }


def validate_als(
    split: Split, n_items: int, als_config: ALSConfig, k: int = 10, n_candidates: int = 100
) -> dict[str, float]:
    """ALS-only quality for model selection: fit on als_train, score on ranker_train.

    Never touches `split.test`. Also reports the recall of the full candidate list
    (`candidate_recall_at_{n}`), the ceiling for any re-ranker.
    """
    ctx = make_validation_context(split, n_items=n_items, k=k)
    als = ALSCandidateGenerator(als_config).fit(ctx.history)
    candidates = als.candidates(ctx.users, n=max(n_candidates, k))
    lists = _candidate_lists(candidates)
    metrics = evaluate_recommendations({u: lst[:k] for u, lst in lists.items()}, ctx)
    recall = evaluate(lists, ctx.target, n_items=n_items, k=n_candidates)
    metrics[f"candidate_recall_at_{n_candidates}"] = recall[f"recall_at_{n_candidates}"]
    return metrics


def train_ranker(
    split: Split, movies: pd.DataFrame, users: pd.DataFrame | None, config: TrainConfig
) -> tuple[LambdaRanker, dict[str, float]]:
    """Stage A: ALS on als_train -> candidates for ranker users -> labels from ranker_train."""
    als = ALSCandidateGenerator(config.als).fit(split.als_train)
    ranker_users = sorted(int(u) for u in split.ranker_train["user_id"].unique())
    candidates = als.candidates(ranker_users, n=config.n_candidates)
    genres = movie_genre_matrix(movies)
    user_feats = drop_demographics(
        user_features(split.als_train, genres, users),
        fraction=config.ranker.demographics_dropout,
        seed=config.ranker.random_state,
    )
    frame = build_training_frame(
        candidates, split.ranker_train, movie_features(split.als_train, movies), user_feats
    )
    ranker = LambdaRanker(config.ranker).fit(frame)
    targets = split.ranker_train[split.ranker_train["user_id"].isin(ranker_users)]
    diagnostics = {
        "ranker_als_n_ratings": float(als.n_train_ratings),
        "ranker_rows": float(len(frame)),
        "ranker_groups": float(frame["user_id"].nunique()),
        "ranker_positive_rate": float(frame["label"].mean()) if len(frame) else 0.0,
        "ranker_candidate_recall": float(frame["label"].sum() / max(len(targets), 1)),
        "ranker_best_iteration": float(ranker.best_iteration),
    }
    return ranker, diagnostics


def train_and_evaluate(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame | None,
    config: TrainConfig = TrainConfig(),  # noqa: B008 (frozen dataclass)
) -> TrainResult:
    validate_ratings(ratings)
    split = temporal_split(ratings)
    ranker, diagnostics = train_ranker(split, movies, users, config)

    ctx = make_test_context(split, n_items=len(movies), k=config.k)
    als = ALSCandidateGenerator(config.als).fit(ctx.history)
    genres = movie_genre_matrix(movies)
    full = TwoStageRecommender(
        als,
        movie_features(ctx.history, movies),
        user_features(ctx.history, genres, users),
        ranker=ranker,
        n_candidates=config.n_candidates,
    )
    als_only = full.without_ranker()

    def als_only_model(c: EvalContext) -> dict[int, list[int]]:
        return als_only.recommend_known_users(c.users, k=c.k)

    def als_ranker_model(c: EvalContext) -> dict[int, list[int]]:
        return full.recommend_known_users(c.users, k=c.k)

    models = {
        "popularity": popularity_model,
        "als_only": als_only_model,
        "als_ranker": als_ranker_model,
    }
    metrics = compare_models(models, ctx)
    variant = select_variant(metrics, k=config.k)
    diagnostics |= {
        "eval_als_n_ratings": float(als.n_train_ratings),
        "n_als_train": float(len(split.als_train)),
        "n_ranker_train": float(len(split.ranker_train)),
        "n_test": float(len(split.test)),
        "n_test_users": float(len(ctx.users)),
    }
    return TrainResult(
        metrics=metrics,
        variant=variant,
        recommender=full if variant == "als_ranker" else als_only,
        diagnostics=diagnostics,
        feature_importance=ranker.feature_importance(),
    )
