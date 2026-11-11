"""Training pipeline: popularity -> ALS-only -> ALS + ranker on one leakage-free protocol.

Stage A (ranker training data): ALS is fit on `als_train` only; its candidates for the
ranker-train users, with features computed from `als_train`, are labelled by `ranker_train`.
Stage B (evaluation): ALS is refit on `als_train ∪ ranker_train` (the history), features are
recomputed from the history, and the three models are evaluated on `test` through the shared
`EvalContext`. The ranker never sees `test`, and the ALS whose score it learns from never saw
its labels.
"""

import argparse
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import mlflow
import pandas as pd

from recsys.data.movielens import load_movies, load_ratings, load_users
from recsys.data.split import Split, temporal_split
from recsys.data.validate import validate_ratings
from recsys.evaluation import (
    EvalContext,
    compare_models,
    evaluate_recommendations,
    format_comparison,
    make_test_context,
    make_validation_context,
)
from recsys.features import drop_demographics, movie_features, movie_genre_matrix, user_features
from recsys.metrics import evaluate
from recsys.models.als import RATING_WEIGHTS, ALSCandidateGenerator, ALSConfig
from recsys.models.pyfunc import log_recommender
from recsys.models.ranker import LambdaRanker, RankerConfig, build_training_frame
from recsys.models.two_stage import TwoStageRecommender
from recsys.pipelines.baseline import popularity_model, positive_int
from recsys.tracking import (
    EXPERIMENT_NAME,
    MODEL_NAME,
    RegistrationOutcome,
    configure_mlflow,
    data_snapshot_hash,
    git_dirty,
    git_sha,
    register_and_promote,
)

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
    if ranker.valid_ndcg is not None:
        diagnostics["ranker_valid_ndcg_at_10"] = ranker.valid_ndcg
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


# ----- MLflow orchestration and CLI ----------------------------------------------------------

# Validation sweep (scored on ranker_train, never on test). `binary` is a reference point only:
# the spec asks for rating-dependent confidence, so only SELECTABLE_WEIGHTS can be picked.
SELECTABLE_WEIGHTS = ("linear",)
DEFAULT_SWEEP_GRID: tuple[ALSConfig, ...] = tuple(
    ALSConfig(factors=f, regularization=r, alpha=a)
    for f in (32, 64, 128)
    for r in (0.05, 5.0)
    for a in (0.25, 0.5, 1.0, 2.0)
) + (ALSConfig(rating_weight="binary", alpha=1.0),)


def flatten_config(config: TrainConfig) -> dict[str, object]:
    flat: dict[str, object] = {"n_candidates": config.n_candidates, "k": config.k}
    flat |= {f"als.{key}": value for key, value in asdict(config.als).items()}
    flat |= {f"ranker.{key}": value for key, value in asdict(config.ranker).items()}
    return flat


def run_metrics(result: TrainResult, k: int = 10) -> dict[str, float]:
    """Everything numeric that goes into the training run."""
    metrics = {
        f"{model}_{name}": value
        for model, values in result.metrics.items()
        for name, value in values.items()
    }
    metrics |= result.metrics[result.variant]  # served variant under plain names
    metrics |= result.diagnostics
    return metrics


def select_best_als(sweep: Sequence[tuple[ALSConfig, dict[str, float]]], k: int = 10) -> ALSConfig:
    """Best validation NDCG@k among configs whose weighting the spec allows (first on ties)."""
    eligible = [(c, m) for c, m in sweep if c.rating_weight in SELECTABLE_WEIGHTS]
    if not eligible:
        raise ValueError(f"sweep has no config with rating_weight in {SELECTABLE_WEIGHTS}")
    return max(eligible, key=lambda cm: cm[1][f"ndcg_at_{k}"])[0]


@dataclass
class TrainingRun:
    result: TrainResult
    config: TrainConfig
    run_id: str
    experiment_id: str
    model_uri: str
    registration: RegistrationOutcome | None
    sweep: list[tuple[ALSConfig, dict[str, float]]]


def run_training(
    raw_dir: Path,
    config: TrainConfig = TrainConfig(),  # noqa: B008 (frozen dataclass)
    tracking_uri: str | None = None,
    experiment: str = EXPERIMENT_NAME,
    model_name: str = MODEL_NAME,
    sweep_grid: Sequence[ALSConfig] | None = None,
    register: bool = True,
    tmp_root: Path | None = None,
    echo: Callable[[str], None] = print,
) -> TrainingRun:
    """Load data, optionally sweep ALS on validation, train + evaluate, log to MLflow,
    register the served variant and move the `champion` alias if it deserves it."""
    raw_dir = Path(raw_dir)
    ratings, movies, users = load_ratings(raw_dir), load_movies(raw_dir), load_users(raw_dir)
    validate_ratings(ratings)
    experiment_id = configure_mlflow(tracking_uri, experiment)
    snapshot = data_snapshot_hash(ratings)
    data_tags = {
        "git_sha": git_sha(),
        "git_dirty": str(git_dirty()).lower(),
        "data_snapshot": snapshot,
        "n_users": str(ratings["user_id"].nunique()),
        "n_ratings": str(len(ratings)),
        "web_share": "0.0",  # MovieLens only until web ratings arrive (stage 3)
    }
    if tmp_root is not None:
        Path(tmp_root).mkdir(parents=True, exist_ok=True)
    k = config.k
    sweep: list[tuple[ALSConfig, dict[str, float]]] = []

    with mlflow.start_run(experiment_id=experiment_id, run_name="train") as run:
        mlflow.set_tags({**data_tags, "run_type": "train"})
        if sweep_grid:
            split = temporal_split(ratings)
            for i, als_config in enumerate(sweep_grid):
                with mlflow.start_run(
                    experiment_id=experiment_id, run_name=f"sweep-{i:02d}", nested=True
                ):
                    mlflow.set_tags({**data_tags, "run_type": "sweep"})
                    mlflow.log_params({f"als.{key}": v for key, v in asdict(als_config).items()})
                    values = validate_als(split, len(movies), als_config, k, config.n_candidates)
                    mlflow.log_metrics({f"val_{key}": v for key, v in values.items()})
                sweep.append((als_config, values))
                echo(f"sweep {i:02d} {als_config}: val ndcg_at_{k}={values[f'ndcg_at_{k}']:.4f}")
            config = replace(config, als=select_best_als(sweep, k))
            echo(f"best ALS config on validation: {config.als}")

        mlflow.log_params(flatten_config(config))
        result = train_and_evaluate(ratings, movies, users, config)
        beats_popularity = (
            result.metrics["als_only"][f"ndcg_at_{k}"]
            > result.metrics["popularity"][f"ndcg_at_{k}"]
        )
        mlflow.log_metrics(run_metrics(result, k))
        mlflow.set_tags(
            {"variant": result.variant, "beats_popularity": str(beats_popularity).lower()}
        )
        table = format_comparison(result.metrics, k)
        mlflow.log_text(table + "\n", "comparison.md")
        mlflow.log_dict(result.metrics, "comparison.json")
        mlflow.log_dict(result.feature_importance, "feature_importance.json")
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            features_path = Path(tmp) / "movie_features.parquet"
            result.recommender.movie_feats.to_parquet(features_path)
            mlflow.log_artifact(str(features_path))
        model_info = log_recommender(result.recommender, tmp_root=tmp_root)

    echo(table)
    echo(f"served variant: {result.variant}")
    if not beats_popularity:
        echo("WARNING: ALS-only does not beat popularity on this split")
    echo(f"mlflow run: {run.info.run_id} (experiment {experiment_id})")
    registration = None
    if register:
        registration = register_and_promote(
            model_info.model_uri,
            run.info.run_id,
            result.metrics[result.variant],
            snapshot,
            tags={"variant": result.variant, "git_sha": data_tags["git_sha"]},
            name=model_name,
            k=k,
        )
        verdict = "promoted to champion" if registration.promoted else "not promoted"
        echo(f"registered {model_name} v{registration.version}: {verdict} ({registration.reason})")
    return TrainingRun(
        result, config, run.info.run_id, experiment_id, model_info.model_uri, registration, sweep
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = ALSConfig()
    parser = argparse.ArgumentParser(description="Train ALS + LightGBM ranker, log to MLflow.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="default: $MLFLOW_TRACKING_URI, else sqlite:///mlruns/mlflow.db",
    )
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--sweep", action="store_true", help="pick ALS params on validation")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--n-candidates", type=positive_int, default=100)
    parser.add_argument("--factors", type=positive_int, default=defaults.factors)
    parser.add_argument("--regularization", type=float, default=defaults.regularization)
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument("--iterations", type=positive_int, default=defaults.iterations)
    parser.add_argument("--rating-weight", choices=RATING_WEIGHTS, default=defaults.rating_weight)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> TrainingRun:
    args = parse_args(argv)
    als = ALSConfig(
        factors=args.factors,
        regularization=args.regularization,
        alpha=args.alpha,
        iterations=args.iterations,
        rating_weight=args.rating_weight,
    )
    config = TrainConfig(als=als, n_candidates=args.n_candidates, k=args.k)
    return run_training(
        args.data_dir,
        config,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        model_name=args.model_name,
        sweep_grid=DEFAULT_SWEEP_GRID if args.sweep else None,
        register=not args.no_register,
        tmp_root=Path("mlruns") / "tmp",
    )


if __name__ == "__main__":
    main()
