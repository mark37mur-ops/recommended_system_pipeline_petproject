"""Baseline pipeline: popularity recommender evaluated on the temporal test split."""

import argparse
from pathlib import Path

from recsys.data.movielens import load_movies, load_ratings
from recsys.data.split import Split, temporal_split
from recsys.data.validate import validate_ratings
from recsys.evaluation import (
    EvalContext,
    build_history,
    evaluate_recommendations,
    make_test_context,
)
from recsys.models.popularity import PopularityRecommender

__all__ = ["build_history", "popularity_model", "recommend_for_test_users", "run_baseline"]


def popularity_model(ctx: EvalContext) -> dict[int, list[int]]:
    """Fit popularity on the context history and recommend k unseen movies to every user."""
    model = PopularityRecommender().fit(ctx.history)
    return model.recommend_batch(ctx.users, ctx.seen, k=ctx.k)


def recommend_for_test_users(split: Split, k: int = 10) -> dict[int, list[int]]:
    """Popularity recommendations for the test users, under the shared test protocol."""
    return popularity_model(make_test_context(split, n_items=1, k=k))


def run_baseline(raw_dir: Path, k: int = 10) -> dict[str, float]:
    ratings = load_ratings(raw_dir)
    movies = load_movies(raw_dir)
    validate_ratings(ratings)

    ctx = make_test_context(temporal_split(ratings), n_items=len(movies), k=k)
    return evaluate_recommendations(popularity_model(ctx), ctx)


def positive_int(value: str) -> int:
    """argparse type: an integer >= 1."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the popularity baseline.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--k", type=positive_int, default=10)
    args = parser.parse_args()
    for name, value in run_baseline(args.data_dir, k=args.k).items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
