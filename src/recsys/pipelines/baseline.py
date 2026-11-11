"""Baseline pipeline: popularity recommender evaluated on the temporal test split."""

import argparse
from pathlib import Path

import pandas as pd

from recsys.data.movielens import load_movies, load_ratings
from recsys.data.split import temporal_split
from recsys.data.validate import validate_ratings
from recsys.metrics import evaluate
from recsys.models.popularity import PopularityRecommender


def run_baseline(raw_dir: Path, k: int = 10) -> dict[str, float]:
    ratings = load_ratings(raw_dir)
    movies = load_movies(raw_dir)
    validate_ratings(ratings)

    split = temporal_split(ratings)
    history = pd.concat([split.als_train, split.ranker_train], ignore_index=True)
    seen = {
        int(user): {int(m) for m in group["movie_id"]} for user, group in history.groupby("user_id")
    }

    model = PopularityRecommender().fit(history)
    test_users = sorted(int(u) for u in split.test["user_id"].unique())
    recommendations = model.recommend_batch(test_users, seen, k=k)
    return evaluate(recommendations, split.test, n_items=len(movies), k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the popularity baseline.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    for name, value in run_baseline(args.data_dir, k=args.k).items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
