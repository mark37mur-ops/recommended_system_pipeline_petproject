"""Per-user temporal split of ratings into ALS-train / ranker-train / test."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    als_train: pd.DataFrame
    ranker_train: pd.DataFrame
    test: pd.DataFrame


def temporal_split(
    ratings: pd.DataFrame, n_ranker: int = 5, n_test: int = 5, min_ratings: int = 15
) -> Split:
    """Split each user's history by time: oldest -> als_train, then ranker_train, then test.

    Users with fewer than `min_ratings` ratings go entirely to als_train. Ties in timestamp
    are broken by movie_id ascending, so the split is deterministic.
    """
    df = ratings.sort_values(["user_id", "timestamp", "movie_id"], kind="stable")
    grouped = df.groupby("user_id", sort=False)
    pos_from_end = grouped.cumcount(ascending=False)
    long_enough = grouped["user_id"].transform("size") >= min_ratings

    is_test = long_enough & (pos_from_end < n_test)
    is_ranker = long_enough & (pos_from_end >= n_test) & (pos_from_end < n_test + n_ranker)
    is_als = ~(is_test | is_ranker)

    def part(mask: pd.Series) -> pd.DataFrame:
        return df[mask].reset_index(drop=True)

    return Split(als_train=part(is_als), ranker_train=part(is_ranker), test=part(is_test))
