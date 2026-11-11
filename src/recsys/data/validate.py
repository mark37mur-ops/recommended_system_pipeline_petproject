"""Validation of the ratings table; raises with every problem found."""

import pandas as pd


class DataValidationError(Exception):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def validate_ratings(
    ratings: pd.DataFrame, previous_count: int | None = None, max_drop: float = 0.05
) -> None:
    problems: list[str] = []

    null_counts = ratings.isna().sum()
    for col, n in null_counts[null_counts > 0].items():
        problems.append(f"column '{col}' has {n} null values")

    rating = ratings["rating"]
    if not pd.api.types.is_numeric_dtype(rating) or pd.api.types.is_bool_dtype(rating):
        problems.append(f"column 'rating' is not numeric (dtype {rating.dtype})")
    else:
        present = rating.dropna()
        bad_ratings = int((~present.between(1, 5)).sum())
        if bad_ratings:
            problems.append(f"{bad_ratings} rows have rating outside 1..5")
        non_integer = int((present % 1 != 0).sum())
        if non_integer:
            problems.append(f"{non_integer} rows have non-integer rating")

    dups = int(ratings.duplicated(subset=["user_id", "movie_id"]).sum())
    if dups:
        problems.append(f"{dups} duplicate (user_id, movie_id) pairs")

    if previous_count is not None and len(ratings) < previous_count * (1 - max_drop):
        problems.append(
            f"row count dropped from {previous_count} to {len(ratings)} (more than {max_drop:.0%})"
        )

    if problems:
        raise DataValidationError(problems)
