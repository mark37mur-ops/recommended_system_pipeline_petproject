"""Popularity baseline: recommend the most-rated movies, ties broken by smaller movie_id."""

from collections.abc import Iterable, Mapping

import pandas as pd


class PopularityRecommender:
    def __init__(self) -> None:
        self._ranked: list[int] | None = None

    def fit(self, ratings: pd.DataFrame) -> "PopularityRecommender":
        counts = ratings.groupby("movie_id").size().rename("n").reset_index()
        ranked = counts.sort_values(["n", "movie_id"], ascending=[False, True])
        self._ranked = [int(m) for m in ranked["movie_id"]]
        return self

    def recommend(self, k: int = 10, exclude: set[int] | None = None) -> list[int]:
        if self._ranked is None:
            raise RuntimeError("fit() must be called before recommend()")
        excluded = exclude or set()
        out: list[int] = []
        for movie_id in self._ranked:
            if len(out) >= k:
                break
            if movie_id not in excluded:
                out.append(movie_id)
        return out

    def recommend_batch(
        self,
        user_ids: Iterable[int],
        seen: Mapping[int, set[int]],
        k: int = 10,
    ) -> dict[int, list[int]]:
        return {int(u): self.recommend(k=k, exclude=seen.get(u)) for u in user_ids}
