"""Two-stage recommender: ALS candidates -> features -> LightGBM re-ranking -> top-k.

This object is what gets registered in MLflow (wrapped by `recsys.models.pyfunc`) and what the
API serves. It is self-contained: ALS factors and id maps, the ranker (optional: `als_only`
variant when the ranker does not beat ALS), the movie feature table and the features of users
known at training time. It saves to / loads from a plain directory ("bundle").
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from recsys.features import GENRE_COLUMNS, build_feature_frame, user_features
from recsys.models.als import ALSCandidateGenerator
from recsys.models.ranker import LambdaRanker

RECOMMENDATION_COLUMNS = ["user_id", "rank", "movie_id", "score"]
BUNDLE_FORMAT_VERSION = 1
NEW_USER_ID = 0


class TwoStageRecommender:
    def __init__(
        self,
        als: ALSCandidateGenerator,
        movie_feats: pd.DataFrame,
        user_feats: pd.DataFrame,
        ranker: LambdaRanker | None = None,
        n_candidates: int = 100,
    ) -> None:
        self.als = als
        self.movie_feats = movie_feats
        self.user_feats = user_feats
        self.ranker = ranker
        self.n_candidates = n_candidates

    @property
    def variant(self) -> str:
        return "als_only" if self.ranker is None else "als_ranker"

    @property
    def movie_genres(self) -> pd.DataFrame:
        return self.movie_feats[GENRE_COLUMNS]

    def without_ranker(self) -> "TwoStageRecommender":
        return TwoStageRecommender(
            self.als, self.movie_feats, self.user_feats, None, self.n_candidates
        )

    # ----- ranking -------------------------------------------------------------------------
    def _rank(self, candidates: pd.DataFrame, user_feats: pd.DataFrame, k: int) -> pd.DataFrame:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        scored = candidates[["user_id", "movie_id", "als_rank", "als_score"]].copy()
        if self.ranker is None or scored.empty:
            scored["score"] = scored["als_score"].astype(np.float64)
        else:
            features = build_feature_frame(candidates, self.movie_feats, user_feats)
            scored["score"] = self.ranker.predict(features)
        scored = scored.sort_values(
            ["user_id", "score", "als_rank"], ascending=[True, False, True], kind="stable"
        )
        top = scored.groupby("user_id", sort=False).head(k).copy()
        top["rank"] = top.groupby("user_id", sort=False).cumcount() + 1
        return top[RECOMMENDATION_COLUMNS].reset_index(drop=True)

    @staticmethod
    def _as_dict(ranked: pd.DataFrame, user_ids: Sequence[int]) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {int(u): [] for u in user_ids}
        for user, group in ranked.groupby("user_id", sort=False):
            out[int(user)] = [int(m) for m in group["movie_id"]]
        return out

    # ----- public API ----------------------------------------------------------------------
    def recommend_known_users(self, user_ids: Sequence[int], k: int = 10) -> dict[int, list[int]]:
        """Top-k for users seen at training time, from their fitted ALS factors.

        Seen items (the training history) are never recommended. Used by offline evaluation.
        """
        candidates = self.als.candidates(list(user_ids), n=max(self.n_candidates, k))
        return self._as_dict(self._rank(candidates, self.user_feats, k), user_ids)

    def user_features_for(
        self, ratings: pd.DataFrame, profiles: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """User features computed from the given ratings (the serving path)."""
        return user_features(ratings, self.movie_genres, profiles)

    def recommend_for_ratings(
        self, ratings: pd.DataFrame, k: int = 10, profiles: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Fold-in path for any users: ratings (user_id, movie_id, rating) -> top-k frame.

        `profiles` (user_id, gender, age, occupation) is optional; missing -> NaN demographics.
        Movies already in `ratings` are never recommended. Users whose rated movies are all
        unknown to the model get no rows. Returns RECOMMENDATION_COLUMNS.
        """
        ratings = ratings.drop_duplicates(["user_id", "movie_id"], keep="last")
        candidates = self.als.candidates_for_ratings(ratings, n=max(self.n_candidates, k))
        return self._rank(candidates, self.user_features_for(ratings, profiles), k)

    def recommend_new_user(
        self,
        ratings: Sequence[tuple[int, int]],
        k: int = 10,
        profile: Mapping[str, object] | None = None,
    ) -> list[int]:
        """Top-k movie ids for one visitor from a list of (movie_id, rating).

        `profile` may hold gender ("F"/"M"), age and occupation; missing keys -> NaN.
        """
        frame = pd.DataFrame(list(ratings), columns=["movie_id", "rating"])
        frame.insert(0, "user_id", NEW_USER_ID)
        profiles = None
        if profile is not None:
            profiles = pd.DataFrame(
                {
                    "user_id": [NEW_USER_ID],
                    "gender": [profile.get("gender")],
                    "age": [profile.get("age")],
                    "occupation": [profile.get("occupation")],
                }
            )
        ranked = self.recommend_for_ratings(frame, k=k, profiles=profiles)
        return [int(m) for m in ranked["movie_id"]]

    # ----- persistence ---------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.als.save(path / "als")
        if self.ranker is not None:
            self.ranker.save(path / "ranker")
        self.movie_feats.to_parquet(path / "movie_features.parquet")
        self.user_feats.to_parquet(path / "user_features.parquet")
        meta = {
            "format_version": BUNDLE_FORMAT_VERSION,
            "variant": self.variant,
            "n_candidates": self.n_candidates,
        }
        (path / "bundle.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        path = Path(path)
        meta = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
        if meta["format_version"] != BUNDLE_FORMAT_VERSION:
            raise ValueError(f"unsupported bundle format {meta['format_version']}")
        ranker = LambdaRanker.load(path / "ranker") if meta["variant"] == "als_ranker" else None
        return cls(
            als=ALSCandidateGenerator.load(path / "als"),
            movie_feats=pd.read_parquet(path / "movie_features.parquet"),
            user_feats=pd.read_parquet(path / "user_features.parquet"),
            ranker=ranker,
            n_candidates=int(meta["n_candidates"]),
        )
