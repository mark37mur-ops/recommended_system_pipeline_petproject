"""Stage 1 of the two-stage model: ALS candidate generator (`implicit`) with fold-in.

Ratings become confidence weights (`implicit` reads the matrix value as confidence, every
non-zero cell is a positive). Internally `implicit` works with row/column numbers; this class
keeps the mapping to `user_id` / `movie_id` and always talks in real ids.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares
from numpy.typing import ArrayLike
from threadpoolctl import threadpool_limits

RATING_WEIGHTS = ("linear", "binary")
CANDIDATE_COLUMNS = ["user_id", "movie_id", "als_score", "als_rank"]
_FILTERED_SCORE = np.finfo(np.float32).min / 2


def confidence_weights(ratings: ArrayLike, scheme: str = "linear") -> np.ndarray:
    """Map ratings 1..5 to ALS confidence: `linear` keeps the rating (5 = strong, 1 = weak),
    `binary` treats every rating as the same positive signal."""
    values = np.asarray(ratings, dtype=np.float32)
    if scheme == "linear":
        return values
    if scheme == "binary":
        return np.ones_like(values)
    raise ValueError(f"unknown rating_weight {scheme!r}, expected one of {RATING_WEIGHTS}")


@dataclass(frozen=True)
class ALSConfig:
    factors: int = 64
    regularization: float = 0.05
    alpha: float = 1.0
    iterations: int = 15
    rating_weight: str = "linear"
    random_state: int = 42
    num_threads: int = 0


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": pd.Series(dtype="int64"),
            "movie_id": pd.Series(dtype="int64"),
            "als_score": pd.Series(dtype="float32"),
            "als_rank": pd.Series(dtype="int32"),
        }
    )


class ALSCandidateGenerator:
    def __init__(self, config: ALSConfig = ALSConfig()) -> None:  # noqa: B008 (frozen dataclass)
        confidence_weights(np.array([1]), config.rating_weight)  # fail fast on a bad scheme
        self.config = config
        self._model: AlternatingLeastSquares | None = None
        self._user_ids = np.empty(0, dtype=np.int64)
        self._movie_ids = np.empty(0, dtype=np.int64)
        self._user_items: sp.csr_matrix | None = None
        self.n_train_ratings = 0

    # ----- fitting -------------------------------------------------------------------------
    def fit(self, ratings: pd.DataFrame) -> Self:
        """Fit on a ratings frame with columns user_id, movie_id, rating."""
        self._user_ids = np.sort(ratings["user_id"].unique()).astype(np.int64)
        self._movie_ids = np.sort(ratings["movie_id"].unique()).astype(np.int64)
        rows = np.searchsorted(self._user_ids, ratings["user_id"].to_numpy())
        cols = np.searchsorted(self._movie_ids, ratings["movie_id"].to_numpy())
        data = confidence_weights(ratings["rating"].to_numpy(), self.config.rating_weight)
        self._user_items = sp.csr_matrix(
            (data, (rows, cols)), shape=(len(self._user_ids), len(self._movie_ids))
        )
        self._model = self._new_model()
        with threadpool_limits(1, "blas"):
            self._model.fit(self._user_items, show_progress=False)
        self.n_train_ratings = len(ratings)
        return self

    def _new_model(self) -> AlternatingLeastSquares:
        c = self.config
        with threadpool_limits(1, "blas"):
            return AlternatingLeastSquares(
                factors=c.factors,
                regularization=c.regularization,
                alpha=c.alpha,
                iterations=c.iterations,
                random_state=c.random_state,
                num_threads=c.num_threads,
            )

    @property
    def model(self) -> AlternatingLeastSquares:
        if self._model is None:
            raise RuntimeError("fit() or load() must be called first")
        return self._model

    @property
    def user_ids(self) -> np.ndarray:
        return self._user_ids

    @property
    def movie_ids(self) -> np.ndarray:
        return self._movie_ids

    def has_user(self, user_id: int) -> bool:
        i = np.searchsorted(self._user_ids, user_id)
        return bool(i < len(self._user_ids) and self._user_ids[i] == user_id)

    # ----- candidates ----------------------------------------------------------------------
    def candidates(self, user_ids: Sequence[int], n: int = 100) -> pd.DataFrame:
        """Top-n unseen movies for users known at fit time (their fitted factors).

        Seen = the user's ratings in the training data. Raises KeyError for unknown users.
        """
        users = np.asarray(user_ids, dtype=np.int64)
        missing = [int(u) for u in users if not self.has_user(int(u))]
        if missing:
            raise KeyError(f"users not seen at fit time: {missing[:5]}")
        if len(users) == 0:
            return _empty_candidates()
        rows = np.searchsorted(self._user_ids, users)
        assert self._user_items is not None
        with threadpool_limits(1, "blas"):
            ids, scores = self.model.recommend(
                rows, self._user_items[rows], N=self._n(n), filter_already_liked_items=True
            )
        return self._to_frame(users, ids, scores)

    def candidates_for_ratings(self, ratings: pd.DataFrame, n: int = 100) -> pd.DataFrame:
        """Fold-in: recompute each user's factors from the given ratings with item factors fixed
        (`recalculate_user=True`), then return top-n movies not in those ratings.

        Works for any user id (new web visitors included). Movies unknown to the model are
        ignored for the factors; users with no known movies get no candidates.
        """
        users = np.sort(ratings["user_id"].unique()).astype(np.int64)
        col = pd.Index(self._movie_ids).get_indexer(ratings["movie_id"].to_numpy())
        known = col >= 0
        if len(users) == 0 or not known.any():
            return _empty_candidates()
        rows = np.searchsorted(users, ratings["user_id"].to_numpy()[known])
        data = confidence_weights(ratings["rating"].to_numpy()[known], self.config.rating_weight)
        matrix = sp.csr_matrix((data, (rows, col[known])), shape=(len(users), len(self._movie_ids)))
        has_known = np.diff(matrix.indptr) > 0
        users, matrix = users[has_known], matrix[has_known]
        with threadpool_limits(1, "blas"):
            ids, scores = self.model.recommend(
                np.arange(len(users)),
                matrix,
                N=self._n(n),
                filter_already_liked_items=True,
                recalculate_user=True,
            )
        return self._to_frame(users, ids, scores)

    def _n(self, n: int) -> int:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        return min(n, len(self._movie_ids))

    def _to_frame(self, users: np.ndarray, ids: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
        ids, scores = np.atleast_2d(ids), np.atleast_2d(scores)
        # With fewer unseen items than N, implicit pads the tail with filtered (seen) items whose
        # score is the float32 minimum; drop those and anything non-finite or out of range.
        valid = (
            np.isfinite(scores)
            & (scores > _FILTERED_SCORE)
            & (ids >= 0)
            & (ids < len(self._movie_ids))
        )
        user_col = np.repeat(users, ids.shape[1]).reshape(ids.shape)
        rank = np.tile(np.arange(1, ids.shape[1] + 1, dtype=np.int32), (ids.shape[0], 1))
        return pd.DataFrame(
            {
                "user_id": user_col[valid].astype(np.int64),
                "movie_id": self._movie_ids[ids[valid]],
                "als_score": scores[valid].astype(np.float32),
                "als_rank": rank[valid],
            }
        )

    # ----- persistence ---------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "als_model.npz").open("wb") as f:
            self.model.save(f)
        np.savez(path / "als_index.npz", user_ids=self._user_ids, movie_ids=self._movie_ids)
        assert self._user_items is not None
        sp.save_npz(path / "als_user_items.npz", self._user_items)
        meta = {"config": asdict(self.config), "n_train_ratings": self.n_train_ratings}
        (path / "als_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        path = Path(path)
        meta = json.loads((path / "als_meta.json").read_text(encoding="utf-8"))
        obj = cls(ALSConfig(**meta["config"]))
        with threadpool_limits(1, "blas"):
            obj._model = AlternatingLeastSquares.load(str(path / "als_model.npz"))
        with np.load(path / "als_index.npz") as index:
            obj._user_ids = index["user_ids"]
            obj._movie_ids = index["movie_ids"]
        obj._user_items = sp.load_npz(path / "als_user_items.npz").tocsr()
        obj.n_train_ratings = int(meta["n_train_ratings"])
        return obj
