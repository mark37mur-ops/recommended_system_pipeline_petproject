"""Stage 2 of the two-stage model: LightGBM `lambdarank` re-ranking of ALS candidates.

One query group = one user's candidates. Labels are binary (did the user rate the candidate in
the label period), matching the binary relevance of the offline metric.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from recsys.features import FEATURE_COLUMNS, build_feature_frame


@dataclass(frozen=True)
class RankerConfig:
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 50
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    valid_fraction: float = 0.2
    demographics_dropout: float = 0.1
    random_state: int = 42
    n_jobs: int = 0


def label_candidates(candidates: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    """1 where the (user_id, movie_id) pair of a candidate is present in `targets`, else 0."""
    keys = pd.MultiIndex.from_arrays(
        [candidates["user_id"].to_numpy(np.int64), candidates["movie_id"].to_numpy(np.int64)]
    )
    target_keys = pd.MultiIndex.from_arrays(
        [targets["user_id"].to_numpy(np.int64), targets["movie_id"].to_numpy(np.int64)]
    )
    return keys.isin(target_keys).astype(np.int8)


def build_training_frame(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    movie_feats: pd.DataFrame,
    user_feats: pd.DataFrame,
) -> pd.DataFrame:
    """Feature frame + `label`, sorted by (user_id, als_rank) so groups are contiguous.

    Users whose candidates contain no positive are dropped: in lambdarank they have zero
    gradient and only cost time.
    """
    frame = build_feature_frame(candidates, movie_feats, user_feats)
    frame["label"] = label_candidates(candidates, targets)
    has_positive = frame.groupby("user_id")["label"].transform("max") > 0
    frame = frame[has_positive].sort_values(["user_id", "als_rank"], kind="stable")
    return frame.reset_index(drop=True)


def _groups(frame: pd.DataFrame) -> np.ndarray:
    # frame is sorted by user_id, so groupby order == row order
    return frame.groupby("user_id", sort=True).size().to_numpy()


class LambdaRanker:
    def __init__(self, config: RankerConfig = RankerConfig()) -> None:  # noqa: B008 (frozen)
        self.config = config
        self._booster: lgb.Booster | None = None
        self.best_iteration = 0
        self.valid_ndcg: float | None = None  # NDCG@10 on the held-out users, if any

    def fit(self, frame: pd.DataFrame) -> Self:
        """Train on a frame from `build_training_frame` (user_id, FEATURE_COLUMNS, label).

        A deterministic `valid_fraction` of users is held out for early stopping on NDCG@10.
        """
        c = self.config
        frame = frame.sort_values(["user_id"], kind="stable").reset_index(drop=True)
        users = np.sort(frame["user_id"].unique())
        n_valid = round(c.valid_fraction * len(users)) if len(users) >= 10 else 0
        rng = np.random.default_rng(c.random_state)
        valid_users = rng.choice(users, size=n_valid, replace=False) if n_valid else []
        is_valid = frame["user_id"].isin(valid_users).to_numpy()
        train, valid = frame[~is_valid], frame[is_valid]

        model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=c.n_estimators,
            learning_rate=c.learning_rate,
            num_leaves=c.num_leaves,
            min_child_samples=c.min_child_samples,
            colsample_bytree=c.colsample_bytree,
            random_state=c.random_state,
            n_jobs=c.n_jobs,
            deterministic=True,
            force_row_wise=True,
            importance_type="gain",
            verbose=-1,
        )
        fit_kwargs: dict = {"group": _groups(train)}
        if len(valid):
            fit_kwargs |= {
                "eval_X": (valid[FEATURE_COLUMNS],),
                "eval_y": (valid["label"],),
                "eval_group": [_groups(valid)],
                "eval_at": [10],
                "callbacks": [lgb.early_stopping(c.early_stopping_rounds, verbose=False)],
            }
        model.fit(train[FEATURE_COLUMNS], train["label"], **fit_kwargs)
        self._booster = model.booster_
        self.best_iteration = int(model.best_iteration_ or c.n_estimators)
        if len(valid):
            self.valid_ndcg = float(model.best_score_["valid_0"]["ndcg@10"])
        return self

    @property
    def booster(self) -> lgb.Booster:
        if self._booster is None:
            raise RuntimeError("fit() or load() must be called first")
        return self._booster

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Scores for rows of a feature frame (columns are picked by name, extras ignored)."""
        return self.booster.predict(features[FEATURE_COLUMNS], num_iteration=self.best_iteration)

    def feature_importance(self) -> dict[str, float]:
        gains = self.booster.feature_importance(importance_type="gain")
        return {name: float(g) for name, g in zip(self.booster.feature_name(), gains, strict=True)}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "ranker.txt"), num_iteration=self.best_iteration)
        meta = {
            "config": asdict(self.config),
            "best_iteration": self.best_iteration,
            "valid_ndcg": self.valid_ndcg,
            "features": FEATURE_COLUMNS,
        }
        (path / "ranker_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        path = Path(path)
        meta = json.loads((path / "ranker_meta.json").read_text(encoding="utf-8"))
        if meta["features"] != FEATURE_COLUMNS:
            raise ValueError("ranker was trained with a different feature set")
        obj = cls(RankerConfig(**meta["config"]))
        obj._booster = lgb.Booster(model_file=str(path / "ranker.txt"))
        obj.best_iteration = int(meta["best_iteration"])
        obj.valid_ndcg = meta.get("valid_ndcg")
        return obj
