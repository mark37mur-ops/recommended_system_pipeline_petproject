"""MLflow pyfunc wrapper around the two-stage recommender bundle (spec 4.3).

The model is logged "from code" (`pyfunc_entry.py`), so no pickled Python object is stored:
the bundle directory (ALS, ranker, feature tables) is an MLflow artifact, and the loading
environment must have the `recsys` package installed (the API image does).

`predict` input: ratings DataFrame (user_id, movie_id, rating), one or many users; params
`{"k": 10}`. Output: RECOMMENDATION_COLUMNS (user_id, rank, movie_id, score), produced by
fold-in on the given ratings, i.e. the same path the API uses for a web visitor.
"""

import tempfile
from importlib.metadata import version
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

from recsys.models.two_stage import RECOMMENDATION_COLUMNS, TwoStageRecommender

ENTRY_POINT = Path(__file__).with_name("pyfunc_entry.py")
BUNDLE_ARTIFACT = "bundle"
SIGNATURE = ModelSignature(
    inputs=Schema(
        [ColSpec("long", "user_id"), ColSpec("long", "movie_id"), ColSpec("double", "rating")]
    ),
    outputs=Schema(
        [
            ColSpec("long", "user_id"),
            ColSpec("long", "rank"),
            ColSpec("long", "movie_id"),
            ColSpec("double", "score"),
        ]
    ),
    params=ParamSchema([ParamSpec("k", "long", 10)]),
)
_REQUIREMENTS = ("implicit", "lightgbm", "mlflow", "numpy", "pandas", "pyarrow", "scipy")


class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        self.recommender: TwoStageRecommender | None = None

    def load_context(self, context) -> None:
        self.recommender = TwoStageRecommender.load(Path(context.artifacts[BUNDLE_ARTIFACT]))

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None):
        if self.recommender is None:
            raise RuntimeError("model context is not loaded")
        k = int((params or {}).get("k", 10))
        ratings = pd.DataFrame(model_input)[["user_id", "movie_id", "rating"]]
        out = self.recommender.recommend_for_ratings(ratings, k=k)
        return out.astype({"user_id": "int64", "rank": "int64", "movie_id": "int64"})[
            RECOMMENDATION_COLUMNS
        ]


def pip_requirements() -> list[str]:
    return [f"{name}=={version(name)}" for name in _REQUIREMENTS]


def log_recommender(
    recommender: TwoStageRecommender, name: str = "model", tmp_root: Path | None = None
) -> ModelInfo:
    """Log the recommender as a pyfunc model inside the active MLflow run."""
    if tmp_root is not None:
        Path(tmp_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        bundle = Path(tmp) / BUNDLE_ARTIFACT
        recommender.save(bundle)
        # A real input example also smoke-tests the bundle: MLflow loads it and predicts once.
        example = pd.DataFrame(
            {
                "user_id": [0, 0, 0],
                "movie_id": recommender.als.movie_ids[:3].astype("int64"),
                "rating": [5.0, 4.0, 3.0],
            }
        )
        return mlflow.pyfunc.log_model(
            name=name,
            python_model=str(ENTRY_POINT),
            artifacts={BUNDLE_ARTIFACT: str(bundle)},
            signature=SIGNATURE,
            input_example=example,
            pip_requirements=pip_requirements(),
        )
