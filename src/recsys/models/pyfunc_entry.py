"""MLflow "models from code" entry point: executed by MLflow when the model is loaded."""

import mlflow

from recsys.models.pyfunc import RecommenderPyfunc

mlflow.models.set_model(RecommenderPyfunc())
