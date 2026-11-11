"""Settings shared by the pipeline tasks (where the database, MLflow, the API and run folders
are). Each task runs in its own process (Airflow external_python), so everything not passed
explicitly comes from that process's environment, which compose sets."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from recsys.pipelines.train import TrainConfig
from recsys.tracking import EXPERIMENT_NAME, MODEL_NAME, resolve_tracking_uri


@dataclass(frozen=True)
class PipelineSettings:
    """Where things live. `None` means: take it from the environment of the task process
    (DATABASE_URL, MLFLOW_TRACKING_URI, RUNS_DIR, API_URL, ADMIN_TOKEN; set in compose)."""

    database_url: str | None = None
    tracking_uri: str | None = None
    experiment: str = EXPERIMENT_NAME
    model_name: str = MODEL_NAME
    runs_root: Path | None = None
    train_config: TrainConfig = field(default_factory=TrainConfig)
    api_url: str | None = None
    admin_token: str | None = None

    def db_url(self) -> str:
        url = self.database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL is not set (the `app` database)")
        return url

    def mlflow_uri(self) -> str:
        return resolve_tracking_uri(self.tracking_uri)


def prepare(settings: PipelineSettings | None) -> PipelineSettings:
    """Defaults + logging for a bare task process (INFO lines end up in the Airflow task log)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return settings or PipelineSettings()
