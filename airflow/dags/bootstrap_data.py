"""bootstrap_data: MovieLens 1M into the `app` database (spec 4.2). Manual, normally once.

    download_ml1m -> verify_checksum -> load_postgres

Safe to run again: a valid archive is not downloaded twice, the loader upserts (an already
loaded database ends up unchanged) and never touches web ratings.
"""

import os
from datetime import UTC, datetime, timedelta

from airflow.sdk import dag, task

# Same split as in retrain_daily.py: Airflow and recsys live in separate virtualenvs because
# their pinned dependencies conflict. The bodies run in the recsys one, so imports sit inside.
RECSYS_PYTHON = os.environ.get("RECSYS_PYTHON", "/opt/recsys-venv/bin/python")
RECSYS = {"python": RECSYS_PYTHON, "expect_airflow": False}


@dag(
    dag_id="bootstrap_data",
    # schedule=None: only a human starts it (first deployment, or to repair the dataset).
    schedule=None,
    start_date=datetime(2026, 11, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    # The download can fail on a flaky network; every step is idempotent, so retry.
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["recsys", "data"],
    doc_md=__doc__,
)
def bootstrap_data():
    @task.external_python(**RECSYS)
    def download_ml1m() -> str:
        from recsys.pipelines import bootstrap

        return bootstrap.download()

    # A separate task (download_ml1m checks too): the file that is loaded is the file that was
    # verified, and a failure here is visibly "bad data", not "network problem".
    @task.external_python(**RECSYS)
    def verify_checksum(archive: str) -> str:
        from recsys.pipelines import bootstrap

        return bootstrap.verify_checksum(archive)

    @task.external_python(**RECSYS)
    def load_postgres(archive: str) -> dict:
        from recsys.pipelines import bootstrap

        return bootstrap.load_postgres(archive)

    load_postgres(verify_checksum(download_ml1m()))


bootstrap_data()
