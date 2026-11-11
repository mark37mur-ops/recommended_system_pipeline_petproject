"""DAG integrity: the DAG files import cleanly in Airflow and have the shape of spec 4.2.

Needs Airflow, which is not in the project venv (its pins conflict with ours). Runs inside the
Airflow image, locally and in CI:
    docker run --rm -v "$PWD:/repo" -w /repo --entrypoint bash movielens-recsys-airflow:local \
        -c "pip install -q pytest && python -m pytest -p no:cacheprovider tests/dags"
Skipped everywhere else.
"""

import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402

DAGS_FOLDER = Path(__file__).resolve().parents[2] / "airflow" / "dags"
RECSYS_PYTHON = os.environ.get("RECSYS_PYTHON", "/opt/recsys-venv/bin/python")

RETRAIN_EDGES = {
    ("extract_ratings", "validate_data"),
    ("validate_data", "drift_report"),
    ("drift_report", "split"),
    ("split", "train_als"),
    ("train_als", "train_ranker"),
    ("train_ranker", "evaluate"),
    ("evaluate", "decide_promotion"),
    ("decide_promotion", "promote"),
    ("promote", "set_champion_alias"),
    ("set_champion_alias", "reload_api"),
    ("decide_promotion", "skip"),
    ("skip", "log_reason"),
}
BOOTSTRAP_EDGES = {("download_ml1m", "verify_checksum"), ("verify_checksum", "load_postgres")}


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    # Examples are off via AIRFLOW__CORE__LOAD_EXAMPLES=false (the image default).
    return DagBag(dag_folder=str(DAGS_FOLDER))


def _edges(dag) -> set[tuple[str, str]]:
    return {(t.task_id, down) for t in dag.tasks for down in t.downstream_task_ids}


def test_dags_import_without_errors(dagbag):
    assert dagbag.import_errors == {}
    assert set(dagbag.dag_ids) == {"bootstrap_data", "retrain_daily"}


def test_retrain_daily_has_the_spec_graph(dagbag):
    dag = dagbag.dags["retrain_daily"]
    assert _edges(dag) == RETRAIN_EDGES
    assert dag.task_dict["decide_promotion"].task_type.startswith("_BranchExternalPython")


def test_bootstrap_data_has_the_spec_graph(dagbag):
    assert _edges(dagbag.dags["bootstrap_data"]) == BOOTSTRAP_EDGES


@pytest.mark.parametrize(
    ("dag_id", "schedule"), [("retrain_daily", "@daily"), ("bootstrap_data", None)]
)
def test_scheduling_and_retries(dagbag, dag_id, schedule):
    dag = dagbag.dags[dag_id]
    assert dag.schedule == schedule
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    for t in dag.tasks:
        assert t.retries == 2, t.task_id


def test_recsys_tasks_run_in_the_recsys_venv(dagbag):
    for dag_id in ("retrain_daily", "bootstrap_data"):
        for t in dagbag.dags[dag_id].tasks:
            if t.task_id != "skip":
                assert t.python == RECSYS_PYTHON, t.task_id
                assert t.expect_airflow is False


@pytest.mark.skipif(not Path(RECSYS_PYTHON).exists(), reason="only inside the Airflow image")
def test_recsys_venv_imports_the_pipeline_with_its_own_pins():
    code = (
        "import recsys.pipelines.retrain, recsys.pipelines.reload, recsys.pipelines.bootstrap;"
        "import sqlalchemy, pandas; print(sqlalchemy.__version__, pandas.__version__)"
    )
    out = subprocess.run([RECSYS_PYTHON, "-c", code], capture_output=True, text=True, check=True)
    venv_sqlalchemy = out.stdout.split()[0]
    import sqlalchemy

    assert venv_sqlalchemy.startswith("2.1")  # uv.lock
    assert sqlalchemy.__version__.startswith("2.0")  # Airflow's constraints
