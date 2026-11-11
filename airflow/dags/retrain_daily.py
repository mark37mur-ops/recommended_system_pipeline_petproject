"""retrain_daily: retrain the recommender on everything in Postgres and promote it if it holds
up against the current champion (spec 4.2).

    extract_ratings -> validate_data -> drift_report -> split -> train_als -> train_ranker
      -> evaluate -> decide_promotion -> promote -> set_champion_alias -> reload_api
                                      -> skip    -> log_reason

Params (Trigger DAG w/ config):
- `include_web=false`: train on MovieLens ratings only (spec 3.4). Such a run is
  evaluation-only: compared and logged, never promoted...
- `allow_promotion=true`: ...unless this is set too (incident: web data poisoned, a
  MovieLens-only model has to replace the champion; the spec 4.2 rule still applies).
- `allow_count_drop=true`: accept >5% fewer ratings than the previous run, or all web
  ratings gone (after a deliberate clean-up; otherwise every later run fails as well).
"""

import os
from datetime import UTC, datetime, timedelta

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import Param, dag, task

# Why a second interpreter: Airflow 3.3 and recsys pin incompatible versions of the same
# libraries (SQLAlchemy 2.0 vs 2.1, pandas 3.0 vs 3.1, pyarrow 25 vs 26). The image therefore
# has two virtualenvs. The scheduler stays in Airflow's, each task body runs in
# /opt/recsys-venv, which is built from the same uv.lock as the API image (identical
# versions for training and serving).
RECSYS_PYTHON = os.environ.get("RECSYS_PYTHON", "/opt/recsys-venv/bin/python")

# Why the bodies are two lines: the logic lives in recsys (tested without Airflow), the DAG
# only wires it up (spec 4.1). external_python ships the body's *source* to the other
# interpreter, so it cannot use names from this file: every import sits inside the body.
# Why XCom carries only a path: the data itself (Parquet, models) lives in /data/runs/{run_id}
# on a shared volume. XCom rows sit in Airflow's metadata DB and are meant for small values.
RECSYS = {"python": RECSYS_PYTHON, "expect_airflow": False}

# After the branch, tasks read the run folder from evaluate's XCom instead of taking it as an
# argument: an argument would add an evaluate -> promote edge and blur the branch in the graph.
RUN_DIR = "{{ ti.xcom_pull(task_ids='evaluate') }}"


@dag(
    dag_id="retrain_daily",
    schedule="@daily",
    start_date=datetime(2026, 11, 1, tzinfo=UTC),
    # catchup=False: one retrain on today's data is what we want, not one per missed day.
    catchup=False,
    # Two runs at once would race for the champion alias and double the memory peak.
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=2),
    # Every task is idempotent per run (spec 4.1), so retrying is always safe.
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
    params={
        "include_web": Param(True, type="boolean", description="train on web ratings too"),
        "allow_promotion": Param(
            False,
            type="boolean",
            description="let an include_web=false run replace the champion (incident only)",
        ),
        "allow_count_drop": Param(
            False,
            type="boolean",
            description="accept >5% fewer ratings or no web ratings (after a clean-up)",
        ),
    },
    # Templates render to Python objects, so params arrive as bools. recsys parses the
    # string forms as well (parse_flag): "False" must never be taken as true.
    render_template_as_native_obj=True,
    tags=["recsys", "training"],
    doc_md=__doc__,
)
def retrain_daily():
    @task.external_python(**RECSYS)
    # Not called `run_id`: that name is reserved for Airflow's context, and Airflow refuses a
    # task argument shadowing it (only at run time, see tests/dags).
    def extract_ratings(airflow_run_id: str, include_web: bool) -> str:
        from recsys.pipelines import retrain

        return retrain.extract_ratings(airflow_run_id, include_web=include_web)

    @task.external_python(**RECSYS)
    def validate_data(run_dir: str, allow_count_drop: bool) -> str:
        from recsys.pipelines import retrain

        return retrain.validate_data(run_dir, allow_count_drop=allow_count_drop)

    @task.external_python(**RECSYS)
    def drift_report(run_dir: str) -> str:
        from recsys.pipelines import retrain

        return retrain.drift_report(run_dir)

    @task.external_python(task_id="split", **RECSYS)
    def split_data(run_dir: str) -> str:
        from recsys.pipelines import retrain

        return retrain.split_data(run_dir)

    @task.external_python(**RECSYS)
    def train_als(run_dir: str) -> str:
        from recsys.pipelines import retrain

        return retrain.train_als(run_dir)

    @task.external_python(**RECSYS)
    def train_ranker(run_dir: str) -> str:
        from recsys.pipelines import retrain

        return retrain.train_ranker(run_dir)

    @task.external_python(**RECSYS)
    def evaluate(run_dir: str) -> str:
        from recsys.pipelines import retrain

        return retrain.evaluate(run_dir)

    # The branch returns the id of the task to run next ("promote" or "skip"); Airflow marks
    # the other branch as skipped.
    @task.branch_external_python(**RECSYS)
    def decide_promotion(run_dir: str, allow_promotion: bool) -> str:
        from recsys.pipelines import retrain

        return retrain.decide_promotion(run_dir, allow_promotion=allow_promotion)

    @task.external_python(**RECSYS)
    def promote(run_dir: str) -> str:
        from recsys.pipelines import retrain

        return retrain.register_challenger(run_dir)

    @task.external_python(**RECSYS)
    def set_champion_alias(run_dir: str) -> str:
        from recsys.pipelines import retrain

        return retrain.set_champion_alias(run_dir)

    # The alias has already moved when this runs: if the API cannot be reached, the task fails
    # after its retries (red in the UI) and the model stays promoted (Decision 10).
    @task.external_python(**RECSYS)
    def reload_api(run_dir: str) -> str:
        from recsys.pipelines import reload

        return reload.reload_api(run_dir)

    @task.external_python(**RECSYS)
    def log_reason(run_dir: str) -> str:
        from recsys.pipelines import retrain

        return retrain.log_reason(run_dir)

    run_dir = extract_ratings("{{ run_id }}", "{{ params.include_web }}")
    run_dir = validate_data(run_dir, "{{ params.allow_count_drop }}")
    for step in (drift_report, split_data, train_als, train_ranker, evaluate):
        run_dir = step(run_dir)
    branch = decide_promotion(run_dir, "{{ params.allow_promotion }}")

    promoted = promote(RUN_DIR)
    branch >> promoted
    reload_api(set_champion_alias(promoted))

    skip = EmptyOperator(task_id="skip")
    branch >> skip >> log_reason(RUN_DIR)


retrain_daily()
