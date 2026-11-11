"""Tasks of `retrain_daily` without Airflow: synthetic snapshot, SQLite run state, SQLite MLflow.

The chain is run the way the DAG runs it (one function per task, data only through run_dir),
so these tests are also the "pipeline end-to-end on a small synthetic dataset" of spec 6.5.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sqlalchemy import create_engine, func, select
from synthetic import make_movies, make_ratings, make_users

from recsys.data.validate import DataValidationError
from recsys.db.run_state import RunRecord, create_run_state, previous_run, record_run
from recsys.db.schema import WEB_USER_ID_START, retrain_runs
from recsys.models.als import ALSConfig
from recsys.models.ranker import RankerConfig
from recsys.pipelines import retrain
from recsys.pipelines.retrain import (
    PROMOTE_TASK,
    SKIP_TASK,
    PipelineSettings,
    decide_promotion,
    drift_report,
    evaluate,
    log_reason,
    register_challenger,
    set_champion_alias,
    split_data,
    train_als,
    train_ranker,
    validate_data,
    write_snapshot,
)
from recsys.pipelines.run_dir import RunDir, read_json
from recsys.pipelines.train import TrainConfig
from recsys.tracking import CHAMPION_ALIAS

SMALL = TrainConfig(
    als=ALSConfig(factors=8, regularization=0.1, iterations=10, num_threads=1),
    ranker=RankerConfig(n_estimators=30, min_child_samples=5, early_stopping_rounds=10, n_jobs=1),
    n_candidates=30,
)
MODEL = "test-retrain-model"


def web_ratings(n_users: int = 3, per_user: int = 20, seed: int = 1) -> pd.DataFrame:
    """Web visitors (ids from WEB_USER_ID_START) who like Comedy, rated "now"."""
    rng = np.random.default_rng(seed)
    comedies = [m for m in range(1, 91) if m % 3 == 0]
    rows = []
    for i in range(n_users):
        user = WEB_USER_ID_START + i
        for pos, movie in enumerate(rng.choice(comedies, size=per_user, replace=False)):
            rows.append((user, int(movie), int(rng.integers(4, 6)), 1_790_000_000 + pos))
    frame = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    return frame.assign(source="web").astype(
        {"user_id": "int32", "movie_id": "int32", "rating": "int8"}
    )


def snapshot(with_web: bool = True) -> pd.DataFrame:
    ratings = make_ratings().assign(source="movielens")
    if with_web:
        ratings = pd.concat([ratings, web_ratings()], ignore_index=True)
    return ratings


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("retrain")
    db_url = f"sqlite:///{(root / 'app.db').as_posix()}"
    engine = create_engine(db_url)
    create_run_state(engine)
    engine.dispose()
    return PipelineSettings(
        database_url=db_url,
        tracking_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        experiment="test-retrain",
        model_name=MODEL,
        runs_root=root / "runs",
        train_config=SMALL,
    )


def run_until_decision(run_id: str, env: PipelineSettings, ratings: pd.DataFrame) -> str:
    rd = write_snapshot(
        RunDir.for_run(run_id, env.runs_root),
        run_id,
        ratings,
        make_movies(),
        make_users(),
        include_web=True,
    )
    run_dir = str(rd.root)
    for task in (validate_data, drift_report, split_data, train_als, train_ranker, evaluate):
        assert task(run_dir, env) == run_dir  # XCom carries only the run folder
    return run_dir


@pytest.fixture(scope="module")
def first(env):
    """First run ever: no champion -> promote -> v1 is champion."""
    run_dir = run_until_decision("manual__first:1", env, snapshot())
    branch = decide_promotion(run_dir, env)
    assert branch == PROMOTE_TASK
    register_challenger(run_dir, env)
    set_champion_alias(run_dir, env)
    return run_dir


def client(env: PipelineSettings) -> MlflowClient:
    return MlflowClient(tracking_uri=env.tracking_uri, registry_uri=env.tracking_uri)


def mlflow_run(env, run_dir):
    return client(env).get_run(read_json(RunDir(run_dir).mlflow_json)["run_id"])


# ----- validate_data ---------------------------------------------------------------------


def _validate(env, tmp_path, run_id, ratings, include_web=True):
    rd = write_snapshot(
        RunDir.for_run(run_id, tmp_path), run_id, ratings, make_movies(), make_users(), include_web
    )
    return validate_data(str(rd.root), dataclasses.replace(env, runs_root=tmp_path))


def _recorded(env, run_id) -> bool:
    engine = create_engine(env.database_url)
    with engine.connect() as conn:
        n = conn.execute(
            select(func.count()).select_from(retrain_runs).where(retrain_runs.c.run_id == run_id)
        ).scalar_one()
    engine.dispose()
    return n == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda r: r.astype({"rating": "float64"}).assign(
                rating=lambda d: d["rating"].where(d.index != 3)
            ),
            "null",
        ),
        (lambda r: r.assign(rating=lambda d: d["rating"].where(d.index != 0, 0)), "outside 1..5"),
        (lambda r: r.assign(rating=lambda d: d["rating"].where(d.index != 0, 6)), "outside 1..5"),
        (lambda r: pd.concat([r, r.iloc[[5]]], ignore_index=True), "duplicate"),
        (lambda r: r.iloc[0:0], "no ratings"),
    ],
    ids=["null", "zero", "six", "duplicate", "empty"],
)
def test_validate_fails_loudly_and_records_nothing(env, tmp_path, mutate, message):
    run_id = f"bad-{message}-{tmp_path.name}"
    with pytest.raises(DataValidationError, match=message):
        _validate(env, tmp_path, run_id, mutate(snapshot()))
    assert not _recorded(env, run_id)


def test_validate_row_count_drop_rule(env, tmp_path):
    engine = create_engine(env.database_url)
    base = snapshot(with_web=False)
    n = len(base)
    record_run(engine, RunRecord("drop-base", False, n, "x", {"n_ratings": n}))
    engine.dispose()

    with pytest.raises(DataValidationError, match="dropped"):
        _validate(env, tmp_path, "drop-6pct", base.iloc[: int(n * 0.94)], include_web=False)
    assert not _recorded(env, "drop-6pct")
    _validate(env, tmp_path, "drop-4pct", base.iloc[: int(n * 0.96)], include_web=False)
    assert _recorded(env, "drop-4pct")


# ----- the full chain ----------------------------------------------------------------------


def test_first_run_promotes_and_logs_everything(env, first):
    rd = RunDir(first)
    for path in (rd.ratings_path, rd.split_path("test"), rd.als_dir, rd.ranker_dir, rd.model_dir):
        assert path.exists()
    decision = read_json(rd.decision_json)
    assert decision["promote"] is True and decision["reason"] == "no current champion"

    run = mlflow_run(env, first)
    tags, metrics, params = run.data.tags, run.data.metrics, run.data.params
    assert float(tags["web_share"]) > 0
    assert tags["n_ratings"] == str(len(snapshot()))
    assert tags["airflow_run_id"] == "manual__first:1"
    assert {"data_snapshot", "git_sha", "n_users", "variant"} <= set(tags)
    assert tags["promoted"] == "true" and tags["registered_version"] == "1"
    for model in ("popularity", "als_only", "als_ranker", "challenger_foldin"):
        assert f"{model}_ndcg_at_10" in metrics and f"{model}_coverage" in metrics
    assert "drift_web_share" in metrics
    assert params["als.factors"] == "8" and params["ranker.n_estimators"] == "30"
    artifacts = {a.path for a in client(env).list_artifacts(run.info.run_id)}
    assert {"drift", "comparison.json", "movie_features.parquet"} <= artifacts

    champion = client(env).get_model_version_by_alias(MODEL, CHAMPION_ALIAS)
    assert str(champion.version) == "1"
    assert champion.tags["airflow_run_id"] == "manual__first:1"


def test_previous_count_and_drift_compare_with_the_previous_run(env, first):
    engine = create_engine(env.database_url)
    assert previous_run(engine, "next-run", include_web=True).run_id == "manual__first:1"
    engine.dispose()
    drift = read_json(RunDir(first).drift_json)
    assert drift["engine"] == "summary-v1"
    assert drift["current"]["n_web_ratings"] == len(web_ratings())


def test_tasks_are_idempotent_for_the_same_run(env, first):
    rd = RunDir(first)
    split_before = pd.read_parquet(rd.split_path("test"))
    eval_before = read_json(rd.evaluate_json)
    versions_before = client(env).search_model_versions(f"name='{MODEL}'")

    for task in (validate_data, drift_report, split_data, train_als, train_ranker, evaluate):
        assert task(first, env) == first
    pd.testing.assert_frame_equal(pd.read_parquet(rd.split_path("test")), split_before)
    assert decide_promotion(first, env) == PROMOTE_TASK
    register_challenger(first, env)  # retry after success: no second version
    set_champion_alias(first, env)  # alias already there: no-op
    versions_after = client(env).search_model_versions(f"name='{MODEL}'")
    assert len(versions_after) == len(versions_before)
    eval_after = read_json(rd.evaluate_json)
    assert eval_after["metrics"] == eval_before["metrics"]
    assert eval_after["challenger"] == eval_before["challenger"]
    assert str(client(env).get_model_version_by_alias(MODEL, CHAMPION_ALIAS).version) == "1"


def test_second_run_evaluates_the_champion_on_the_same_split(env, first):
    run_dir = run_until_decision("manual__second", env, snapshot())
    ev = read_json(RunDir(run_dir).evaluate_json)
    assert ev["champion_version"] == "1"
    assert set(ev["champion"]) == {"ndcg_at_10", "recall_at_10", "coverage"}
    # same data, same seeds: the champion (v1) is the challenger's twin
    assert ev["champion"] == pytest.approx(ev["challenger"])
    metrics = mlflow_run(env, run_dir).data.metrics
    assert metrics["champion_foldin_ndcg_at_10"] == pytest.approx(ev["champion"]["ndcg_at_10"])

    assert decide_promotion(run_dir, env) == PROMOTE_TASK  # within tolerance -> fresher model
    register_challenger(run_dir, env)
    set_champion_alias(run_dir, env)
    assert str(client(env).get_model_version_by_alias(MODEL, CHAMPION_ALIAS).version) == "2"


def test_evaluate_is_deterministic_for_the_same_run(env, tmp_path):
    settings = dataclasses.replace(env, runs_root=tmp_path, model_name="no-such-model")
    run_dir = run_until_decision("determinism", settings, snapshot())
    first_eval = read_json(RunDir(run_dir).evaluate_json)
    for task in (split_data, train_als, train_ranker, evaluate):
        task(run_dir, settings)
    again = read_json(RunDir(run_dir).evaluate_json)
    assert again["metrics"] == first_eval["metrics"]
    assert again["challenger"] == first_eval["challenger"]
    assert again["champion"] is None and again["champion_version"] is None


def test_extract_reuses_a_frozen_snapshot(env, tmp_path, monkeypatch):
    rd = write_snapshot(
        RunDir.for_run("frozen", tmp_path), "frozen", snapshot(), make_movies(), make_users(), True
    )
    monkeypatch.setattr(retrain, "extract_snapshot", lambda *a, **k: pytest.fail("re-extracted"))
    settings = dataclasses.replace(env, runs_root=tmp_path)
    assert retrain.extract_ratings("frozen", settings) == str(rd.root)


def test_extract_writes_the_snapshot_once(env, tmp_path, monkeypatch):
    calls = []

    def fake_extract(engine, include_web=True):
        calls.append(include_web)
        return snapshot(), make_movies(), make_users()

    monkeypatch.setattr(retrain, "extract_snapshot", fake_extract)
    settings = dataclasses.replace(env, runs_root=tmp_path)
    run_dir = retrain.extract_ratings("fresh:run", settings, include_web=False)
    assert run_dir == str(RunDir.for_run("fresh:run", tmp_path).root)
    assert read_json(RunDir(run_dir).extract_json)["run_id"] == "fresh:run"
    assert read_json(RunDir(run_dir).extract_json)["include_web"] is False
    retrain.extract_ratings("fresh:run", settings, include_web=False)
    assert calls == [False]


# ----- decide_promotion branches, guards ------------------------------------------------


def _fake_evaluation(rd: RunDir, challenger, champion, beats_popularity=True):
    from recsys.pipelines.run_dir import write_json_atomic

    rd.root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(rd.mlflow_json, {"run_id": "unused", "experiment_id": "0"})
    write_json_atomic(
        rd.evaluate_json,
        {
            "k": 10,
            "challenger": challenger,
            "champion": champion,
            "champion_version": None if champion is None else "7",
            "beats_popularity": beats_popularity,
        },
    )


@pytest.mark.parametrize(
    ("challenger", "champion", "beats", "branch", "reason"),
    [
        ({"ndcg_at_10": 0.05, "coverage": 0.2}, None, True, PROMOTE_TASK, "no current champion"),
        (
            {"ndcg_at_10": 0.096, "coverage": 0.2},
            {"ndcg_at_10": 0.1, "coverage": 0.2},
            True,
            PROMOTE_TASK,
            "within tolerance",
        ),
        (
            {"ndcg_at_10": 0.094, "coverage": 0.2},
            {"ndcg_at_10": 0.1, "coverage": 0.2},
            True,
            SKIP_TASK,
            "< champion",
        ),
        (
            {"ndcg_at_10": 0.12, "coverage": 0.17},
            {"ndcg_at_10": 0.1, "coverage": 0.2},
            True,
            SKIP_TASK,
            "coverage",
        ),
        ({"ndcg_at_10": 0.12, "coverage": 0.2}, None, False, SKIP_TASK, "popularity"),
    ],
    ids=["no-champion", "within-tolerance", "worse", "coverage-drop", "not-beating-popularity"],
)
def test_decide_promotion_branches(
    env, tmp_path, monkeypatch, challenger, champion, beats, branch, reason
):
    monkeypatch.setattr(retrain, "_set_run_tags", lambda *a, **k: None)
    rd = RunDir(tmp_path / "decide")
    _fake_evaluation(rd, challenger, champion, beats)
    assert decide_promotion(str(rd.root), env) == branch
    decision = read_json(rd.decision_json)
    assert decision["promote"] is (branch == PROMOTE_TASK)
    assert reason in decision["reason"]


def test_register_refuses_when_the_decision_was_skip(env, tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "_set_run_tags", lambda *a, **k: None)
    rd = RunDir(tmp_path / "skip")
    _fake_evaluation(
        rd, {"ndcg_at_10": 0.01, "coverage": 0.2}, {"ndcg_at_10": 0.1, "coverage": 0.2}
    )
    assert decide_promotion(str(rd.root), env) == SKIP_TASK
    with pytest.raises(RuntimeError, match="skip"):
        register_challenger(str(rd.root), env)


def test_log_reason_tags_the_run(env, tmp_path):
    settings = dataclasses.replace(env, runs_root=tmp_path, model_name="skip-model")
    run_dir = run_until_decision("skip-run", settings, snapshot())
    rd = RunDir(run_dir)
    ev = read_json(rd.evaluate_json)
    # pretend there is a much better champion -> skip
    ev["champion"] = {"ndcg_at_10": 1.0, "recall_at_10": 1.0, "coverage": 1.0}
    ev["champion_version"] = "3"
    from recsys.pipelines.run_dir import write_json_atomic

    write_json_atomic(rd.evaluate_json, ev)
    assert decide_promotion(run_dir, settings) == SKIP_TASK
    assert log_reason(run_dir, settings) == run_dir
    tags = mlflow_run(settings, run_dir).data.tags
    assert tags["promoted"] == "false"
    assert "< champion" in tags["promotion_reason"]
    assert client(settings).search_model_versions("name='skip-model'") == []


def test_set_champion_alias_refuses_when_the_champion_moved(env, tmp_path):
    settings = dataclasses.replace(env, runs_root=tmp_path, model_name="race-model")
    run_dir = run_until_decision("race", settings, snapshot())
    assert decide_promotion(run_dir, settings) == PROMOTE_TASK
    register_challenger(run_dir, settings)
    # someone (a manual trainer) promotes another version after evaluate saw "no champion"
    other = client(settings).create_model_version("race-model", "file:///nowhere", run_id=None)
    client(settings).set_registered_model_alias("race-model", CHAMPION_ALIAS, other.version)
    with pytest.raises(RuntimeError, match="moved"):
        set_champion_alias(run_dir, settings)
    alias = client(settings).get_model_version_by_alias("race-model", CHAMPION_ALIAS)
    assert str(alias.version) == str(other.version)
