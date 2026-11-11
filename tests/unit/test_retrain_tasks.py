"""Tasks of `retrain_daily` without Airflow: synthetic snapshot, SQLite run state, SQLite MLflow.

The chain is run the way the DAG runs it (one function per task, data only through run_dir),
so these tests are also the "pipeline end-to-end on a small synthetic dataset" of spec 6.5.
"""

import dataclasses
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sqlalchemy import create_engine, func, select
from synthetic import make_movies, make_ratings, make_users

from recsys.data.validate import DataValidationError
from recsys.db.run_state import RunRecord, create_run_state, previous_run, record_run
from recsys.db.schema import (
    WEB_USER_ID_START,
    drift_metrics,
    recommendation_impressions,
    retrain_runs,
    run_metrics,
)
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


def run_until_decision(
    run_id: str, env: PipelineSettings, ratings: pd.DataFrame, include_web: bool = True
) -> str:
    rd = write_snapshot(
        RunDir.for_run(run_id, env.runs_root),
        run_id,
        ratings,
        make_movies(),
        make_users(),
        include_web=include_web,
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


def _validate(env, tmp_path, run_id, ratings, include_web=True, **kwargs):
    rd = write_snapshot(
        RunDir.for_run(run_id, tmp_path), run_id, ratings, make_movies(), make_users(), include_web
    )
    return validate_data(str(rd.root), dataclasses.replace(env, runs_root=tmp_path), **kwargs)


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


@pytest.mark.parametrize("allow", [True, "true", "1"])
def test_validate_count_drop_can_be_overridden(env, tmp_path, allow):
    """After a deliberate clean-up (spam removed) every later run would fail on the -5% rule
    forever: failed runs are not recorded, so the baseline never moves. allow_count_drop is the
    documented way out."""
    engine = create_engine(env.database_url)
    base = snapshot(with_web=False)
    n = len(base)
    record_run(engine, RunRecord(f"cleanup-base-{allow}", False, n, "x", {"n_ratings": n}))
    engine.dispose()
    run_id = f"cleanup-{allow}"
    _validate(env, tmp_path, run_id, base.iloc[: n // 2], include_web=False, allow_count_drop=allow)
    assert _recorded(env, run_id)
    assert read_json(RunDir.for_run(run_id, tmp_path).validation_json)["allow_count_drop"] is True


def test_validate_fails_when_all_web_ratings_disappear(env, tmp_path):
    """Web ratings are 0.16% of the data, so losing all of them passes the -5% rule; they are
    the only data that cannot be re-downloaded (spec 6.2), so this is checked on its own."""
    engine = create_engine(env.database_url)
    full = snapshot(with_web=True)
    summary = {"n_ratings": len(full), "n_web_ratings": len(web_ratings())}
    record_run(engine, RunRecord("web-base", True, len(full), "x", summary))
    engine.dispose()

    no_web = snapshot(with_web=False)
    with pytest.raises(DataValidationError, match="web ratings"):
        _validate(env, tmp_path, "web-gone", no_web, include_web=True)
    assert not _recorded(env, "web-gone")
    _validate(env, tmp_path, "web-gone-ok", no_web, include_web=True, allow_count_drop=True)
    assert _recorded(env, "web-gone-ok")


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
    # where the model came from is visible in the registry, not only on the run (review I1)
    assert champion.tags["include_web"] == "true"
    assert champion.tags["n_web_ratings"] == str(len(web_ratings()))
    assert champion.tags["data_snapshot"] == tags["data_snapshot"]


def test_previous_count_and_drift_compare_with_the_previous_run(env, first):
    engine = create_engine(env.database_url)
    assert previous_run(engine, "next-run", include_web=True).run_id == "manual__first:1"
    engine.dispose()
    drift = read_json(RunDir(first).drift_json)
    assert drift["engine"] == "evidently"
    assert drift["summary"]["engine"] == "summary-v1"
    assert drift["summary"]["current"]["n_web_ratings"] == len(web_ratings())


def test_tasks_are_idempotent_for_the_same_run(env, first):
    rd = RunDir(first)
    split_before = pd.read_parquet(rd.split_path("test"))
    eval_before = read_json(rd.evaluate_json)
    versions_before = client(env).search_model_versions(f"name='{MODEL}'")
    promotion_before = read_json(rd.promotion_json)

    for task in (validate_data, drift_report, split_data, train_als, train_ranker, evaluate):
        assert task(first, env) == first
    pd.testing.assert_frame_equal(pd.read_parquet(rd.split_path("test")), split_before)
    # After the promotion the champion is this run's own model. Re-running evaluate must not
    # compare the model with itself, re-log it, or change what the decision was based on.
    assert read_json(rd.evaluate_json) == eval_before
    assert decide_promotion(first, env) == PROMOTE_TASK
    assert "already promoted v1" in read_json(rd.decision_json)["reason"]
    register_challenger(first, env)  # retry after success: no second version
    set_champion_alias(first, env)  # alias already there: no-op
    versions_after = client(env).search_model_versions(f"name='{MODEL}'")
    assert len(versions_after) == len(versions_before)
    assert read_json(rd.promotion_json) == promotion_before  # previous champion: still none
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("True", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("1", True),
        (0, False),
        (1, True),
        (" no ", False),
        ("yes", True),
    ],
)
def test_parse_flag_accepts_bools_and_their_string_forms(value, expected):
    """Templates render "{{ params.x }}" to a string unless render_template_as_native_obj is
    on; "False" is truthy, so a plain bool() would silently include the web ratings."""
    assert retrain.parse_flag(value, "x") is expected


@pytest.mark.parametrize("value", ["maybe", "", None, 2, 0.5])
def test_parse_flag_rejects_anything_else(value):
    with pytest.raises(ValueError, match="include_web"):
        retrain.parse_flag(value, "include_web")


def test_extract_reads_string_flags_correctly(env, tmp_path, monkeypatch):
    calls = []

    def fake_extract(engine, include_web=True):
        calls.append(include_web)
        return snapshot(), make_movies(), make_users()

    monkeypatch.setattr(retrain, "extract_snapshot", fake_extract)
    settings = dataclasses.replace(env, runs_root=tmp_path)
    run_dir = retrain.extract_ratings("string-flag", settings, include_web="False")
    assert calls == [False]
    assert read_json(RunDir(run_dir).extract_json)["include_web"] is False


def test_extract_refuses_a_folder_of_another_run(env, tmp_path):
    """safe_run_id maps "a:b" and "a-b" to one folder; the second run must not silently train
    on the first one's frozen snapshot (review M9)."""
    write_snapshot(
        RunDir.for_run("a:b", tmp_path), "a:b", snapshot(), make_movies(), make_users(), True
    )
    settings = dataclasses.replace(env, runs_root=tmp_path)
    with pytest.raises(RuntimeError, match="a:b"):
        retrain.extract_ratings("a-b", settings)


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


def _fake_evaluation(rd: RunDir, challenger, champion, beats_popularity=True, include_web=True):
    from recsys.pipelines.run_dir import write_json_atomic

    rd.extract_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(rd.extract_json, {"run_id": "fake", "include_web": include_web})
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


BETTER = {"ndcg_at_10": 0.12, "coverage": 0.2}
WITHIN = {"ndcg_at_10": 0.096, "coverage": 0.2}
WORSE = {"ndcg_at_10": 0.05, "coverage": 0.2}
CHAMP = {"ndcg_at_10": 0.1, "coverage": 0.2}


@pytest.mark.parametrize(
    ("challenger", "champion", "allow", "branch", "reason"),
    [
        (BETTER, CHAMP, False, SKIP_TASK, "evaluation-only"),
        (BETTER, None, False, SKIP_TASK, "evaluation-only"),  # even without a champion
        (BETTER, CHAMP, "False", SKIP_TASK, "evaluation-only"),
        (BETTER, CHAMP, "0", SKIP_TASK, "evaluation-only"),
        (BETTER, CHAMP, True, PROMOTE_TASK, "within tolerance"),
        (WITHIN, CHAMP, "true", PROMOTE_TASK, "within tolerance"),
        (WORSE, CHAMP, True, SKIP_TASK, "< champion"),  # the override does not bypass the rule
    ],
    ids=["default", "no-champion", "str-False", "str-0", "allowed", "allowed-str", "allowed-worse"],
)
def test_no_web_runs_are_evaluation_only_unless_allowed(
    env, tmp_path, monkeypatch, challenger, champion, allow, branch, reason
):
    """Review I1: a MovieLens-only model is measured without the API's live audience, so it
    must never replace the champion as a side effect of an experiment. allow_promotion is the
    explicit switch for the incident case (web data poisoned)."""
    monkeypatch.setattr(retrain, "_set_run_tags", lambda *a, **k: None)
    rd = RunDir(tmp_path / "noweb")
    _fake_evaluation(rd, challenger, champion, include_web=False)
    assert decide_promotion(str(rd.root), env, allow_promotion=allow) == branch
    decision = read_json(rd.decision_json)
    assert reason in decision["reason"]
    assert decision["include_web"] is False
    assert decision["allow_promotion"] is retrain.parse_flag(allow, "allow_promotion")


def test_allow_promotion_changes_nothing_for_full_data_runs(env, tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "_set_run_tags", lambda *a, **k: None)
    for allow in (False, True):
        rd = RunDir(tmp_path / f"web-{allow}")
        _fake_evaluation(rd, WORSE, CHAMP, include_web=True)
        assert decide_promotion(str(rd.root), env, allow_promotion=allow) == SKIP_TASK
        _fake_evaluation(rd, WITHIN, CHAMP, include_web=True)
        assert decide_promotion(str(rd.root), env, allow_promotion=allow) == PROMOTE_TASK


def test_decide_after_promotion_reports_it_instead_of_comparing_again(env, tmp_path, monkeypatch):
    """Review M2: "Clear" on decide_promotion after a successful promote must not end in a
    skip ("champion v4 stays") while the alias already points at this run's model."""
    from recsys.pipelines.run_dir import write_json_atomic

    monkeypatch.setattr(retrain, "_set_run_tags", lambda *a, **k: None)
    rd = RunDir(tmp_path / "done")
    _fake_evaluation(rd, WORSE, CHAMP)  # would be a skip on a fresh comparison
    write_json_atomic(
        rd.promotion_json, {"version": "9", "expected_previous": "7", "promoted": True}
    )
    assert decide_promotion(str(rd.root), env) == PROMOTE_TASK
    decision = read_json(rd.decision_json)
    assert decision["promote"] is True and "already promoted v9" in decision["reason"]


def test_no_web_chain_skips_and_registers_nothing(env, first, caplog):
    """The whole chain with include_web=false, as the DAG runs it: evaluated and logged,
    skip branch, the champion and the registry untouched."""
    before = client(env).get_model_version_by_alias(MODEL, CHAMPION_ALIAS).version
    n_versions = len(client(env).search_model_versions(f"name='{MODEL}'"))
    run_dir = run_until_decision("manual__noweb", env, snapshot(with_web=False), include_web=False)
    ev = read_json(RunDir(run_dir).evaluate_json)
    assert ev["champion"] is not None  # the comparison is still made and logged
    assert decide_promotion(run_dir, env) == SKIP_TASK
    with caplog.at_level("WARNING", logger="recsys.retrain"):
        log_reason(run_dir, env)
    assert "evaluation-only" in caplog.text
    tags = mlflow_run(env, run_dir).data.tags
    assert tags["decision"] == SKIP_TASK and tags["promoted"] == "false"
    assert tags["include_web"] == "false" and tags["allow_promotion"] == "false"
    assert "evaluation-only" in tags["promotion_reason"]
    assert client(env).get_model_version_by_alias(MODEL, CHAMPION_ALIAS).version == before
    assert len(client(env).search_model_versions(f"name='{MODEL}'")) == n_versions


def test_register_refuses_when_the_decision_was_skip(env, tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "_set_run_tags", lambda *a, **k: None)
    rd = RunDir(tmp_path / "skip")
    _fake_evaluation(
        rd, {"ndcg_at_10": 0.01, "coverage": 0.2}, {"ndcg_at_10": 0.1, "coverage": 0.2}
    )
    assert decide_promotion(str(rd.root), env) == SKIP_TASK
    with pytest.raises(RuntimeError, match="skip"):
        register_challenger(str(rd.root), env)


def test_log_reason_tags_the_run(env, tmp_path, caplog):
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
    with caplog.at_level("WARNING", logger="recsys.retrain"):
        assert log_reason(run_dir, settings) == run_dir
    # the log names the champion the alias points at now, not the one evaluate saw (M2)
    assert "current champion: none" in caplog.text
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


# ----- monitoring rows (stage 5) ---------------------------------------------------------------


def _rows(env, table, **where):
    engine = create_engine(env.database_url)
    try:
        query = select(table)
        for column, value in where.items():
            query = query.where(table.c[column] == value)
        with engine.connect() as conn:
            return [dict(r) for r in conn.execute(query).mappings()]
    finally:
        engine.dispose()


def test_drift_without_recent_web_data_is_recorded_as_insufficient(env, first):
    # web_ratings() are from September: outside the 7 days before the snapshot
    [row] = _rows(env, drift_metrics, run_id="manual__first:1")
    assert row["drift_share"] is None and row["prediction_drift_share"] is None
    assert row["ratings_status"] == row["recommendations_status"] == "insufficient_data"
    assert row["n_current"] == 0 and row["n_reference"] == len(make_ratings())
    run = mlflow_run(env, first)
    assert "drift_share" not in run.data.metrics  # not computed, not 0
    assert run.data.tags["drift_ratings_status"] == "insufficient_data"
    drift_files = {a.path for a in client(env).list_artifacts(run.info.run_id, "drift")}
    assert {"drift/drift.json", "drift/ratings.html", "drift/recommendations.html"} <= drift_files


def test_evaluate_writes_run_metrics_for_the_dashboard(env, first):
    rows = {r["metric"]: r["value"] for r in _rows(env, run_metrics, run_id="manual__first:1")}
    evaluation = read_json(RunDir(first).evaluate_json)
    assert rows["challenger_foldin_ndcg_at_10"] == pytest.approx(
        evaluation["challenger"]["ndcg_at_10"]
    )
    assert rows["popularity_ndcg_at_10"] == pytest.approx(
        evaluation["metrics"]["popularity"]["ndcg_at_10"]
    )
    assert "champion_foldin_ndcg_at_10" not in rows  # first run: no champion yet


def test_drift_with_fresh_web_ratings_and_impressions(env, tmp_path):
    """Visitors who rated today and were shown the model's recommendations: both Evidently
    sections run, drift_share lands in MLflow and in drift_metrics, a retry keeps one row."""
    settings = dataclasses.replace(env, runs_root=tmp_path)
    now = datetime.now(UTC)
    fresh = web_ratings(n_users=6, per_user=15).assign(
        timestamp=int((now - timedelta(hours=2)).timestamp())
    )
    ratings = pd.concat([make_ratings().assign(source="movielens"), fresh], ignore_index=True)
    engine = create_engine(env.database_url)
    recommendation_impressions.create(engine, checkfirst=True)
    shown_at = now - timedelta(hours=1)
    with engine.begin() as conn:
        conn.execute(
            recommendation_impressions.insert(),
            [
                {
                    "user_id": WEB_USER_ID_START + visitor,
                    "movie_id": m,
                    "position": m,
                    "model_version": "1",
                    "path": "full",
                    "shown_at": shown_at,
                }
                for visitor in range(6)
                for m in range(1, 11)
            ]
            + [  # cold-start impressions are not the model's predictions: not counted
                {
                    "user_id": WEB_USER_ID_START,
                    "movie_id": 1,
                    "position": 1,
                    "model_version": None,
                    "path": "cold_start",
                    "shown_at": shown_at,
                }
            ],
        )
    engine.dispose()
    rd = write_snapshot(
        RunDir.for_run("fresh-drift", tmp_path),
        "fresh-drift",
        ratings,
        make_movies(),
        make_users(),
        include_web=True,
    )
    run_dir = str(rd.root)
    validate_data(run_dir, settings)
    drift_report(run_dir, settings)
    drift_report(run_dir, settings)  # retry: same window, same row

    [row] = _rows(env, drift_metrics, run_id="fresh-drift")
    assert row["ratings_status"] == "ok" and row["n_current"] == len(fresh)
    assert row["recommendations_status"] == "ok" and row["n_current_recommendations"] == 60
    assert 0 < row["drift_share"] <= 1 and row["prediction_drift_share"] is not None
    assert set(row["details"]["ratings"]) == {"rating", "genre", "release_year", "user_activity"}
    report = read_json(rd.drift_json)
    assert report["metrics"]["drift_share"] == pytest.approx(row["drift_share"])
    run = mlflow_run(env, run_dir)
    assert run.data.metrics["drift_share"] == pytest.approx(row["drift_share"])
    assert run.data.metrics["prediction_drift_share"] == pytest.approx(
        row["prediction_drift_share"]
    )
    assert (rd.drift_dir / "ratings.html").stat().st_size > 10_000


def test_retrain_module_does_not_need_evidently_to_import():
    """The API image has no Evidently: importing the pipeline modules must not pull it in."""
    import subprocess
    import sys as _sys

    code = (
        "import sys; import recsys.pipelines.retrain, recsys.pipelines.reload, recsys.drift; "
        "assert 'evidently' not in sys.modules, 'evidently imported eagerly'"
    )
    subprocess.run([_sys.executable, "-c", code], check=True)


def test_cleanup_impressions_deletes_rows_past_the_retention(env, tmp_path):
    """s5-review I-4: recommendation_impressions must not grow without bound."""
    from recsys.db.monitoring import IMPRESSIONS_RETENTION

    db_url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    settings = dataclasses.replace(env, database_url=db_url)
    engine = create_engine(db_url)
    recommendation_impressions.create(engine, checkfirst=True)
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            recommendation_impressions.insert(),
            [
                {
                    "user_id": WEB_USER_ID_START,
                    "movie_id": m,
                    "position": 1,
                    "model_version": "1",
                    "path": "full",
                    "shown_at": at,
                }
                for m, at in ((1, now - IMPRESSIONS_RETENTION - timedelta(days=1)), (2, now))
            ],
        )
    assert retrain.cleanup_impressions(settings) == 1
    assert retrain.cleanup_impressions(settings) == 0
    with engine.connect() as conn:
        left = conn.execute(select(recommendation_impressions.c.movie_id)).scalars().all()
    engine.dispose()
    assert left == [2]
