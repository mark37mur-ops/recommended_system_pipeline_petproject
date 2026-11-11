"""MLflow layer against a throw-away SQLite store in tmp (no server, nothing under ./mlruns)."""

import dataclasses
import inspect
from importlib.metadata import version as package_version
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient
from synthetic import make_ratings, write_raw_dir

from recsys.models.als import ALSConfig
from recsys.models.pyfunc import log_recommender, pip_requirements
from recsys.models.ranker import RankerConfig
from recsys.models.two_stage import RECOMMENDATION_COLUMNS
from recsys.pipelines.train import TrainConfig, main, parse_args, run_training
from recsys.tracking import (
    CHAMPION_ALIAS,
    DEFAULT_TRACKING_URI,
    champion_version,
    configure_mlflow,
    data_snapshot_hash,
    decide_promotion,
    git_sha,
    load_champion,
    register_and_promote,
    resolve_tracking_uri,
)

SMALL = TrainConfig(
    als=ALSConfig(factors=8, regularization=0.1, iterations=10, num_threads=1),
    ranker=RankerConfig(n_estimators=30, min_child_samples=5, early_stopping_rounds=10, n_jobs=1),
    n_candidates=30,
)


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory) -> str:
    db = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    return f"sqlite:///{db.as_posix()}"


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    return write_raw_dir(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="module")
def first_run(tracking_uri, raw_dir, tmp_path_factory):
    """One full training run on synthetic data, registered as v1 of `test-model`."""
    lines: list[str] = []
    run = run_training(
        raw_dir,
        SMALL,
        tracking_uri=tracking_uri,
        experiment="test-exp",
        model_name="test-model",
        sweep_grid=[SMALL.als, dataclasses.replace(SMALL.als, factors=4)],
        tmp_root=tmp_path_factory.mktemp("tmp"),
        echo=lines.append,
    )
    return run, lines


def test_resolve_tracking_uri_precedence(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri() == DEFAULT_TRACKING_URI == "sqlite:///mlruns/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert resolve_tracking_uri() == "http://mlflow:5000"
    assert resolve_tracking_uri("sqlite:///x.db") == "sqlite:///x.db"


def test_data_snapshot_hash_order_invariant_and_content_sensitive():
    ratings = make_ratings()
    base = data_snapshot_hash(ratings)
    assert data_snapshot_hash(ratings.sample(frac=1, random_state=3)) == base
    changed = ratings.copy()
    changed.loc[0, "rating"] = 6 - changed.loc[0, "rating"]
    assert data_snapshot_hash(changed) != base
    assert len(base) == 16


CHAMPION = {"ndcg_at_10": 0.100, "coverage": 0.40}


@pytest.mark.parametrize(
    ("challenger", "champion", "promoted"),
    [
        ({"ndcg_at_10": 0.01, "coverage": 0.01}, None, True),  # no champion yet
        ({"ndcg_at_10": 0.110, "coverage": 0.40}, CHAMPION, True),  # better
        ({"ndcg_at_10": 0.0951, "coverage": 0.40}, CHAMPION, True),  # within tolerance
        ({"ndcg_at_10": 0.0949, "coverage": 0.40}, CHAMPION, False),  # worse than tolerance
        ({"ndcg_at_10": 0.120, "coverage": 0.361}, CHAMPION, True),  # coverage -9.75%
        ({"ndcg_at_10": 0.120, "coverage": 0.359}, CHAMPION, False),  # coverage -10.25%
    ],
)
def test_decide_promotion_rules(challenger, champion, promoted):
    decision, reason = decide_promotion(challenger, champion)
    assert decision is promoted and reason


def test_configure_mlflow_puts_local_artifacts_next_to_db(tracking_uri, first_run):
    experiment_id = configure_mlflow(tracking_uri, "test-exp")
    assert experiment_id == first_run[0].experiment_id
    location = mlflow.get_experiment(experiment_id).artifact_location
    assert location.startswith("file:") and location.rstrip("/").endswith("artifacts")


def test_training_run_logs_params_metrics_tags_artifacts(first_run):
    run, lines = first_run
    data = MlflowClient().get_run(run.run_id).data
    assert {"als.factors", "als.alpha", "ranker.learning_rate", "n_candidates"} <= set(data.params)
    for model in ("popularity", "als_only", "als_ranker"):
        for metric in ("ndcg_at_10", "recall_at_10", "coverage"):
            assert f"{model}_{metric}" in data.metrics
    assert data.metrics["ndcg_at_10"] == run.result.metrics[run.result.variant]["ndcg_at_10"]
    for tag in ("git_sha", "data_snapshot", "n_users", "n_ratings", "web_share", "variant"):
        assert tag in data.tags
    assert data.tags["run_type"] == "train" and data.tags["n_users"] == "60"
    artifacts = {a.path for a in MlflowClient().list_artifacts(run.run_id)}
    assert {"comparison.md", "comparison.json", "feature_importance.json"} <= artifacts
    assert "movie_features.parquet" in artifacts
    assert any("ndcg_at_10" in line and "als_ranker" in line for line in lines)


def test_sweep_runs_are_nested_children(first_run):
    run, _ = first_run
    children = mlflow.search_runs(
        experiment_ids=[run.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{run.run_id}'",
        output_format="list",
    )
    assert len(children) == 2
    assert all(c.data.tags["run_type"] == "sweep" for c in children)
    assert all("val_ndcg_at_10" in c.data.metrics for c in children)
    best = max(run.sweep, key=lambda cm: cm[1]["ndcg_at_10"])[0]
    assert run.config.als == best


def test_first_version_becomes_champion_and_loads(tracking_uri, first_run, raw_dir, tmp_path):
    run, _ = first_run
    assert run.registration is not None
    assert run.registration.version == "1" and run.registration.promoted
    champion = load_champion(tracking_uri, name="test-model", dst_dir=tmp_path)
    assert champion.version == "1" and champion.run_id == run.run_id
    users = [1, 2, 3]
    assert champion.recommender.recommend_known_users(
        users
    ) == run.result.recommender.recommend_known_users(users)
    recs = champion.recommender.recommend_new_user([(3, 5), (6, 4), (9, 5)], k=10)
    assert len(recs) == 10 and not {3, 6, 9} & set(recs)


def test_pyfunc_predict_schema(tracking_uri, first_run):
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(f"models:/test-model@{CHAMPION_ALIAS}")
    ratings = pd.DataFrame({"user_id": [7, 7, 8], "movie_id": [3, 6, 9], "rating": [5, 4, 5]})
    out = model.predict(ratings, params={"k": 5})
    assert list(out.columns) == RECOMMENDATION_COLUMNS
    assert out.groupby("user_id").size().to_dict() == {7: 5, 8: 5}
    assert not set(out.loc[out["user_id"] == 7, "movie_id"]) & {3, 6}


def _register_copy(run, metrics, snapshot, tmp_root):
    """Log the same recommender as a new model and register it with the given metrics."""
    with mlflow.start_run(experiment_id=run.experiment_id) as new_run:
        info = log_recommender(run.result.recommender, tmp_root=tmp_root)
    return register_and_promote(
        info.model_uri,
        new_run.info.run_id,
        metrics,
        snapshot,
        name="test-model",
        beats_popularity=True,
    )


def test_weaker_version_does_not_take_alias(tracking_uri, first_run, tmp_path):
    run, _ = first_run
    configure_mlflow(tracking_uri, "test-exp")
    served = run.result.metrics[run.result.variant]
    snapshot = MlflowClient().get_run(run.run_id).data.tags["data_snapshot"]
    weaker = {**served, "ndcg_at_10": served["ndcg_at_10"] - 0.05}
    outcome = _register_copy(run, weaker, snapshot, tmp_path)
    assert not outcome.promoted and outcome.previous_champion == "1"
    assert outcome.champion_metrics["ndcg_at_10"] == pytest.approx(served["ndcg_at_10"])
    client = MlflowClient()
    assert str(client.get_model_version_by_alias("test-model", CHAMPION_ALIAS).version) == "1"
    assert client.get_model_version("test-model", outcome.version).tags["promotion"] == "skipped"


def test_champion_from_other_snapshot_is_not_compared(tracking_uri, first_run, tmp_path):
    run, _ = first_run
    configure_mlflow(tracking_uri, "test-exp")
    better = {"ndcg_at_10": 0.99, "recall_at_10": 0.99, "coverage": 0.99}
    outcome = _register_copy(run, better, "other-snapshot", tmp_path)
    assert not outcome.promoted and "snapshot" in outcome.reason
    champion = MlflowClient().get_model_version_by_alias("test-model", CHAMPION_ALIAS)
    assert str(champion.version) == "1"


def test_cli_end_to_end_on_tiny_raw_dir(tracking_uri, raw_dir, capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # CLI temp files go under ./mlruns/tmp of the cwd
    run = main(
        [
            "--data-dir",
            str(raw_dir),
            "--tracking-uri",
            tracking_uri,
            "--experiment",
            "cli-exp",
            "--model-name",
            "cli-model",
            "--factors",
            "8",
            "--n-candidates",
            "30",
        ]
    )
    out = capsys.readouterr().out
    assert "popularity" in out and "als_only" in out and "als_ranker" in out
    assert "registered cli-model v1: promoted to champion" in out
    assert run.config.als.factors == 8 and run.config.n_candidates == 30


def test_data_snapshot_hash_independent_of_dtype():
    ratings = make_ratings()  # int32 ids, int8 rating, int64 timestamp
    base = data_snapshot_hash(ratings)
    wide = ratings.astype({"user_id": "int64", "movie_id": "int64", "rating": "int64"})
    floats = ratings.astype({"rating": "float64", "user_id": "float64"})
    assert data_snapshot_hash(wide) == base
    assert data_snapshot_hash(floats) == base


def test_load_champion_resolves_alias_once_and_loads_that_exact_version(
    tracking_uri, first_run, monkeypatch, tmp_path
):
    """Race guard: the alias may move between two lookups, so it must be read exactly once and
    the weights loaded by the resolved version number."""
    calls = []
    original = MlflowClient.get_model_version_by_alias

    def counting(self, name, alias):
        calls.append((name, alias))
        return original(self, name, alias)

    loaded = []
    original_download = mlflow.artifacts.download_artifacts

    def recording(artifact_uri=None, *args, **kwargs):
        loaded.append(artifact_uri)
        return original_download(artifact_uri, *args, **kwargs)

    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", counting)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", recording)
    champion = load_champion(tracking_uri, name="test-model", dst_dir=tmp_path)
    assert calls == [("test-model", CHAMPION_ALIAS)]
    expected = MlflowClient(tracking_uri).get_model_version_download_uri(
        "test-model", champion.version
    )
    assert not expected.startswith("models:/")  # the local store gives the files' location
    assert loaded == [expected]
    assert champion.model_uri == f"models:/test-model/{champion.version}"


@pytest.mark.parametrize("dtype", ["int8", "int32", "int64"])
def test_pyfunc_predict_accepts_integer_ratings(tracking_uri, first_run, dtype):
    """ML-1M ratings are ints (load_ratings gives int8, the API/Postgres plain ints)."""
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(f"models:/test-model@{CHAMPION_ALIAS}")
    ratings = pd.DataFrame(
        {"user_id": [7, 7, 8], "movie_id": [3, 6, 9], "rating": pd.array([5, 4, 5], dtype=dtype)}
    )
    out = model.predict(ratings, params={"k": 5})
    assert out.groupby("user_id").size().to_dict() == {7: 5, 8: 5}


def test_foldin_metric_logged_for_served_variant(first_run):
    run, _ = first_run
    data = MlflowClient().get_run(run.run_id).data
    key = f"{run.result.variant}_foldin_ndcg_at_10"
    assert key in data.metrics and 0.0 < data.metrics[key] <= 1.0


@pytest.fixture(scope="module")
def ranker_run(tracking_uri, raw_dir, tmp_path_factory):
    """Training run forced onto the als_ranker variant (on synthetic data the pipeline would
    pick als_only), so the MLflow path with a ranker bundle is covered."""
    patch = pytest.MonkeyPatch()
    patch.setattr("recsys.pipelines.train.select_variant", lambda metrics, k=10: "als_ranker")
    try:
        return run_training(
            raw_dir,
            SMALL,
            tracking_uri=tracking_uri,
            experiment="ranker-exp",
            model_name="ranker-model",
            tmp_root=tmp_path_factory.mktemp("tmp-ranker"),
            echo=lambda _: None,
        )
    finally:
        patch.undo()


def test_ranker_variant_logs_loads_and_predicts_with_int_ratings(
    tracking_uri, ranker_run, tmp_path
):
    assert ranker_run.result.variant == "als_ranker"
    champion = load_champion(tracking_uri, name="ranker-model", dst_dir=tmp_path)
    assert champion.recommender.variant == "als_ranker"
    assert champion.recommender.ranker is not None
    data = MlflowClient().get_run(ranker_run.run_id).data
    assert data.tags["variant"] == "als_ranker"
    assert "als_ranker_foldin_ndcg_at_10" in data.metrics
    model = mlflow.pyfunc.load_model(champion.model_uri)
    ratings = pd.DataFrame(
        {
            "user_id": [7, 7, 7, 8, 8],
            "movie_id": [3, 6, 9, 12, 15],
            "rating": pd.array([5, 4, 5, 4, 5], dtype="int64"),
        }
    )
    out = model.predict(ratings, params={"k": 5})
    assert list(out.columns) == RECOMMENDATION_COLUMNS
    assert out.groupby("user_id").size().to_dict() == {7: 5, 8: 5}
    expected = champion.recommender.recommend_for_ratings(ratings.astype("float64"), k=5)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))


@pytest.mark.parametrize("champion", [None, CHAMPION])
def test_decide_promotion_blocked_when_not_beating_popularity(champion):
    decision, reason = decide_promotion(
        {"ndcg_at_10": 0.5, "coverage": 0.9}, champion, beats_popularity=False
    )
    assert decision is False and "popularity" in reason


def test_register_and_promote_skips_model_that_loses_to_popularity(
    tracking_uri, first_run, tmp_path
):
    run, _ = first_run
    configure_mlflow(tracking_uri, "test-exp")
    with mlflow.start_run(experiment_id=run.experiment_id) as new_run:
        info = log_recommender(run.result.recommender, tmp_root=tmp_path)
    outcome = register_and_promote(
        info.model_uri,
        new_run.info.run_id,
        run.result.metrics[run.result.variant],
        "snap",
        name="gate-model",
        beats_popularity=False,
    )
    client = MlflowClient()
    assert not outcome.promoted and "popularity" in outcome.reason
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias("gate-model", CHAMPION_ALIAS)
    version = client.get_model_version("gate-model", outcome.version)
    assert version.tags["promotion"] == "skipped"
    assert "popularity" in version.tags["promotion_reason"]
    assert version.tags["beats_popularity"] == "false"
    run_tags = client.get_run(new_run.info.run_id).data.tags
    assert run_tags["promoted"] == "false" and "popularity" in run_tags["promotion_reason"]


def test_git_sha_env_override(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "abc1234")
    assert git_sha() == "abc1234"
    monkeypatch.delenv("GIT_SHA")
    assert git_sha() != "abc1234"


def test_load_champion_does_not_use_or_change_global_tracking_uri(
    tracking_uri, first_run, tmp_path
):
    """The API passes its URI explicitly; a process-wide MLflow URI must neither be needed nor
    be overwritten (the global points at an unreachable server here)."""
    previous = mlflow.get_tracking_uri()
    unreachable = "http://127.0.0.1:9"
    mlflow.set_tracking_uri(unreachable)
    try:
        champion = load_champion(tracking_uri, name="test-model", dst_dir=tmp_path)
        assert mlflow.get_tracking_uri() == unreachable
    finally:
        mlflow.set_tracking_uri(previous)
    assert champion.recommender.recommend_new_user([(3, 5), (6, 4)], k=3)


def test_load_champion_downloads_into_dst_dir_and_cleans_up(
    tracking_uri, first_run, tmp_path, monkeypatch
):
    destinations = []
    original_download = mlflow.artifacts.download_artifacts

    def recording(*args, **kwargs):
        destinations.append(Path(kwargs["dst_path"]))
        return original_download(*args, **kwargs)

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", recording)
    dst = tmp_path / "models"
    champion = load_champion(tracking_uri, name="test-model", dst_dir=dst)
    assert len(destinations) == 1 and destinations[0].parent == dst
    assert not destinations[0].exists()  # weights are in memory, the download is removed
    assert list(dst.iterdir()) == []
    assert len(champion.recommender.recommend_new_user([(3, 5), (6, 4)], k=5)) == 5


def test_champion_version_resolves_alias_or_none(tracking_uri, first_run):
    assert champion_version(tracking_uri, name="test-model") == "1"
    assert champion_version(tracking_uri, name="no-such-model") is None


def test_register_and_promote_requires_explicit_beats_popularity():
    param = inspect.signature(register_and_promote).parameters["beats_popularity"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


def test_pip_requirements_pin_mlflow_skinny():
    requirements = pip_requirements()
    assert f"mlflow-skinny=={package_version('mlflow-skinny')}" in requirements
    assert not any(r.startswith("mlflow==") for r in requirements)


def test_cli_tmp_dir_option():
    assert parse_args(["--data-dir", "d"]).tmp_dir == Path("mlruns") / "tmp"
    assert parse_args(["--data-dir", "d", "--tmp-dir", "/tmp/recsys"]).tmp_dir == Path(
        "/tmp/recsys"
    )
