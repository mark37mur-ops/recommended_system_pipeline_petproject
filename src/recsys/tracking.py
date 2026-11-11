"""MLflow tracking and Model Registry helpers (spec 4.3).

Backend: `MLFLOW_TRACKING_URI` if set (stages 3-4 point it at the MLflow server), otherwise a
local SQLite store `sqlite:///mlruns/mlflow.db` with artifacts in `mlruns/artifacts`. The
registry uses aliases (`champion`), never the deprecated stages.
"""

import hashlib
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from recsys.models.two_stage import TwoStageRecommender

DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
EXPERIMENT_NAME = "movielens-recsys"
MODEL_NAME = "movielens-recommender"
CHAMPION_ALIAS = "champion"
NDCG_TOLERANCE = 0.005
MAX_COVERAGE_DROP = 0.10
# Metrics are means over thousands of users: 1e-9 is far below any real difference.
_EPS = 1e-9
# Where load_champion downloads model artifacts unless told otherwise (never the system temp
# dir, which is on the small C: drive locally). The download is removed once loaded.
DEFAULT_MODEL_DIR = Path("mlruns") / "tmp" / "champion"


def resolve_tracking_uri(uri: str | None = None) -> str:
    """Explicit argument > $MLFLOW_TRACKING_URI > local SQLite default."""
    return uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def _sqlite_path(uri: str) -> Path | None:
    prefix = "sqlite:///"
    return Path(uri[len(prefix) :]) if uri.startswith(prefix) else None


def configure_mlflow(uri: str | None = None, experiment: str = EXPERIMENT_NAME) -> str:
    """Point MLflow at the tracking store and make sure the experiment exists.

    For a local SQLite store the parent directory is created and the experiment's artifacts
    go to `<db dir>/artifacts`; for a server the server decides. Returns the experiment id.
    """
    uri = resolve_tracking_uri(uri)
    artifact_location = None
    db = _sqlite_path(uri)
    if db is not None:
        db.parent.mkdir(parents=True, exist_ok=True)
        artifact_location = (db.parent / "artifacts").resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    existing = mlflow.get_experiment_by_name(experiment)
    if existing is not None:
        return existing.experiment_id
    return mlflow.create_experiment(experiment, artifact_location=artifact_location)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_sha() -> str:
    """`$GIT_SHA` (set when building Docker images, which have no .git) > git HEAD > unknown."""
    return os.environ.get("GIT_SHA") or _git("rev-parse", "HEAD") or "unknown"


def git_dirty() -> bool:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return bool(status)


def data_snapshot_hash(ratings: pd.DataFrame) -> str:
    """Content hash of the ratings table, independent of row order and column dtypes."""
    cols = ["user_id", "movie_id", "rating", "timestamp"]
    canonical = (
        ratings[cols]
        .astype("int64")  # int8 / int32 / integral floats all hash alike
        .sort_values(cols, kind="stable")
        .reset_index(drop=True)
    )
    hashed = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def decide_promotion(
    challenger: Mapping[str, float],
    champion: Mapping[str, float] | None,
    ndcg_tolerance: float = NDCG_TOLERANCE,
    max_coverage_drop: float = MAX_COVERAGE_DROP,
    k: int = 10,
    beats_popularity: bool = True,
) -> tuple[bool, str]:
    """Spec 4.2: promote if there is no champion, or NDCG@k(challenger) >= NDCG@k(champion) -
    tolerance and coverage dropped by at most `max_coverage_drop` (relative).

    Spec 2.5 gate first: a model that does not beat the popularity baseline is never promoted,
    whatever the champion situation."""
    if not beats_popularity:
        return False, "does not beat the popularity baseline (spec 2.5)"
    if champion is None:
        return True, "no current champion"
    key = f"ndcg_at_{k}"
    # Both bounds are inclusive ("no more than"); _EPS keeps an exact boundary on the right
    # side of float rounding (0.2 * (1 - 0.10) = 0.18000000000000002 > 0.18).
    floor = champion[key] - ndcg_tolerance
    if challenger[key] < floor - _EPS:
        return False, f"{key} {challenger[key]:.4f} < champion {champion[key]:.4f} - tolerance"
    min_coverage = champion["coverage"] * (1 - max_coverage_drop)
    if challenger["coverage"] < min_coverage - _EPS:
        return False, (
            f"coverage {challenger['coverage']:.4f} dropped more than "
            f"{max_coverage_drop:.0%} from champion {champion['coverage']:.4f}"
        )
    return True, f"{key} {challenger[key]:.4f} vs champion {champion[key]:.4f}: within tolerance"


@dataclass(frozen=True)
class RegistrationOutcome:
    version: str
    promoted: bool
    reason: str
    previous_champion: str | None
    champion_metrics: dict[str, float] | None


# Error codes the registry uses for "no such model" / "no such alias". Anything else (server
# unreachable, internal error) must propagate: treating it as "no champion" would make
# register_and_promote promote without comparing.
_NOT_FOUND_CODES = {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}


def _current_champion(client: MlflowClient, name: str):
    try:
        return client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except MlflowException as exc:
        if exc.error_code in _NOT_FOUND_CODES:
            return None
        raise


def register_and_promote(
    model_uri: str,
    run_id: str,
    metrics: Mapping[str, float],
    snapshot: str,
    tags: Mapping[str, str] | None = None,
    name: str = MODEL_NAME,
    k: int = 10,
    *,
    beats_popularity: bool,
) -> RegistrationOutcome:
    """Register `model_uri` as a new version of `name` and move `champion` if it deserves it.

    `metrics` are the served variant's test metrics. The current champion is compared only if
    it was evaluated on the same data snapshot (same deterministic split); otherwise the alias
    is left alone and re-evaluating the champion on the new split is left to the retraining
    DAG (stage 4). Champion metrics are also logged into `run_id` as champion_*.
    `beats_popularity=False` (spec 2.5 gate) registers the version but never promotes it.
    """
    client = MlflowClient()
    champion = _current_champion(client, name)
    version = mlflow.register_model(model_uri, name).version
    version_tags = {f"metric.{key}": f"{value:.6f}" for key, value in metrics.items()}
    version_tags |= {"data_snapshot": snapshot, **(tags or {})}
    version_tags["beats_popularity"] = str(beats_popularity).lower()
    for key, value in version_tags.items():
        client.set_model_version_tag(name, version, key, value)

    champion_metrics = None
    if not beats_popularity:
        promoted, reason = decide_promotion(metrics, None, k=k, beats_popularity=False)
    elif champion is None:
        promoted, reason = decide_promotion(metrics, None, k=k)
    elif champion.tags.get("data_snapshot") != snapshot:
        promoted = False
        reason = (
            f"champion v{champion.version} was evaluated on data snapshot "
            f"{champion.tags.get('data_snapshot')}, not {snapshot}; re-evaluation needed"
        )
    else:
        champion_metrics = {
            key.removeprefix("metric."): float(value)
            for key, value in champion.tags.items()
            if key.startswith("metric.")
        }
        promoted, reason = decide_promotion(metrics, champion_metrics, k=k)
        for key, value in champion_metrics.items():
            client.log_metric(run_id, f"champion_{key}", value)

    if promoted:
        client.set_registered_model_alias(name, CHAMPION_ALIAS, version)
    client.set_model_version_tag(name, version, "promotion", "promoted" if promoted else "skipped")
    client.set_model_version_tag(name, version, "promotion_reason", reason)
    client.set_tag(run_id, "registered_version", version)
    client.set_tag(run_id, "promoted", str(promoted).lower())
    client.set_tag(run_id, "promotion_reason", reason)
    return RegistrationOutcome(
        version=str(version),
        promoted=promoted,
        reason=reason,
        previous_champion=None if champion is None else str(champion.version),
        champion_metrics=champion_metrics,
    )


# ----- building blocks of the retraining DAG (stage 4) --------------------------------------


class ChampionMoved(RuntimeError):
    """The `champion` alias changed between evaluation and promotion."""


def register_version(
    model_uri: str, run_id: str, tags: Mapping[str, str], name: str = MODEL_NAME
) -> str:
    """Register `model_uri` as a new version of `name`, at most once per MLflow run.

    The registry itself is the source of truth: a version of `name` that already belongs to
    `run_id` is reused. A retry after a crash anywhere in here (after register_model, between
    two tags, before the run's `registered_version` tag) therefore never adds a second version,
    and it writes every tag again, so the retry finishes what the first attempt started."""
    client = MlflowClient()
    existing = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
    if existing:
        version = str(max(int(v.version) for v in existing))
    else:
        version = str(mlflow.register_model(model_uri, name).version)
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    client.set_tag(run_id, "registered_version", version)
    return version


def set_champion(name: str, version: str, expected_previous: str | None) -> str | None:
    """Point `champion` at `version` unless someone else moved it since it was evaluated.

    `expected_previous` is the champion version the decision was based on (None = there was
    none). Idempotent: if the alias already points at `version`, nothing happens. Returns the
    version the alias pointed to before."""
    client = MlflowClient()
    current = _current_champion(client, name)
    current_version = None if current is None else str(current.version)
    if current_version == str(version):
        return expected_previous
    if current_version != expected_previous:
        raise ChampionMoved(
            f"champion of {name} moved from v{expected_previous} to v{current_version} after "
            f"evaluation; re-run the evaluation instead of overwriting it"
        )
    client.set_registered_model_alias(name, CHAMPION_ALIAS, str(version))
    return current_version


@dataclass(frozen=True)
class ChampionModel:
    recommender: TwoStageRecommender
    version: str
    run_id: str | None
    model_uri: str


def _client(tracking_uri: str | None) -> tuple[MlflowClient, str]:
    uri = resolve_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=uri, registry_uri=uri), uri


def _storage_location(client: MlflowClient, name: str, version: str) -> str:
    """Where the version's files live (file://, mlflow-artifacts:/ ...), resolved with `client`.

    Downloading `models:/name/version` directly would let MLflow resolve the registry through
    the process-wide registry URI (mlflow 3.18 `ModelsArtifactRepository._get_model_uri_infos`
    ignores the URIs passed to `download_artifacts`), so the indirection is done here."""
    location = client.get_model_version_download_uri(name, version)
    if location.startswith("models:/"):  # MLflow 3: the version points at a logged model
        model_id = location.removeprefix("models:/").split("/", 1)[0]
        location = client.get_logged_model(model_id).artifact_location
    return location


def champion_version(tracking_uri: str | None = None, name: str = MODEL_NAME) -> str | None:
    """Version number the `champion` alias points to, or None if there is no such alias/model.

    Cheap (one registry call): lets the API skip a reload when the champion did not change."""
    client, _ = _client(tracking_uri)
    champion = _current_champion(client, name)
    return None if champion is None else str(champion.version)


def load_champion(
    tracking_uri: str | None = None, name: str = MODEL_NAME, dst_dir: Path | None = None
) -> ChampionModel:
    """Load the `champion` version of the registered model (what the API serves).

    The alias is resolved exactly once and the weights are loaded by that version number, so a
    promotion happening meanwhile cannot make `version` and the loaded model disagree.
    Artifacts are downloaded into a fresh sub-directory of `dst_dir` (default
    DEFAULT_MODEL_DIR), read into memory and the sub-directory is deleted again. The tracking
    and registry URIs are passed explicitly: process-wide MLflow state is neither read nor
    changed, so this is safe to call from a background thread of a running service.
    """
    client, uri = _client(tracking_uri)
    model_version = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    model_uri = f"models:/{name}/{model_version.version}"
    root = Path(dst_dir) if dst_dir is not None else DEFAULT_MODEL_DIR
    target = root / f"v{model_version.version}-{uuid.uuid4().hex[:8]}"
    try:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=_storage_location(client, name, str(model_version.version)),
            dst_path=str(target),
            tracking_uri=uri,
            registry_uri=uri,
        )
        pyfunc_model = mlflow.pyfunc.load_model(local)
        recommender = pyfunc_model.unwrap_python_model().recommender
    finally:
        shutil.rmtree(target, ignore_errors=True)
    return ChampionModel(
        recommender=recommender,
        version=str(model_version.version),
        run_id=model_version.run_id,
        model_uri=model_uri,
    )
