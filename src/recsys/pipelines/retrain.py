"""Tasks of the `retrain_daily` DAG (spec 4.2), runnable without Airflow.

    extract_ratings -> validate_data -> drift_report -> split_data -> train_als -> train_ranker
      -> evaluate -> decide_promotion -> promote (register_challenger) -> set_champion_alias
                                          -> reload_api (recsys.pipelines.reload)
                                      -> skip -> log_reason

Every task takes the run folder (`/data/runs/{run_id}`, see `run_dir.py`) and returns it, so
the only thing passing through XCom is that path (and the branch id from decide_promotion).
Inputs and outputs are files in the folder, written atomically, and every task can be re-run
for the same run with the same result (spec 4.1):
- extract_ratings freezes the snapshot: a complete snapshot already in the folder is reused;
- training is seeded and deterministic, so retries rewrite identical files;
- one MLflow run per pipeline run (its id is kept in the folder), reused by every task;
- registration happens at most once per MLflow run (looked up in the registry), the alias
  move is a no-op if already done, and once a run has promoted, evaluate / decide_promotion
  keep that outcome instead of comparing the model with itself.

Run options (DAG params, all booleans; strings like "false" / "0" are accepted too):
- include_web (default true): train on web ratings too. A run with include_web=false is
  evaluation-only: it is evaluated and logged, but never promoted...
- allow_promotion (default false): ...unless this is set. Meant for the incident "web data
  poisoned", where a MovieLens-only model must replace the champion; the spec 4.2 rule still
  applies. No effect on include_web=true runs.
- allow_count_drop (default false): accept a snapshot with >5% fewer ratings than the
  previous run, or with all web ratings gone (after a deliberate clean-up). Without it, every
  later run would fail too: failed runs are not recorded, so the baseline never moves.

CLI (the whole chain in one process, e.g. to debug without Airflow):
    python -m recsys.pipelines.retrain --run-id manual-1 [--no-web] [--no-reload]
        [--allow-promotion] [--allow-count-drop]
"""

import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from recsys.data.split import Split, temporal_split
from recsys.data.validate import DataValidationError, validate_ratings
from recsys.db.extract import extract_snapshot
from recsys.db.monitoring import (
    delete_old_impressions,
    recent_impressions,
    record_drift,
    record_run_metrics,
)
from recsys.db.run_state import RunRecord, create_run_state, previous_run, record_run
from recsys.db.schema import make_engine
from recsys.drift import CURRENT_WINDOW, build_drift_sections, summarize_ratings, summary_report
from recsys.evaluation import format_comparison, make_test_context
from recsys.models.als import ALSCandidateGenerator
from recsys.models.pyfunc import log_recommender
from recsys.models.ranker import LambdaRanker
from recsys.pipelines.run_dir import RunDir, atomic_dir, read_json, write_json_atomic
from recsys.pipelines.settings import PipelineSettings, prepare
from recsys.pipelines.train import (
    evaluate_variants,
    fit_history_als,
    flatten_config,
    foldin_metrics,
    run_metrics,
)
from recsys.pipelines.train import train_ranker as fit_ranker
from recsys.tracking import (
    champion_version,
    configure_mlflow,
    data_snapshot_hash,
    git_dirty,
    git_sha,
    load_champion,
    register_version,
    set_champion,
)
from recsys.tracking import decide_promotion as promotion_rule

log = logging.getLogger("recsys.retrain")

PROMOTE_TASK = "promote"
SKIP_TASK = "skip"
SPLIT_PARTS = ("als_train", "ranker_train", "test")
EVALUATION_ONLY = (
    "include_web=false runs are evaluation-only: the model was measured without the web "
    "visitors it would serve; trigger with allow_promotion=true to let the promotion rule "
    "decide (incident: web data poisoned)"
)

_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})


def parse_flag(value: object, name: str) -> bool:
    """A run option as a bool. Airflow hands over a bool when templates render natively, but a
    string ("False") otherwise, and "False" is truthy: bool() would silently flip the flag."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise ValueError(f"{name} must be a boolean (true/false), got {value!r}")


# ----- MLflow: one run per pipeline run ------------------------------------------------------


def _mlflow_run(rd: RunDir, settings: PipelineSettings) -> str:
    """The MLflow run of this pipeline run, created by the first task that logs."""
    experiment_id = configure_mlflow(settings.mlflow_uri(), settings.experiment)
    if rd.mlflow_json.exists():
        return read_json(rd.mlflow_json)["run_id"]
    airflow_run_id = read_json(rd.extract_json)["run_id"]
    run = MlflowClient().create_run(
        experiment_id,
        run_name=f"retrain {airflow_run_id}",
        tags={"run_type": "retrain", "airflow_run_id": airflow_run_id},
    )
    write_json_atomic(rd.mlflow_json, {"run_id": run.info.run_id, "experiment_id": experiment_id})
    return run.info.run_id


def _set_run_tags(rd: RunDir, settings: PipelineSettings, tags: dict[str, str]) -> None:
    run_id = _mlflow_run(rd, settings)
    client = MlflowClient()
    for key, value in tags.items():
        client.set_tag(run_id, key, value)


# ----- data ----------------------------------------------------------------------------------


def write_snapshot(
    rd: RunDir,
    run_id: str,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
    include_web: bool,
) -> RunDir:
    """Write the extract folder atomically; `extract.json` inside marks it complete."""
    with atomic_dir(rd.extract_dir) as tmp:
        ratings.to_parquet(tmp / rd.ratings_path.name, index=False)
        movies.to_parquet(tmp / rd.movies_path.name, index=False)
        users.to_parquet(tmp / rd.users_path.name, index=False)
        write_json_atomic(
            tmp / rd.extract_json.name,
            {
                "run_id": run_id,
                "include_web": bool(include_web),
                "n_ratings": int(len(ratings)),
                "extracted_at": datetime.now(UTC).isoformat(),
            },
        )
    return rd


def _snapshot(rd: RunDir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(rd.ratings_path),
        pd.read_parquet(rd.movies_path),
        pd.read_parquet(rd.users_path),
    )


def _split(rd: RunDir) -> Split:
    return Split(*(pd.read_parquet(rd.split_path(part)) for part in SPLIT_PARTS))


def extract_ratings(
    run_id: str, settings: PipelineSettings | None = None, include_web: bool | str = True
) -> str:
    """Ratings (MovieLens + web, or MovieLens only), movies and users from Postgres -> Parquet.

    The first complete snapshot of a run is frozen (Decision 6): the ratings table changes all
    the time, and a retry must train on the same data as the first attempt. A folder holding
    another run's snapshot (run ids that map to the same folder name) is refused."""
    settings = prepare(settings)
    include_web = parse_flag(include_web, "include_web")
    rd = RunDir.for_run(run_id, settings.runs_root)
    if rd.extract_json.exists():
        meta = read_json(rd.extract_json)
        if meta["run_id"] != run_id or meta["include_web"] != include_web:
            raise RuntimeError(
                f"{rd.root} holds the snapshot of run {meta['run_id']!r} (include_web="
                f"{meta['include_web']}), not of {run_id!r} (include_web={include_web})"
            )
        log.info("reusing the frozen snapshot of %s: %s", run_id, meta)
        return str(rd.root)
    engine = make_engine(settings.db_url())
    try:
        ratings, movies, users = extract_snapshot(engine, include_web=include_web)
    finally:
        engine.dispose()
    write_snapshot(rd, run_id, ratings, movies, users, include_web)
    log.info(
        "extracted %d ratings (%d web), %d movies, %d users -> %s",
        len(ratings),
        int((ratings["source"] == "web").sum()),
        len(movies),
        len(users),
        rd.extract_dir,
    )
    return str(rd.root)


def validate_data(
    run_dir: str, settings: PipelineSettings | None = None, allow_count_drop: bool | str = False
) -> str:
    """Spec 4.2: no NULLs, ratings 1..5, no duplicate (user_id, movie_id), not more than 5%
    fewer ratings than the previous run with the same `include_web`; and (include_web runs)
    not all web ratings gone when the previous run had some. Any violation fails the task with
    the full list of problems; only a valid snapshot is recorded in retrain_runs.

    `allow_count_drop` turns the two count rules off for this run (after a deliberate
    clean-up); the run is then recorded and becomes the new baseline."""
    settings = prepare(settings)
    allow_count_drop = parse_flag(allow_count_drop, "allow_count_drop")
    rd = RunDir(Path(run_dir))
    meta = read_json(rd.extract_json)
    ratings = pd.read_parquet(rd.ratings_path)
    engine = make_engine(settings.db_url())
    try:
        create_run_state(engine)
        previous = previous_run(engine, meta["run_id"], meta["include_web"])
        check_counts = previous is not None and not allow_count_drop
        if previous is not None and allow_count_drop:
            log.warning(
                "allow_count_drop: count rules off (previous run %s: %d ratings, %s web)",
                previous.run_id,
                previous.n_ratings,
                previous.summary.get("n_web_ratings"),
            )
        problems: list[str] = []
        try:
            validate_ratings(
                ratings,
                previous_count=previous.n_ratings if check_counts else None,
                previous_web_count=(
                    previous.summary.get("n_web_ratings")
                    if check_counts and meta["include_web"]
                    else None
                ),
            )
        except DataValidationError as exc:
            problems = list(exc.problems)
        if ratings.empty:
            problems.append("snapshot has no ratings (was bootstrap_data run?)")
        if problems:
            for problem in problems:
                log.error("validation failed: %s", problem)
            raise DataValidationError(problems)

        summary = summarize_ratings(ratings)
        snapshot = data_snapshot_hash(ratings)
        record_run(
            engine,
            RunRecord(
                run_id=meta["run_id"],
                include_web=meta["include_web"],
                n_ratings=len(ratings),
                data_snapshot=snapshot,
                summary=summary,
            ),
        )
    finally:
        engine.dispose()
    write_json_atomic(
        rd.validation_json,
        {
            "data_snapshot": snapshot,
            "summary": summary,
            "previous_run_id": None if previous is None else previous.run_id,
            "previous_n_ratings": None if previous is None else previous.n_ratings,
            "allow_count_drop": allow_count_drop,
        },
    )
    _set_run_tags(
        rd,
        settings,
        {
            "git_sha": git_sha(),
            "git_dirty": str(git_dirty()).lower(),
            "data_snapshot": snapshot,
            "n_users": str(summary["n_users"]),
            "n_ratings": str(summary["n_ratings"]),
            "n_web_ratings": str(summary["n_web_ratings"]),
            "web_share": f"{summary['web_share']:.6f}",
            "include_web": str(meta["include_web"]).lower(),
            "allow_count_drop": str(allow_count_drop).lower(),
        },
    )
    log.info(
        "snapshot %s valid: %d ratings, web share %.4f, previous run %s (%s ratings)",
        snapshot,
        summary["n_ratings"],
        summary["web_share"],
        None if previous is None else previous.run_id,
        None if previous is None else previous.n_ratings,
    )
    return run_dir


def drift_report(run_dir: str, settings: PipelineSettings | None = None) -> str:
    """Spec 5.3: Evidently reports (MovieLens vs web ratings of the last 7 days; what MovieLens
    users rated vs what the model recommended) + the stage-4 summary against the previous run.

    Outputs: run_dir/drift.json and run_dir/drift/*.html; MLflow artifacts drift/, metrics
    drift_share / prediction_drift_share (only when computed) and drift_*; one drift_metrics
    row in Postgres for Grafana. Never fails on drift or on missing data: too few web ratings
    or impressions give status insufficient_data and a NULL share (recsys.drift)."""
    settings = prepare(settings)
    rd = RunDir(Path(run_dir))
    meta = read_json(rd.extract_json)
    current = read_json(rd.validation_json)["summary"]
    # The 7-day window ends at the snapshot, not at "now": a retry computes the same report.
    until = datetime.fromisoformat(meta["extracted_at"])
    ratings, movies, _ = _snapshot(rd)
    engine = make_engine(settings.db_url())
    try:
        previous = previous_run(engine, meta["run_id"], meta["include_web"])
        shown = recent_impressions(engine, since=until - CURRENT_WINDOW, until=until)
        sections = build_drift_sections(ratings, movies, shown, until=until, out_dir=rd.drift_dir)
        summary = summary_report(current, None if previous is None else previous.summary)
        rating_part, rec_part = sections["ratings"], sections["recommendations"]
        record_drift(
            engine,
            {
                "run_id": meta["run_id"],
                "include_web": bool(meta["include_web"]),
                "drift_share": sections["drift_share"],
                "prediction_drift_share": sections["prediction_drift_share"],
                "ratings_status": rating_part["status"],
                "recommendations_status": rec_part["status"],
                "n_reference": rating_part["n_reference"],
                "n_current": rating_part["n_current"],
                "n_current_recommendations": rec_part["n_current"],
                "details": {
                    "window_start": sections["window_start"],
                    "window_end": sections["window_end"],
                    "ratings": rating_part["columns"],
                    "recommendations": rec_part["columns"],
                },
            },
        )
    finally:
        engine.dispose()

    metrics = dict(summary["metrics"])
    for key in ("drift_share", "prediction_drift_share"):
        if sections[key] is not None:  # MLflow metrics cannot be NULL: absent = not computed
            metrics[key] = float(sections[key])
    report = {"engine": sections["engine"], "summary": summary, "metrics": metrics} | {
        key: sections[key] for key in ("window_start", "window_end", "ratings", "recommendations")
    }
    write_json_atomic(rd.drift_json, report)
    run_id = _mlflow_run(rd, settings)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(report, "drift/drift.json")
        mlflow.log_artifacts(str(rd.drift_dir), "drift")
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "drift_ratings_status": rating_part["status"],
                "drift_recommendations_status": rec_part["status"],
            }
        )
    log.info(
        "drift: ratings %s (share %s, %d current rows), recommendations %s (share %s, %d "
        "impressions), window %s .. %s",
        rating_part["status"],
        sections["drift_share"],
        rating_part["n_current"],
        rec_part["status"],
        sections["prediction_drift_share"],
        rec_part["n_current"],
        sections["window_start"],
        sections["window_end"],
    )
    for key, value in sorted(metrics.items()):
        log.info("drift %s = %.4f", key, value)
    return run_dir


# ----- training ------------------------------------------------------------------------------


def split_data(run_dir: str, settings: PipelineSettings | None = None) -> str:
    """Per-user temporal split (spec 2.1) -> split/{als_train,ranker_train,test}.parquet."""
    prepare(settings)
    rd = RunDir(Path(run_dir))
    split = temporal_split(pd.read_parquet(rd.ratings_path))
    with atomic_dir(rd.split_dir) as tmp:
        for part in SPLIT_PARTS:
            getattr(split, part).to_parquet(tmp / rd.split_path(part).name, index=False)
    log.info(
        "split: als_train %d, ranker_train %d, test %d ratings",
        len(split.als_train),
        len(split.ranker_train),
        len(split.test),
    )
    return run_dir


def train_als(run_dir: str, settings: PipelineSettings | None = None) -> str:
    """The served ALS: fit on the history (als_train + ranker_train)."""
    settings = prepare(settings)
    rd = RunDir(Path(run_dir))
    config = settings.train_config
    als = fit_history_als(_split(rd), config)
    with atomic_dir(rd.als_dir) as tmp:
        als.save(tmp)
    run_id = _mlflow_run(rd, settings)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(flatten_config(config))
    log.info("ALS %s fit on %d ratings -> %s", config.als, als.n_train_ratings, rd.als_dir)
    return run_dir


def train_ranker(run_dir: str, settings: PipelineSettings | None = None) -> str:
    """LightGBM ranker on candidates of an ALS fit on als_train only, labels from
    ranker_train (spec 2.3: the ranker must not learn a score that already saw its labels)."""
    settings = prepare(settings)
    rd = RunDir(Path(run_dir))
    _, movies, users = _snapshot(rd)
    ranker, diagnostics = fit_ranker(_split(rd), movies, users, settings.train_config)
    with atomic_dir(rd.ranker_dir) as tmp:
        ranker.save(tmp)
        write_json_atomic(tmp / rd.ranker_json.name, diagnostics)
    run_id = _mlflow_run(rd, settings)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(diagnostics)
    log.info("ranker trained: %s", diagnostics)
    return run_dir


def _load_champion(settings: PipelineSettings, rd: RunDir):
    """The current champion, or None if the model or the alias does not exist (any other
    registry error propagates: an unreachable registry is not 'no champion')."""
    if champion_version(settings.mlflow_uri(), settings.model_name) is None:
        return None
    return load_champion(
        settings.mlflow_uri(), settings.model_name, dst_dir=rd.scratch_dir / "champion"
    )


def evaluate(run_dir: str, settings: PipelineSettings | None = None) -> str:
    """Challenger, champion and baselines on the SAME test split of this run (spec 4.2).

    popularity / als_only / als_ranker: the standard comparison (fitted factors), which also
    picks the served variant (spec 2.5). challenger_foldin / champion_foldin: both models get
    the test users' history through fold-in, i.e. the serving path, which is the only fair
    way to score the champion (its ALS never saw this snapshot). The decision uses these."""
    settings = prepare(settings)
    rd = RunDir(Path(run_dir))
    done = _promoted(rd)
    if done is not None:
        # The champion is now this run's own model: a fresh evaluation would compare it with
        # itself, log a second copy of it and rewrite what the promotion was based on.
        log.warning("this run already promoted v%s; keeping its evaluation", done["version"])
        return run_dir
    config = settings.train_config
    k = config.k
    split = _split(rd)
    _, movies, users = _snapshot(rd)
    ranker = LambdaRanker.load(rd.ranker_dir)
    als = ALSCandidateGenerator.load(rd.als_dir)
    result = evaluate_variants(
        split, movies, users, ranker, als, config, diagnostics=read_json(rd.ranker_json)
    )
    ctx = make_test_context(split, n_items=len(movies), k=k)
    challenger = foldin_metrics(result.recommender, ctx, users)
    champion = _load_champion(settings, rd)
    champion_metrics = (
        None if champion is None else foldin_metrics(champion.recommender, ctx, users)
    )
    beats_popularity = (
        result.metrics["als_only"][f"ndcg_at_{k}"] > result.metrics["popularity"][f"ndcg_at_{k}"]
    )

    with atomic_dir(rd.model_dir) as tmp:
        result.recommender.save(tmp)
    comparison = dict(result.metrics) | {"challenger_foldin": challenger}
    if champion_metrics is not None:
        comparison["champion_foldin"] = champion_metrics
    table = format_comparison(comparison, k)

    run_id = _mlflow_run(rd, settings)
    metrics = run_metrics(result, k)
    metrics |= {f"challenger_foldin_{key}": v for key, v in challenger.items()}
    metrics |= {f"champion_foldin_{key}": v for key, v in (champion_metrics or {}).items()}
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "variant": result.variant,
                "beats_popularity": str(beats_popularity).lower(),
                "champion_version": "none" if champion is None else champion.version,
            }
        )
        mlflow.log_text(table + "\n", "comparison.md")
        mlflow.log_dict(comparison, "comparison.json")
        mlflow.log_dict(result.feature_importance, "feature_importance.json")
        features_path = rd.scratch_dir / "movie_features.parquet"
        features_path.parent.mkdir(parents=True, exist_ok=True)
        result.recommender.movie_feats.to_parquet(features_path)
        mlflow.log_artifact(str(features_path))
        model_info = log_recommender(result.recommender, tmp_root=rd.scratch_dir)

    # The same numbers for Grafana's "NDCG@10 of the last runs" (stage-5 plan, Decision 7).
    engine = make_engine(settings.db_url())
    try:
        record_run_metrics(engine, read_json(rd.extract_json)["run_id"], metrics)
    finally:
        engine.dispose()
    write_json_atomic(
        rd.evaluate_json,
        {
            "k": k,
            "metrics": result.metrics,
            "variant": result.variant,
            "beats_popularity": beats_popularity,
            "challenger": challenger,
            "champion": champion_metrics,
            "champion_version": None if champion is None else champion.version,
            "model_uri": model_info.model_uri,
        },
    )
    log.info("evaluation on the test split of this run:\n%s", table)
    return run_dir


# ----- promotion -----------------------------------------------------------------------------


def _promoted(rd: RunDir) -> dict | None:
    """promotion.json if this run has already moved the alias, else None."""
    if not rd.promotion_json.exists():
        return None
    promotion = read_json(rd.promotion_json)
    return promotion if promotion.get("promoted") else None


def decide_promotion(
    run_dir: str, settings: PipelineSettings | None = None, allow_promotion: bool | str = False
) -> str:
    """Spec 4.2: promote if there is no champion, or NDCG@10 >= champion - 0.005 and coverage
    fell by at most 10% (both through fold-in on this run's split). Returns the id of the
    branch task to follow: PROMOTE_TASK or SKIP_TASK.

    Before the rule: a run with include_web=false is evaluation-only (skip) unless
    `allow_promotion`; a run that has already promoted reports that again (PROMOTE_TASK, the
    downstream tasks are no-ops) instead of comparing its model with itself."""
    settings = prepare(settings)
    allow_promotion = parse_flag(allow_promotion, "allow_promotion")
    rd = RunDir(Path(run_dir))
    ev = read_json(rd.evaluate_json)
    include_web = bool(read_json(rd.extract_json)["include_web"])
    done = _promoted(rd)
    if done is not None:
        promote, reason = True, f"already promoted v{done['version']} in this run"
        first_reason = read_json(rd.decision_json)["reason"] if rd.decision_json.exists() else ""
        if first_reason.startswith("already promoted"):  # third and later re-runs
            reason = first_reason
        elif first_reason:
            reason += f" ({first_reason})"
    elif not include_web and not allow_promotion:
        promote, reason = False, EVALUATION_ONLY
    else:
        promote, reason = promotion_rule(
            ev["challenger"], ev["champion"], k=ev["k"], beats_popularity=ev["beats_popularity"]
        )
    write_json_atomic(
        rd.decision_json,
        {
            "promote": promote,
            "reason": reason,
            "include_web": include_web,
            "allow_promotion": allow_promotion,
            "champion_version": ev["champion_version"],
            "challenger": ev["challenger"],
            "champion": ev["champion"],
        },
    )
    _set_run_tags(
        rd,
        settings,
        {
            "decision": PROMOTE_TASK if promote else SKIP_TASK,
            "promotion_reason": reason,
            "allow_promotion": str(allow_promotion).lower(),
        },
    )
    log.info("decision: %s (%s)", PROMOTE_TASK if promote else SKIP_TASK, reason)
    return PROMOTE_TASK if promote else SKIP_TASK


def register_challenger(run_dir: str, settings: PipelineSettings | None = None) -> str:
    """`promote` branch: register the challenger as a new version (once per run)."""
    settings = prepare(settings)
    rd = RunDir(Path(run_dir))
    decision = read_json(rd.decision_json)
    if not decision["promote"]:
        raise RuntimeError(f"decision was {SKIP_TASK}: {decision['reason']}; not registering")
    ev = read_json(rd.evaluate_json)
    validation = read_json(rd.validation_json)
    run_id = _mlflow_run(rd, settings)
    tags = {f"metric.{key}": f"{value:.6f}" for key, value in ev["metrics"][ev["variant"]].items()}
    tags |= {f"foldin.{key}": f"{value:.6f}" for key, value in ev["challenger"].items()}
    extract = read_json(rd.extract_json)
    # Where the model came from, readable in the registry without opening the run (I1).
    tags |= {
        "data_snapshot": validation["data_snapshot"],
        "include_web": str(extract["include_web"]).lower(),
        "n_web_ratings": str(validation["summary"]["n_web_ratings"]),
        "allow_promotion": str(decision.get("allow_promotion", False)).lower(),
        "variant": ev["variant"],
        "git_sha": git_sha(),
        "airflow_run_id": extract["run_id"],
        "beats_popularity": str(ev["beats_popularity"]).lower(),
        "promotion_reason": decision["reason"],
    }
    version = register_version(ev["model_uri"], run_id, tags, settings.model_name)
    known = read_json(rd.promotion_json) if rd.promotion_json.exists() else None
    if known is not None and known["version"] == version:
        log.info("%s v%s already registered by this run: %s", settings.model_name, version, known)
        return run_dir
    write_json_atomic(
        rd.promotion_json,
        {"version": version, "expected_previous": ev["champion_version"], "promoted": False},
    )
    log.info("registered %s v%s", settings.model_name, version)
    return run_dir


def set_champion_alias(run_dir: str, settings: PipelineSettings | None = None) -> str:
    """Move `champion` to the registered version, unless it moved since evaluation."""
    settings = prepare(settings)
    rd = RunDir(Path(run_dir))
    promotion = read_json(rd.promotion_json)
    _mlflow_run(rd, settings)  # also points MLflow at the right tracking server
    previous = set_champion(
        settings.model_name, promotion["version"], promotion["expected_previous"]
    )
    client = MlflowClient()
    client.set_model_version_tag(settings.model_name, promotion["version"], "promotion", "promoted")
    _set_run_tags(rd, settings, {"promoted": "true", "previous_champion": str(previous)})
    write_json_atomic(rd.promotion_json, promotion | {"promoted": True, "previous": previous})
    log.info("champion of %s: v%s -> v%s", settings.model_name, previous, promotion["version"])
    return run_dir


def log_reason(run_dir: str, settings: PipelineSettings | None = None) -> str:
    """`skip` branch: keep the champion, record why in MLflow and in the task log."""
    settings = prepare(settings)
    rd = RunDir(Path(run_dir))
    decision = read_json(rd.decision_json)
    _set_run_tags(rd, settings, {"promoted": "false", "promotion_reason": decision["reason"]})
    # The alias as it is now (not the version evaluate compared against, which may be stale).
    current = champion_version(settings.mlflow_uri(), settings.model_name)
    log.warning(
        "challenger not promoted, current champion: %s; reason: %s",
        "none" if current is None else f"v{current}",
        decision["reason"],
    )
    return run_dir


# ----- CLI -----------------------------------------------------------------------------------


def cleanup_impressions(settings: PipelineSettings | None = None) -> int:
    """Retention of recommendation_impressions (s5-review I-4): delete rows older than
    IMPRESSIONS_RETENTION. Runs in retrain_daily next to (not inside) the training chain.
    Idempotent; returns the number of rows deleted."""
    settings = prepare(settings)
    engine = make_engine(settings.db_url())
    try:
        deleted = delete_old_impressions(engine)
    finally:
        engine.dispose()
    log.info("deleted %d impressions past the retention", deleted)
    return deleted


def run_all(
    run_id: str,
    settings: PipelineSettings | None = None,
    include_web: bool = True,
    reload: bool = True,
    allow_promotion: bool = False,
    allow_count_drop: bool = False,
) -> str:
    """The whole DAG in one process (same functions, same order)."""
    from recsys.pipelines.reload import reload_api

    run_dir = extract_ratings(run_id, settings, include_web=include_web)
    validate_data(run_dir, settings, allow_count_drop=allow_count_drop)
    for task in (drift_report, split_data, train_als, train_ranker, evaluate):
        task(run_dir, settings)
    if decide_promotion(run_dir, settings, allow_promotion=allow_promotion) == PROMOTE_TASK:
        register_challenger(run_dir, settings)
        set_champion_alias(run_dir, settings)
        if reload:
            reload_api(run_dir, settings)
    else:
        log_reason(run_dir, settings)
    return run_dir


def main(argv: Sequence[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description="Run the retrain_daily chain without Airflow.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--no-web", action="store_true", help="train on MovieLens ratings only")
    parser.add_argument("--no-reload", action="store_true", help="do not call the API")
    parser.add_argument(
        "--allow-promotion", action="store_true", help="let a --no-web run be promoted"
    )
    parser.add_argument(
        "--allow-count-drop", action="store_true", help="accept a >5%% drop in ratings"
    )
    parser.add_argument("--runs-dir", type=Path, default=None, help="default: $RUNS_DIR")
    args = parser.parse_args(argv)
    settings = PipelineSettings(runs_root=args.runs_dir)
    return run_all(
        args.run_id,
        settings,
        include_web=not args.no_web,
        reload=not args.no_reload,
        allow_promotion=args.allow_promotion,
        allow_count_drop=args.allow_count_drop,
    )


if __name__ == "__main__":
    main()
