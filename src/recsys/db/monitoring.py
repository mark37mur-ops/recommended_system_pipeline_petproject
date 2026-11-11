"""Monitoring rows the pipeline writes for Grafana (stage 5): drift_metrics, run_metrics, and
the impressions drift_report reads. Plain SQLAlchemy Core (delete + insert in one transaction
instead of a dialect-specific upsert), so the unit tests run it on SQLite.

The tables are created here with `checkfirst` when missing (a pipeline run before the API ever
started); the API's `create_schema` creates them as well and grants them to the dashboard
role.
"""

from collections.abc import Mapping
from datetime import UTC, datetime

from sqlalchemy import Engine, inspect, select

from recsys.db.schema import drift_metrics, recommendation_impressions, run_metrics


def recent_impressions(
    engine: Engine, since: datetime, until: datetime, path: str = "full"
) -> list[int]:
    """Movie ids of the impressions with `since < shown_at <= until` (one entry per
    impression). Empty if the table does not exist yet (the API never served anyone)."""
    if not inspect(engine).has_table(recommendation_impressions.name):
        return []
    t = recommendation_impressions
    query = (
        select(t.c.movie_id)
        .where(t.c.path == path, t.c.shown_at > since, t.c.shown_at <= until)
        .order_by(t.c.impression_id)
    )
    with engine.connect() as conn:
        return [int(m) for m in conn.execute(query).scalars()]


def record_drift(engine: Engine, row: Mapping) -> None:
    """One row per run (`run_id`); a retry of drift_report replaces it."""
    drift_metrics.create(engine, checkfirst=True)
    values = dict(row)
    values.setdefault("computed_at", datetime.now(UTC))
    with engine.begin() as conn:
        conn.execute(drift_metrics.delete().where(drift_metrics.c.run_id == values["run_id"]))
        conn.execute(drift_metrics.insert().values(values))


def record_run_metrics(engine: Engine, run_id: str, metrics: Mapping[str, float]) -> None:
    """Replace the metrics of `run_id` (long format: one row per metric)."""
    run_metrics.create(engine, checkfirst=True)
    now = datetime.now(UTC)
    rows = [
        {"run_id": run_id, "metric": name, "value": float(value), "recorded_at": now}
        for name, value in sorted(metrics.items())
    ]
    with engine.begin() as conn:
        conn.execute(run_metrics.delete().where(run_metrics.c.run_id == run_id))
        if rows:
            conn.execute(run_metrics.insert(), rows)
