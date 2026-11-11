"""State shared between `retrain_daily` runs: the `retrain_runs` table (Decision 4, stage 4).

Plain SQLAlchemy Core without dialect-specific upserts, so unit tests run it on SQLite.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import Engine, select

from recsys.db.schema import retrain_runs


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    include_web: bool
    n_ratings: int
    data_snapshot: str
    summary: dict = field(default_factory=dict)
    recorded_at: datetime | None = None


def create_run_state(engine: Engine) -> None:
    """Create only `retrain_runs` (the full schema comes from `create_schema`)."""
    retrain_runs.create(engine, checkfirst=True)


def record_run(engine: Engine, record: RunRecord) -> None:
    """Upsert by run_id. A retry keeps the original `recorded_at`, i.e. its place in history."""
    values = {
        "include_web": record.include_web,
        "n_ratings": record.n_ratings,
        "data_snapshot": record.data_snapshot,
        "summary": record.summary,
    }
    with engine.begin() as conn:
        exists = conn.execute(
            select(retrain_runs.c.run_id).where(retrain_runs.c.run_id == record.run_id)
        ).first()
        if exists:
            conn.execute(
                retrain_runs.update().where(retrain_runs.c.run_id == record.run_id).values(values)
            )
        else:
            conn.execute(
                retrain_runs.insert().values(
                    run_id=record.run_id, recorded_at=datetime.now(UTC), **values
                )
            )


def previous_run(engine: Engine, run_id: str, include_web: bool) -> RunRecord | None:
    """The most recently recorded run other than `run_id` with the same `include_web` flag."""
    query = (
        select(retrain_runs)
        .where(retrain_runs.c.run_id != run_id, retrain_runs.c.include_web == include_web)
        .order_by(retrain_runs.c.recorded_at.desc())
        .limit(1)
    )
    with engine.connect() as conn:
        row = conn.execute(query).mappings().first()
    if row is None:
        return None
    return RunRecord(
        run_id=row["run_id"],
        include_web=bool(row["include_web"]),
        n_ratings=int(row["n_ratings"]),
        data_snapshot=row["data_snapshot"],
        summary=dict(row["summary"]),
        recorded_at=row["recorded_at"],
    )
