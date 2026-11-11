"""retrain_runs bookkeeping on a throw-away SQLite database (the code is dialect-neutral)."""

import pytest
from sqlalchemy import create_engine

from recsys.db.run_state import RunRecord, create_run_state, previous_run, record_run


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'state.db').as_posix()}")
    create_run_state(engine)
    yield engine
    engine.dispose()


def _record(run_id: str, n: int, include_web: bool = True) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        include_web=include_web,
        n_ratings=n,
        data_snapshot=f"snap-{n}",
        summary={"n_ratings": n, "web_share": 0.1},
    )


def test_no_previous_run_on_empty_state(engine):
    assert previous_run(engine, "r1", include_web=True) is None


def test_previous_run_is_the_latest_other_run_with_the_same_flag(engine):
    record_run(engine, _record("r1", 100))
    record_run(engine, _record("r2", 110))
    record_run(engine, _record("no-web", 90, include_web=False))
    record_run(engine, _record("r3", 120))

    prev = previous_run(engine, "r3", include_web=True)
    assert prev.run_id == "r2" and prev.n_ratings == 110
    assert prev.summary == {"n_ratings": 110, "web_share": 0.1}
    assert previous_run(engine, "r4", include_web=True).run_id == "r3"
    assert previous_run(engine, "r4", include_web=False).run_id == "no-web"


def test_record_run_is_an_upsert_that_keeps_the_original_order(engine):
    record_run(engine, _record("r1", 100))
    record_run(engine, _record("r2", 110))
    record_run(engine, _record("r1", 100))  # retry of an older run
    assert previous_run(engine, "r3", include_web=True).run_id == "r2"
    assert previous_run(engine, "r2", include_web=True).run_id == "r1"
