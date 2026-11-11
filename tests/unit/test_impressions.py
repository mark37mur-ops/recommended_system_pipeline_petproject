"""Recommendation impressions (spec 5.2): written off the request path, never break a response."""

import threading
import time
from datetime import UTC, datetime

import pytest
from api_fakes import FakeChampion, FakeRepository, FakeStore, StubRecommender
from fastapi.testclient import TestClient

from recsys_api.impressions import (
    Impression,
    ImpressionLogger,
    ImpressionObserver,
    RecentImpressions,
)
from recsys_api.main import create_app
from recsys_api.metrics import ApiMetrics
from recsys_api.settings import Settings

SESSION_SECRET = "unit-test-session-secret-0123456789abcdef"
NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


class Sink:
    def __init__(self, fail: bool = False, block: threading.Event | None = None) -> None:
        self.batches: list[list[Impression]] = []
        self.fail = fail
        self.block = block
        self.calls = 0

    def __call__(self, rows):
        self.calls += 1
        if self.block is not None:
            self.block.wait(5)
        if self.fail:
            raise RuntimeError("database is down")
        self.batches.append(list(rows))

    @property
    def rows(self):
        return [row for batch in self.batches for row in batch]


def counter(metrics: ApiMetrics, result: str) -> float:
    return metrics.registry.get_sample_value("recommendation_impressions_total", {"result": result})


def test_record_turns_a_response_into_one_row_per_position():
    sink, metrics = Sink(), ApiMetrics()
    logger = ImpressionLogger(sink, metrics=metrics)
    assert logger.record(1_000_001, [30, 20, 10], "full", "6", shown_at=NOW) == 3
    logger.drain()
    assert sink.rows == [
        Impression(1_000_001, 30, 1, "6", "full", NOW),
        Impression(1_000_001, 20, 2, "6", "full", NOW),
        Impression(1_000_001, 10, 3, "6", "full", NOW),
    ]
    assert counter(metrics, "written") == 3


def test_batches_are_bounded():
    sink = Sink()
    logger = ImpressionLogger(sink, batch_size=25)
    for _ in range(6):
        logger.record(1_000_001, list(range(10)), "full", "6", shown_at=NOW)
    logger.drain()
    assert [len(b) for b in sink.batches] == [30, 30]  # whole responses, ~batch_size rows
    assert len(sink.rows) == 60


def test_record_never_waits_for_a_slow_database():
    release = threading.Event()
    sink = Sink(block=release)
    logger = ImpressionLogger(sink, max_queue=1000)
    logger.start()
    try:
        logger.record(1_000_001, [1, 2], "full", "6")  # the writer thread now hangs in the sink
        started = time.perf_counter()
        for _ in range(200):
            logger.record(1_000_001, list(range(10)), "full", "6")
        assert time.perf_counter() - started < 0.5
    finally:
        release.set()
        logger.close(timeout=5)
    assert len(sink.rows) == 2 + 200 * 10


def test_full_queue_drops_and_counts_instead_of_blocking():
    metrics = ApiMetrics()
    logger = ImpressionLogger(Sink(), metrics=metrics, max_queue=2)  # writer not started
    for _ in range(5):
        logger.record(1_000_001, [1, 2, 3], "full", "6")
    assert counter(metrics, "dropped") == 9  # 3 responses x 3 rows did not fit
    logger.drain()
    assert counter(metrics, "written") == 6


def test_write_errors_are_counted_and_the_writer_keeps_going(caplog):
    sink, metrics = Sink(fail=True), ApiMetrics()
    logger = ImpressionLogger(sink, metrics=metrics)
    logger.record(1_000_001, [1, 2], "full", "6")
    logger.drain()
    assert counter(metrics, "failed") == 2
    assert "impressions not written" in caplog.text
    sink.fail = False
    logger.record(1_000_001, [3], "full", "6")
    logger.drain()
    assert [r.movie_id for r in sink.rows] == [3]


def test_close_flushes_what_is_queued():
    sink = Sink()
    logger = ImpressionLogger(sink)
    logger.start()
    for i in range(20):
        logger.record(1_000_001, [i], "full", "6")
    logger.close(timeout=5)
    assert sorted(r.movie_id for r in sink.rows) == list(range(20))
    assert not logger.running


def test_observer_logs_visitors_only():
    sink = Sink()
    logger = ImpressionLogger(sink)
    observer = ImpressionObserver(logger)
    observer.recommendations_served(None, "cold_start", [1, 2, 3], "6", cached=False)
    observer.recommendations_served(1_000_002, "fallback", [], None, cached=False)
    observer.recommendations_served(1_000_003, "cold_start", [5, 6], None, cached=True)
    logger.drain()
    assert [(r.user_id, r.movie_id, r.model_version) for r in sink.rows] == [
        (1_000_003, 5, None),
        (1_000_003, 6, None),
    ]


# ----- through the HTTP app ------------------------------------------------------------------


class Registry:
    def probe(self):
        return "6"

    def load(self):
        return FakeChampion(StubRecommender(), "6")


class HangingRepository(FakeRepository):
    """insert_impressions hangs (database stuck) or fails; everything else works."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.release = threading.Event()
        self.inserted = []

    def insert_impressions(self, rows):
        if self.mode == "hang":
            self.release.wait(10)
        if self.mode == "fail":
            raise RuntimeError("relation recommendation_impressions does not exist")
        self.inserted.extend(rows)


def make_app(repo):
    settings = Settings(session_secret=SESSION_SECRET, log_format="text")
    registry = Registry()
    return create_app(
        settings,
        repo=repo,
        store=FakeStore(),
        loader=registry.load,
        probe=registry.probe,
        background=False,
    )


def rate_five(client):
    for movie in range(1, 6):
        assert client.post("/ratings", json={"movie_id": movie, "rating": 5}).status_code == 200


def test_app_logs_every_shown_recommendation():
    repo = FakeRepository()
    app = make_app(repo)
    with TestClient(app) as client:
        client.get("/recommendations")  # anonymous: not logged
        rate_five(client)
        body = client.get("/recommendations", params={"k": 4}).json()
        cached = client.get("/recommendations", params={"k": 4}).json()
        assert app.state.impressions.flush(timeout=5)
    assert body["path"] == "full" and cached["cached"] is True
    shown = [m["movie_id"] for m in body["items"]]
    # The cached repeat shows the same movies again within the hour: not logged twice
    # (s5-review I-4); the online metric only ever uses the first impression of a pair.
    assert [(r.movie_id, r.position, r.path, r.model_version) for r in repo.impressions] == [
        (m, i, "full", "6") for i, m in enumerate(shown, start=1)
    ]
    assert counter(app.state.metrics, "deduplicated") == 4


@pytest.mark.parametrize("mode", ["hang", "fail"])
def test_a_stuck_or_failing_impression_write_does_not_affect_responses(mode):
    repo = HangingRepository(mode)
    app = make_app(repo)
    try:
        with TestClient(app) as client:
            rate_five(client)
            started = time.perf_counter()
            for _ in range(20):
                response = client.get("/recommendations")
                assert response.status_code == 200 and response.json()["path"] == "full"
            assert time.perf_counter() - started < 2.0
            repo.release.set()
    finally:
        repo.release.set()


# ----- repeated impressions (s5-review I-4) ----------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_pair_is_logged_once_per_group_within_the_window():
    clock = Clock()
    recent = RecentImpressions(window=3600, clock=clock)
    user = 1_000_001
    assert recent.fresh(user, "full", "6", [1, 2, 3]) == [(1, 1), (2, 2), (3, 3)]
    clock.now += 1800
    # a refresh shows the same three plus one more: only the new one, at its real position
    assert recent.fresh(user, "full", "6", [1, 2, 3, 4]) == [(4, 4)]
    # another group of the online metric (path, model version) counts on its own
    assert recent.fresh(user, "cold_start", None, [1, 2]) == [(1, 1), (2, 2)]
    assert recent.fresh(user, "full", "7", [2]) == [(1, 2)]
    assert recent.fresh(user + 1, "full", "6", [1]) == [(1, 1)]
    clock.now += 1799
    assert recent.fresh(user, "full", "6", [1, 2, 3, 4]) == []
    clock.now += 1  # an hour after 1-3 were logged: logged again; 4 is only 30 min old
    assert recent.fresh(user, "full", "6", [1, 4]) == [(1, 1)]


def test_recent_impressions_are_bounded():
    clock = Clock()
    recent = RecentImpressions(window=3600, max_entries=3, clock=clock)
    assert len(recent.fresh(1_000_001, "full", "6", [1, 2, 3, 4, 5])) == 5
    assert len(recent) == 3
    assert recent.fresh(1_000_001, "full", "6", [1]) == [(1, 1)]  # evicted: logged again
    clock.now += 7200
    recent.fresh(1_000_002, "full", "6", [9])
    assert len(recent) == 1  # expired entries are pruned


def test_window_zero_logs_every_impression():
    recent = RecentImpressions(window=0)
    assert recent.fresh(1_000_001, "full", "6", [1, 2]) == [(1, 1), (2, 2)]
    assert recent.fresh(1_000_001, "full", "6", [1, 2]) == [(1, 1), (2, 2)]


def test_observer_skips_repeats_and_counts_them():
    sink, metrics = Sink(), ApiMetrics()
    logger = ImpressionLogger(sink, metrics=metrics)
    observer = ImpressionObserver(logger, recent=RecentImpressions(window=3600))
    observer.recommendations_served(1_000_001, "full", [7, 8, 9], "6", cached=False)
    observer.recommendations_served(1_000_001, "full", [7, 8, 9], "6", cached=True)
    observer.recommendations_served(1_000_001, "full", [7, 5], "6", cached=False)
    logger.drain()
    assert [(r.movie_id, r.position) for r in sink.rows] == [(7, 1), (8, 2), (9, 3), (5, 2)]
    assert counter(metrics, "deduplicated") == 4
    assert counter(metrics, "written") == 4


def test_dropped_impressions_are_not_remembered_as_logged():
    """Queue full: the rows are lost, so the next showing must be logged instead of skipped."""
    metrics = ApiMetrics()
    logger = ImpressionLogger(Sink(), metrics=metrics, max_queue=1)  # writer not started
    observer = ImpressionObserver(logger, recent=RecentImpressions(window=3600))
    observer.recommendations_served(1_000_001, "full", [1], "6", cached=False)  # queued
    observer.recommendations_served(1_000_001, "full", [2], "6", cached=False)  # dropped
    logger.drain()
    observer.recommendations_served(1_000_001, "full", [1, 2], "6", cached=True)
    logger.drain()
    assert counter(metrics, "dropped") == 1
    assert counter(metrics, "deduplicated") == 1  # movie 1
    assert counter(metrics, "written") == 2  # movie 1 first time, movie 2 on the retry
