"""Recommendation impressions (spec 5.2): written off the request path, never break a response."""

import threading
import time
from datetime import UTC, datetime

import pytest
from api_fakes import FakeChampion, FakeRepository, FakeStore, StubRecommender
from fastapi.testclient import TestClient

from recsys_api.impressions import Impression, ImpressionLogger, ImpressionObserver
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
    assert [(r.movie_id, r.position, r.path, r.model_version) for r in repo.impressions] == [
        (m, i, "full", "6") for i, m in enumerate(shown, start=1)
    ] * 2  # the cached answer was shown too


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
