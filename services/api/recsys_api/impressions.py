"""Recommendation impressions (spec 5.2): what was shown to whom, for the online metric.

The request only puts the shown list on an in-memory queue (microseconds, never blocks); one
background thread writes queued rows to Postgres in batches. Impressions are analytics, not a
transaction, so the trade-off is explicit:
- queue full (Postgres slow or stuck) -> the response's rows are dropped;
- write fails (Postgres down, bad SQL) -> that batch is lost, no retries, no disk buffer;
both are counted in `recommendation_impressions_total{result=dropped|failed}` and the response
is never affected.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import NamedTuple

from recsys_api.metrics import ApiMetrics
from recsys_api.observer import Observer

log = logging.getLogger(__name__)

MAX_QUEUE = 2_000  # responses (each up to MAX_K = 50 rows), i.e. at most ~100k rows in memory
BATCH_SIZE = 500  # rows per INSERT, roughly


class Impression(NamedTuple):
    user_id: int
    movie_id: int
    position: int  # 1-based
    model_version: str | None
    path: str
    shown_at: datetime


Sink = Callable[[Sequence[Impression]], None]
_STOP = object()


class ImpressionLogger:
    def __init__(
        self,
        sink: Sink,
        *,
        metrics: ApiMetrics | None = None,
        max_queue: int = MAX_QUEUE,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._sink = sink
        self._metrics = metrics
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._batch_size = batch_size
        self._thread: threading.Thread | None = None

    # ----- request side ----------------------------------------------------------------------
    def record(
        self,
        user_id: int,
        movie_ids: Sequence[int],
        path: str,
        model_version: str | None,
        shown_at: datetime | None = None,
    ) -> int:
        """Queue one response's impressions; returns the number of rows queued (0 = dropped)."""
        if not movie_ids:
            return 0
        shown_at = shown_at or datetime.now(UTC)
        rows = tuple(
            Impression(int(user_id), int(m), pos, model_version, path, shown_at)
            for pos, m in enumerate(movie_ids, start=1)
        )
        try:
            self._queue.put_nowait(rows)
        except queue.Full:
            self._count("dropped", len(rows))
            return 0
        return len(rows)

    # ----- writer side -----------------------------------------------------------------------
    def _count(self, result: str, n: int) -> None:
        if self._metrics is not None and n:
            self._metrics.impressions.labels(result).inc(n)

    def _write(self, rows: list[Impression]) -> None:
        try:
            self._sink(rows)
        except Exception as exc:  # noqa: BLE001 - see the module docstring
            log.warning("%d impressions not written: %s", len(rows), exc)
            self._count("failed", len(rows))
        else:
            self._count("written", len(rows))

    def _next_batch(self, first) -> tuple[list[Impression], int, bool]:
        """`first` plus whatever is queued, up to ~batch_size rows: (rows, items taken, stop)."""
        rows: list[Impression] = []
        taken, stop, item = 0, False, first
        while True:
            taken += 1
            if item is _STOP:
                stop = True
                break
            rows.extend(item)
            if len(rows) >= self._batch_size:
                break
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
        return rows, taken, stop

    def _process(self, first) -> bool:
        rows, taken, stop = self._next_batch(first)
        try:
            if rows:
                self._write(rows)
        finally:
            for _ in range(taken):
                self._queue.task_done()
        return stop

    def drain(self) -> None:
        """Write everything queued now, in this thread (tests; the writer thread is not used)."""
        while True:
            try:
                first = self._queue.get_nowait()
            except queue.Empty:
                return
            if self._process(first):
                return

    def _run(self) -> None:
        while True:
            if self._process(self._queue.get()):
                return

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if not self.running:
            self._thread = threading.Thread(target=self._run, name="impressions", daemon=True)
            self._thread.start()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until everything queued so far is written (or failed); False on timeout."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def close(self, timeout: float = 5.0) -> None:
        """Flush and stop the writer (app shutdown). Rows still queued after `timeout` are lost."""
        if not self.running:
            return
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            log.warning("impression queue still full at shutdown, dropping it")
            return
        self._thread.join(timeout)


class ImpressionObserver(Observer):
    """Logs every recommendation response shown to a visitor with an id (spec 5.2). Anonymous
    responses are skipped: without a user id they cannot be joined to a later rating."""

    def __init__(self, logger: ImpressionLogger) -> None:
        self.logger = logger

    def recommendations_served(
        self,
        user_id: int | None,
        path: str,
        movie_ids: Sequence[int],
        model_version: str | None,
        cached: bool,
    ) -> None:
        if user_id is not None:
            self.logger.record(user_id, movie_ids, path, model_version)
