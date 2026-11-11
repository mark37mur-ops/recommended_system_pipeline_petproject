"""Recommendation impressions (spec 5.2): what was shown to whom, for the online metric.

The request only puts the shown list on an in-memory queue (microseconds, never blocks); one
background thread writes queued rows to Postgres in batches. Impressions are analytics, not a
transaction, so the trade-off is explicit:
- queue full (Postgres slow or stuck) -> the response's rows are dropped;
- write fails (Postgres down, bad SQL) -> that batch is lost, no retries, no disk buffer;
both are counted in `recommendation_impressions_total{result=dropped|failed}` and the response
is never affected.

Repeats are not written (s5-review I-4): the page refreshes recommendations after every rating
and a cached answer shows the same list again, so one visitor would add up to 50 rows per
request. The online metric uses only the first impression of a pair within its group
(path, model version), so a (visitor, path, version, movie) already logged within the last
hour (`RecentImpressions`) is skipped and counted as `result="deduplicated"`. What this costs:
the memory of what was logged is per process and bounded, so after a restart or an eviction a
pair may be written once more (harmless for the metric: the view keeps the first).
"""

import logging
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import NamedTuple

from recsys_api.metrics import ApiMetrics
from recsys_api.observer import Observer

log = logging.getLogger(__name__)

MAX_QUEUE = 2_000  # responses (each up to MAX_K = 50 rows), i.e. at most ~100k rows in memory
BATCH_SIZE = 500  # rows per INSERT, roughly
DEDUP_SECONDS = 3600.0
DEDUP_MAX_ENTRIES = 50_000  # (visitor, path, version, movie) keys, ~10 MB at most


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
        positions: Sequence[int] | None = None,
    ) -> int:
        """Queue one response's impressions; returns the number of rows queued (0 = dropped).
        `positions` (1-based, one per movie) when only part of the response is logged;
        default 1..n."""
        if not movie_ids:
            return 0
        shown_at = shown_at or datetime.now(UTC)
        positions = positions or range(1, len(movie_ids) + 1)
        rows = tuple(
            Impression(int(user_id), int(m), int(pos), model_version, path, shown_at)
            for pos, m in zip(positions, movie_ids, strict=True)
        )
        try:
            self._queue.put_nowait(rows)
        except queue.Full:
            self._count("dropped", len(rows))
            return 0
        return len(rows)

    def skipped(self, n: int) -> None:
        """Count rows not queued because the same pair was logged recently."""
        self._count("deduplicated", n)

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


class RecentImpressions:
    """Which (visitor, path, model version, movie) were logged within the last `window`
    seconds. An LRU in insertion-time order: expired keys are pruned from the front, and at
    most `max_entries` are kept (an evicted pair is merely logged again). Thread-safe: request
    handlers run in a thread pool."""

    def __init__(
        self,
        window: float = DEDUP_SECONDS,
        max_entries: int = DEDUP_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.window = window
        self.max_entries = max_entries
        self._clock = clock
        self._logged: OrderedDict[tuple, float] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._logged)

    def fresh(
        self, user_id: int, path: str, model_version: str | None, movie_ids: Sequence[int]
    ) -> list[tuple[int, int]]:
        """(position, movie) of the movies not logged for this group within the window, which
        are now remembered as logged."""
        if self.window <= 0:
            return list(enumerate(movie_ids, start=1))
        now = self._clock()
        out = []
        with self._lock:
            while self._logged:
                key, at = next(iter(self._logged.items()))
                if now - at < self.window:
                    break
                del self._logged[key]
            for position, movie in enumerate(movie_ids, start=1):
                key = (user_id, path, model_version, int(movie))
                at = self._logged.get(key)
                if at is not None and now - at < self.window:
                    continue
                self._logged[key] = now
                self._logged.move_to_end(key)
                out.append((position, movie))
            while len(self._logged) > self.max_entries:
                self._logged.popitem(last=False)
        return out

    def forget(
        self, user_id: int, path: str, model_version: str | None, movie_ids: Sequence[int]
    ) -> None:
        """The rows were not queued after all (queue full): log them the next time."""
        with self._lock:
            for movie in movie_ids:
                self._logged.pop((user_id, path, model_version, int(movie)), None)


class ImpressionObserver(Observer):
    """Logs every recommendation response shown to a visitor with an id (spec 5.2). Anonymous
    responses are skipped: without a user id they cannot be joined to a later rating. With
    `recent`, movies already logged for the same group within its window are skipped."""

    def __init__(self, logger: ImpressionLogger, recent: RecentImpressions | None = None) -> None:
        self.logger = logger
        self.recent = recent

    def recommendations_served(
        self,
        user_id: int | None,
        path: str,
        movie_ids: Sequence[int],
        model_version: str | None,
        cached: bool,
    ) -> None:
        if user_id is None or not movie_ids:
            return
        if self.recent is None:
            self.logger.record(user_id, movie_ids, path, model_version)
            return
        fresh = self.recent.fresh(user_id, path, model_version, movie_ids)
        self.logger.skipped(len(movie_ids) - len(fresh))
        if not fresh:
            return
        positions = [pos for pos, _ in fresh]
        movies = [movie for _, movie in fresh]
        if not self.logger.record(user_id, movies, path, model_version, positions=positions):
            self.recent.forget(user_id, path, model_version, movies)
