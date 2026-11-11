"""Prometheus metrics of the API (spec 5.1), served on GET /metrics.

Why a private `CollectorRegistry` per app instead of prometheus_client's global one: every
`create_app` (tests create dozens) gets its own counters, and nothing else in the process
(libraries registering their own collectors) leaks into our /metrics.

Label cardinality is bounded on purpose: every distinct label combination is a separate time
series that Prometheus keeps in memory. `endpoint` is the route *template* (`/movies/search`,
never the raw URL with its query string or a 404 path somebody made up), unknown paths are one
value `unmatched`, `status` is the response code, `model_info` keeps exactly one series.

uvicorn runs one process (no --workers), so the ordinary in-process registry is right; several
worker processes would need prometheus_client's multiprocess mode.
"""

import logging
import time
from collections.abc import Sequence

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
    generate_latest,
)

from recsys_api.observer import Observer

# `*_created` series (one timestamp per counter) double the output and no query here uses them.
disable_created_metrics()

# Seconds. Dense around the p95 target of 100 ms (spec 3.5) and with an exact 0.3 bucket for
# the 300 ms alert (spec 5.5): histogram_quantile interpolates linearly inside a bucket, so a
# boundary at the threshold keeps "p95 > 300 ms" from firing on interpolation alone.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 2.5, 5, 10)
PATHS = ("full", "cold_start", "fallback")
CACHE_RESULTS = ("hit", "miss")
SOURCES = ("web",)
IMPRESSION_RESULTS = ("written", "dropped", "failed")
UNMATCHED = "unmatched"
# Scraped every 15 s / probed by healthchecks: logged at DEBUG, otherwise they drown the rest.
QUIET_ENDPOINTS = frozenset({"/metrics", "/health"})

access_log = logging.getLogger("recsys_api.access")


class ApiMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.http_requests = Counter(
            "http_requests_total",
            "HTTP requests by route template and response status code",
            ["endpoint", "status"],
            registry=r,
        )
        self.http_duration = Histogram(
            "http_request_duration_seconds",
            "Time from request to the end of the response, by route template",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.recommendations_served = Counter(
            "recommendations_served_total",
            "Recommendation responses by path: full (model), cold_start, fallback (popular)",
            ["path"],
            registry=r,
        )
        self.ratings_received = Counter(
            "ratings_received_total", "Ratings accepted by POST /ratings", ["source"], registry=r
        )
        self.cache_requests = Counter(
            "cache_requests_total", "Recommendation cache lookups", ["result"], registry=r
        )
        self.model_info = Gauge(
            "model_info", "1 for the model version being served", ["version"], registry=r
        )
        self.model_loaded = Gauge(
            "model_loaded_timestamp_seconds",
            "Unix time the API loaded the served model",
            registry=r,
        )
        self.model_created = Gauge(
            "model_created_timestamp_seconds",
            "Unix time the served model version was created in the MLflow registry",
            registry=r,
        )
        self.model_errors = Counter(
            "model_errors_total",
            "Inference failures answered with popular movies instead (spec 3.5)",
            registry=r,
        )
        self.impressions = Counter(
            "recommendation_impressions_total",
            "Impression rows: written to Postgres, dropped (queue full), failed (write error)",
            ["result"],
            registry=r,
        )
        # Every known label value exported as 0 from the start: rate() and ratios need a series
        # before the first event, and a missing series looks like "no data" on the dashboard.
        for path in PATHS:
            self.recommendations_served.labels(path)
        for result in CACHE_RESULTS:
            self.cache_requests.labels(result)
        for source in SOURCES:
            self.ratings_received.labels(source)
        for result in IMPRESSION_RESULTS:
            self.impressions.labels(result)

    def observe_request(self, endpoint: str, status: int, seconds: float) -> None:
        self.http_requests.labels(endpoint, str(status)).inc()
        self.http_duration.labels(endpoint).observe(seconds)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


def endpoint_of(scope: dict, mounts: tuple[str, ...] = ()) -> str:
    """Route template the router matched. FastAPI puts the matched APIRoute into the scope;
    for a mounted app (StaticFiles) it does not, so a request under one of `mounts` is
    labelled with the mount prefix."""
    path = getattr(scope.get("route"), "path", None)
    if isinstance(path, str) and path:
        return path
    raw = scope.get("path", "")
    for prefix in mounts:
        if raw == prefix or raw.startswith(prefix + "/"):
            return prefix
    return UNMATCHED


class MetricsMiddleware:
    """Pure ASGI middleware: request count + latency per route template, one access-log line.

    Pure ASGI rather than BaseHTTPMiddleware: no extra task and response copy per request.
    Latency runs until the last body chunk is sent. An exception escaping the app is counted
    as 500 (the outer ServerErrorMiddleware turns it into exactly that) and re-raised."""

    def __init__(self, app, metrics: ApiMetrics, mounts: tuple[str, ...] = ()) -> None:
        self.app = app
        self.metrics = metrics
        self.mounts = mounts

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status = 500

        async def send_and_record(message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_and_record)
        finally:
            elapsed = time.perf_counter() - start
            endpoint = endpoint_of(scope, self.mounts)
            self.metrics.observe_request(endpoint, status, elapsed)
            access_log.log(
                logging.DEBUG if endpoint in QUIET_ENDPOINTS else logging.INFO,
                "%s %s %d",
                scope["method"],
                endpoint,
                status,
                extra={
                    "method": scope["method"],
                    "endpoint": endpoint,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )


class MetricsObserver(Observer):
    """Service events (recsys_api.service, recsys_api.model) -> counters and gauges."""

    def __init__(self, metrics: ApiMetrics) -> None:
        self.metrics = metrics

    def recommendations_served(
        self,
        user_id: int | None,
        path: str,
        movie_ids: Sequence[int],
        model_version: str | None,
        cached: bool,
    ) -> None:
        self.metrics.recommendations_served.labels(path).inc()

    def rating_received(self, source: str) -> None:
        self.metrics.ratings_received.labels(source).inc()

    def cache_lookup(self, hit: bool) -> None:
        self.metrics.cache_requests.labels("hit" if hit else "miss").inc()

    def model_error(self, exc: BaseException) -> None:
        self.metrics.model_errors.inc()

    def model_loaded(self, version: str, loaded_at: float, created_at: float | None = None) -> None:
        # clear + set: the previous version's series disappears instead of lingering at 1.
        self.metrics.model_info.clear()
        self.metrics.model_info.labels(version).set(1)
        self.metrics.model_loaded.set(loaded_at)
        # Unknown creation time -> NaN, not the previous model's (wrong) age.
        self.metrics.model_created.set(float("nan") if created_at is None else created_at)
