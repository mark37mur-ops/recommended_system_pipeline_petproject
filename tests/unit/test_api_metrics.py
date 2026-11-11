"""Prometheus metrics of the API (spec 5.1): names, types, labels, bounded cardinality."""

import logging
import math

import pytest
from api_fakes import FakeChampion, FakeRepository, FakeStore, StubRecommender
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from recsys_api.main import create_app
from recsys_api.metrics import LATENCY_BUCKETS, ApiMetrics
from recsys_api.settings import Settings

SESSION_SECRET = "unit-test-session-secret-0123456789abcdef"
TOKEN = "unit-test-admin-token-0123456789abcdef"
CREATED_AT = 1_790_000_000.0


class Registry:
    def __init__(self, version="7"):
        self.version = version

    def probe(self):
        return self.version

    def load(self):
        champion = FakeChampion(StubRecommender(), self.version)
        champion.created_at = CREATED_AT + int(self.version)
        return champion


class BrokenRepository(FakeRepository):
    """A bug (not an outage): user_ratings raises something the service does not handle."""

    def user_ratings(self, user_id):
        if self.ratings.get(user_id):
            raise RuntimeError("bug")
        return super().user_ratings(user_id)


def make(repo=None, registry=None):
    settings = Settings(session_secret=SESSION_SECRET, admin_token=TOKEN, log_format="text")
    registry = registry or Registry()
    app = create_app(
        settings,
        repo=repo or FakeRepository(),
        store=FakeStore(),
        loader=registry.load,
        probe=registry.probe,
        background=False,
    )
    return app, registry


@pytest.fixture
def app():
    app, _ = make()
    return app


def scrape(client) -> dict:
    """{family name: {(sample name, frozenset(labels)): value}} plus types under '__types__'."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    out: dict = {"__types__": {}}
    for family in text_string_to_metric_families(response.text):
        out["__types__"][family.name] = family.type
        out[family.name] = {(s.name, frozenset(s.labels.items())): s.value for s in family.samples}
    return out


def value(metrics, family, sample=None, **labels):
    return metrics[family].get((sample or family, frozenset(labels.items())))


def series(metrics, family, sample):
    return [labels for (name, labels) in metrics[family] if name == sample]


def test_spec_metrics_exist_with_their_types(app):
    with TestClient(app) as client:
        metrics = scrape(client)
    types = metrics["__types__"]
    # The parser drops the _total suffix of counters from the family name.
    assert types["http_requests"] == "counter"
    assert types["http_request_duration_seconds"] == "histogram"
    assert types["recommendations_served"] == "counter"
    assert types["ratings_received"] == "counter"
    assert types["cache_requests"] == "counter"
    assert types["model_info"] == "gauge"
    assert types["model_loaded_timestamp_seconds"] == "gauge"
    assert types["model_created_timestamp_seconds"] == "gauge"
    assert types["model_errors"] == "counter"
    assert types["recommendation_impressions"] == "counter"
    assert not any(name.endswith("_created") for name in metrics)  # no *_created noise


def test_label_values_start_at_zero_so_rates_exist_before_the_first_event(app):
    with TestClient(app) as client:
        metrics = scrape(client)
    for path in ("full", "cold_start", "fallback"):
        assert (
            value(metrics, "recommendations_served", "recommendations_served_total", path=path) == 0
        )
    for result in ("hit", "miss"):
        assert value(metrics, "cache_requests", "cache_requests_total", result=result) == 0
    assert value(metrics, "ratings_received", "ratings_received_total", source="web") == 0


def test_endpoint_label_is_the_route_template(app):
    with TestClient(app) as client:
        for q in ("matrix", "alien", "toy story"):
            client.get("/movies/search", params={"q": q})
        client.get("/static/app.js")
        client.get("/static/style.css")
        metrics = scrape(client)
    total = "http_requests_total"
    assert value(metrics, "http_requests", total, endpoint="/movies/search", status="200") == 3
    assert value(metrics, "http_requests", total, endpoint="/static", status="200") == 2
    count = value(
        metrics,
        "http_request_duration_seconds",
        "http_request_duration_seconds_count",
        endpoint="/movies/search",
    )
    assert count == 3


def test_unknown_paths_share_one_series(app):
    with TestClient(app) as client:
        for i in range(100):
            client.get(f"/no/such/page/{i}")
        metrics = scrape(client)
    unmatched = [
        dict(labels)
        for labels in series(metrics, "http_requests", "http_requests_total")
        if dict(labels)["endpoint"] == "unmatched"
    ]
    assert unmatched == [{"endpoint": "unmatched", "status": "404"}]
    assert (
        value(metrics, "http_requests", "http_requests_total", endpoint="unmatched", status="404")
        == 100
    )


def test_unhandled_error_is_counted_as_500():
    app, _ = make(repo=BrokenRepository())
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/ratings", json={"movie_id": 1, "rating": 5})
        assert client.get("/recommendations").status_code == 500
        metrics = scrape(client)
    assert (
        value(
            metrics,
            "http_requests",
            "http_requests_total",
            endpoint="/recommendations",
            status="500",
        )
        == 1
    )


def test_service_events(app):
    with TestClient(app) as client:
        client.get("/recommendations")  # no cookie: cold start, no cache lookup
        for movie in range(1, 6):
            client.post("/ratings", json={"movie_id": movie, "rating": 5})
        client.get("/recommendations")  # miss -> full
        client.get("/recommendations")  # hit
        metrics = scrape(client)
    served = "recommendations_served_total"
    assert value(metrics, "recommendations_served", served, path="cold_start") == 1
    assert value(metrics, "recommendations_served", served, path="full") == 2
    assert value(metrics, "ratings_received", "ratings_received_total", source="web") == 5
    assert value(metrics, "cache_requests", "cache_requests_total", result="miss") == 1
    assert value(metrics, "cache_requests", "cache_requests_total", result="hit") == 1


def test_model_info_keeps_one_series_across_reloads():
    app, registry = make()
    with TestClient(app) as client:
        before = scrape(client)
        assert series(before, "model_info", "model_info") == [frozenset({("version", "7")})]
        assert value(before, "model_info", version="7") == 1
        assert value(before, "model_created_timestamp_seconds") == CREATED_AT + 7
        loaded_before = value(before, "model_loaded_timestamp_seconds")
        assert loaded_before > 0

        registry.version = "8"
        reload = client.post("/admin/reload-model", headers={"Authorization": f"Bearer {TOKEN}"})
        assert reload.json()["status"] == "loaded"
        after = scrape(client)
    assert series(after, "model_info", "model_info") == [frozenset({("version", "8")})]
    assert value(after, "model_created_timestamp_seconds") == CREATED_AT + 8
    assert value(after, "model_loaded_timestamp_seconds") >= loaded_before


def test_model_error_is_counted():
    app, _ = make()
    with TestClient(app) as client:
        for movie in range(1, 6):
            client.post("/ratings", json={"movie_id": movie, "rating": 5})
        app.state.models.current.recommender.fail = True
        assert client.get("/recommendations").json()["path"] == "fallback"
        metrics = scrape(client)
    assert value(metrics, "model_errors", "model_errors_total") == 1
    served = "recommendations_served_total"
    assert value(metrics, "recommendations_served", served, path="fallback") == 1


def test_apps_do_not_share_metrics():
    first, _ = make()
    second, _ = make()
    with TestClient(first) as a, TestClient(second) as b:
        a.get("/health")
        assert (
            value(
                scrape(b), "http_requests", "http_requests_total", endpoint="/health", status="200"
            )
            is None
        )


def test_latency_buckets_have_the_alert_threshold():
    assert 0.1 in LATENCY_BUCKETS and 0.3 in LATENCY_BUCKETS  # spec 3.5 target, 5.5 alert
    assert list(LATENCY_BUCKETS) == sorted(LATENCY_BUCKETS)


def test_one_access_log_line_per_request(app, caplog):
    with TestClient(app) as client, caplog.at_level(logging.DEBUG, logger="recsys_api.access"):
        client.get("/movies/search", params={"q": "matrix"})
        client.get("/metrics")
    records = [r for r in caplog.records if r.name == "recsys_api.access"]
    search = [r for r in records if r.endpoint == "/movies/search"]
    assert len(search) == 1
    assert search[0].levelno == logging.INFO
    assert search[0].status == 200 and search[0].method == "GET"
    assert search[0].duration_ms >= 0
    assert "q=matrix" not in search[0].getMessage()  # no query strings (search terms) in logs
    scrape_line = [r for r in records if r.endpoint == "/metrics"]
    assert scrape_line and scrape_line[0].levelno == logging.DEBUG  # Prometheus every 15 s


def test_metrics_registry_is_reusable_directly():
    metrics = ApiMetrics()
    metrics.observe_request("/ready", 200, 0.01)
    assert (
        metrics.registry.get_sample_value(
            "http_requests_total", {"endpoint": "/ready", "status": "200"}
        )
        == 1
    )


# ----- review fixes (s5-review I-1, M-2) ---------------------------------------------------


class FailingRegistry(Registry):
    """MLflow unreachable: no model is ever loaded."""

    def load(self):
        raise ConnectionError("mlflow unreachable")


def test_model_timestamps_are_not_exported_as_epoch_zero_before_a_model_loads():
    """A gauge without a value would be exported as 0.0: `time() - 0` is ~56 years and fires
    "model older than 36 h" on every API restart (or for as long as MLflow is down)."""
    assert math.isnan(ApiMetrics().registry.get_sample_value("model_created_timestamp_seconds"))
    assert math.isnan(ApiMetrics().registry.get_sample_value("model_loaded_timestamp_seconds"))
    app, _ = make(registry=FailingRegistry())
    with TestClient(app) as client:
        metrics = scrape(client)
    assert math.isnan(value(metrics, "model_created_timestamp_seconds"))
    assert math.isnan(value(metrics, "model_loaded_timestamp_seconds"))


def test_error_series_of_every_route_exist_before_the_first_error(app):
    """rate() needs a sample before the increase: a 5xx series born with its first error
    would hide that error from the 5xx alert."""
    with TestClient(app) as client:
        metrics = scrape(client)
    total = "http_requests_total"
    for endpoint in ("/recommendations", "/ratings", "/movies/search", "/ready"):
        assert value(metrics, "http_requests", total, endpoint=endpoint, status="500") == 0
