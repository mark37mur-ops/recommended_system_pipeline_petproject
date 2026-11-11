"""Monitoring config as code (stage 5): Prometheus, Grafana provisioning, the dashboard, the alert
rules and the public-dashboard job parse, point at things that exist and match spec 5.4/5.5.
The real check is the e2e run; this catches typos (a metric renamed in the API but not in a
panel, a wrong data source uid) without Docker."""

import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

from recsys.db.schema import DASHBOARD_OBJECTS
from recsys.secrets_check import GRAFANA_SECRETS, main, secret_problem
from recsys_api.metrics import ApiMetrics

ROOT = Path(__file__).resolve().parents[2]
GRAFANA = ROOT / "infra" / "grafana"
PROVISIONING = GRAFANA / "provisioning"


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads((GRAFANA / "dashboards" / "recsys.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rules() -> list[dict]:
    groups = load_yaml(PROVISIONING / "alerting" / "rules.yml")["groups"]
    return [rule for group in groups for rule in group["rules"]]


@pytest.fixture(scope="module")
def datasources() -> dict[str, dict]:
    config = load_yaml(PROVISIONING / "datasources" / "datasources.yml")
    return {ds["uid"]: ds for ds in config["datasources"]}


def api_metric_names() -> set[str]:
    """Every sample name the API exports (counters with _total, histogram series)."""
    names = set()
    for family in ApiMetrics().registry.collect():
        names |= {sample.name for sample in family.samples}
        names.add(family.name)
        if family.type == "counter":  # a labelled counter has no samples before first use
            names.add(f"{family.name}_total")
        if family.type == "histogram":
            names |= {f"{family.name}_{s}" for s in ("bucket", "count", "sum")}
    return names


METRIC_RE = re.compile(r"\b([a-z][a-z0-9_]*(?:_total|_seconds|_bucket|_info))\b")


def promql_expressions(dashboard, rules) -> list[str]:
    exprs = [
        t["expr"]
        for p in dashboard["panels"]
        for t in p["targets"]
        if t["datasource"]["uid"] == "prometheus"
    ]
    exprs += [
        q["model"]["expr"] for r in rules for q in r["data"] if q["datasourceUid"] == "prometheus"
    ]
    return exprs


# ----- Prometheus ------------------------------------------------------------------------------


def test_prometheus_scrapes_the_api_and_itself():
    config = load_yaml(ROOT / "infra" / "prometheus" / "prometheus.yml")
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert set(jobs) == {"api", "prometheus"}
    assert jobs["api"]["metrics_path"] == "/metrics"
    assert jobs["api"]["static_configs"][0]["targets"] == ["api:8000"]
    assert config["global"]["scrape_interval"] == "15s"


# ----- data sources and dashboard --------------------------------------------------------------


def test_datasources(datasources):
    assert set(datasources) == {"prometheus", "app-postgres"}
    pg = datasources["app-postgres"]
    assert pg["type"] == "grafana-postgresql-datasource"
    assert pg["user"] == "grafana_reader" and pg["jsonData"]["database"] == "app"
    assert pg["secureJsonData"]["password"] == "${GRAFANA_DB_PASSWORD}"  # from env, not git
    assert datasources["prometheus"]["url"] == "http://prometheus:9090"


def test_dashboard_provider_reads_the_mounted_folder():
    provider = load_yaml(PROVISIONING / "dashboards" / "dashboards.yml")["providers"][0]
    assert provider["options"]["path"] == "/var/lib/grafana-dashboards"
    assert provider["allowUiUpdates"] is False


def test_dashboard_has_the_spec_panels(dashboard):
    assert dashboard["uid"] == "recsys-overview"
    titles = " | ".join(p["title"] for p in dashboard["panels"]).lower()
    # spec 5.4: RPS, p95, error share, fallback share, cache hit rate, ratings per hour by
    # source, model version and age, online metric, drift_share, NDCG@10 of the last runs
    for needle in (
        "requests per second",
        "p95",
        "error share",
        "by path",
        "cache hit rate",
        "ratings per hour by source",
        "model version",
        "model age",
        "online",
        "drift_share",
        "ndcg@10",
    ):
        assert needle in titles, needle
    assert not dashboard["templating"]["list"]  # public dashboards do not support variables


def test_every_panel_uses_a_provisioned_datasource(dashboard, datasources):
    ids = [p["id"] for p in dashboard["panels"]]
    assert len(ids) == len(set(ids))
    for panel in dashboard["panels"]:
        assert panel["datasource"]["uid"] in datasources, panel["title"]
        for target in panel["targets"]:
            assert target["datasource"]["uid"] in datasources, panel["title"]


def test_panels_do_not_overlap(dashboard):
    cells = set()
    for panel in dashboard["panels"]:
        g = panel["gridPos"]
        assert g["x"] + g["w"] <= 24
        mine = {
            (x, y) for x in range(g["x"], g["x"] + g["w"]) for y in range(g["y"], g["y"] + g["h"])
        }
        assert not mine & cells, panel["title"]
        cells |= mine


def test_dashboard_sql_reads_only_objects_granted_to_grafana(dashboard):
    sources = set()
    for panel in dashboard["panels"]:
        for target in panel["targets"]:
            if "rawSql" in target:
                sources |= set(re.findall(r"\bFROM\s+([a-z_]+)", target["rawSql"], re.I))
    assert sources and sources <= set(DASHBOARD_OBJECTS)


def test_every_metric_in_panels_and_alerts_is_exported_by_the_api(dashboard, rules):
    exported = api_metric_names()
    used = {m for expr in promql_expressions(dashboard, rules) for m in METRIC_RE.findall(expr)}
    assert used, "no metric names found"
    assert used <= exported, used - exported


def test_promql_selects_the_api_job(dashboard, rules):
    for expr in promql_expressions(dashboard, rules):
        assert 'job="api"' in expr, expr


# ----- alerts ----------------------------------------------------------------------------------


def _threshold(rule):
    [c] = [q for q in rule["data"] if q["refId"] == rule["condition"]]
    evaluator = c["model"]["conditions"][0]["evaluator"]
    return evaluator["type"], evaluator["params"][0]


def _expr(rule):
    return next(q["model"]["expr"] for q in rule["data"] if q["datasourceUid"] == "prometheus")


def test_spec_alert_rules(rules):
    by_uid = {r["uid"]: r for r in rules}
    assert set(by_uid) == {
        "recsys-api-down",
        "recsys-api-5xx",
        "recsys-api-latency",
        "recsys-fallback-share",
        "recsys-model-age",
    }
    down = by_uid["recsys-api-down"]
    assert _expr(down) == 'up{job="api"}' and _threshold(down) == ("lt", 1)
    assert down["for"] == "2m" and down["noDataState"] == "Alerting"

    assert _threshold(by_uid["recsys-api-5xx"]) == ("gt", 0.05)
    assert "[5m]" in _expr(by_uid["recsys-api-5xx"])
    assert _threshold(by_uid["recsys-api-latency"]) == ("gt", 0.3)
    assert "histogram_quantile(0.95" in _expr(by_uid["recsys-api-latency"])
    assert "[10m]" in _expr(by_uid["recsys-api-latency"])
    assert _threshold(by_uid["recsys-fallback-share"]) == ("gt", 0.2)
    assert "[10m]" in _expr(by_uid["recsys-fallback-share"])
    assert _threshold(by_uid["recsys-model-age"]) == ("gt", 36 * 3600)
    assert "model_created_timestamp_seconds" in _expr(by_uid["recsys-model-age"])


def test_alert_queries_use_known_datasources(rules, datasources):
    for rule in rules:
        for query in rule["data"]:
            assert query["datasourceUid"] in set(datasources) | {"__expr__"}


def test_telegram_is_optional_and_outside_the_provisioning_folder():
    config = load_yaml(GRAFANA / "telegram.yml")
    [contact] = config["contactPoints"]
    [receiver] = contact["receivers"]
    assert receiver["type"] == "telegram"
    assert receiver["settings"] == {
        "bottoken": "${TELEGRAM_BOT_TOKEN}",
        "chatid": "${TELEGRAM_CHAT_ID}",
    }
    assert config["policies"][0]["receiver"] == contact["name"]
    provisioned = [p.name for p in PROVISIONING.rglob("*.yml")]
    assert "telegram.yml" not in provisioned  # Grafana would refuse an empty bot token
    start = (GRAFANA / "start.sh").read_text(encoding="utf-8")
    assert '[ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]' in start
    assert "exec /run.sh" in start
    # without Telegram, a contact point provisioned on an earlier start is removed again
    off = load_yaml(GRAFANA / "telegram-off.yml")
    assert off["deleteContactPoints"] == [{"orgId": 1, "uid": receiver["uid"]}]
    assert off["resetPolicies"] == [1]
    assert "telegram-off.yml" in start


# ----- compose ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def services() -> dict:
    return load_yaml(ROOT / "docker-compose.yml")["services"]


def test_monitoring_ports_are_local_only(services):
    for name in ("prometheus", "grafana", "api"):
        for port in services[name]["ports"]:
            assert port.startswith("127.0.0.1:"), (name, port)


def test_grafana_waits_for_its_secrets_check_and_the_reader_role(services):
    grafana = services["grafana"]
    assert grafana["depends_on"]["grafana-init"] == {"condition": "service_completed_successfully"}
    assert grafana["depends_on"]["db-init"] == {"condition": "service_completed_successfully"}
    assert services["grafana-init"]["command"] == [
        "python",
        "-m",
        "recsys.secrets_check",
        "grafana",
    ]
    assert set(GRAFANA_SECRETS) <= set(services["grafana-init"]["environment"])
    env = grafana["environment"]
    assert env["GF_AUTH_ANONYMOUS_ENABLED"] == "false"
    assert env["GF_PLUGINS_PREINSTALL_DISABLED"] == "true"
    assert services["api"]["depends_on"]["db-init"] == {
        "condition": "service_completed_successfully"
    }


def test_init_script_creates_a_read_only_reader():
    script = (ROOT / "infra" / "postgres" / "init" / "01-create-databases.sh").read_text(
        encoding="utf-8"
    )
    assert "create_reader grafana_reader" in script
    assert "default_transaction_read_only = on" in script
    assert "statement_timeout" in script


# ----- secrets ---------------------------------------------------------------------------------


def test_grafana_secrets_group():
    strong = {name: "g" * 40 for name in GRAFANA_SECRETS}
    assert main(["grafana"], environ=strong) == 0
    assert main(["grafana"], environ=strong | {"GRAFANA_ADMIN_PASSWORD": "admin"}) == 1
    assert main(["nope"], environ=strong) == 2


def test_env_example_grafana_values_are_refused():
    example = dict(
        line.split("=", 1)
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.startswith("#")
    )
    for name in GRAFANA_SECRETS:
        assert secret_problem(name, example[name]) is not None, name
    assert example["TELEGRAM_BOT_TOKEN"] == "" and example["TELEGRAM_CHAT_ID"] == ""


# ----- public dashboard job --------------------------------------------------------------------


@pytest.fixture(scope="module")
def publish():
    spec = importlib.util.spec_from_file_location(
        "publish_dashboard", GRAFANA / "publish_dashboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGrafana:
    def __init__(self, existing=None):
        self.public = existing
        self.calls = []

    def __call__(self, method, path, body):
        self.calls.append((method, path))
        if method == "GET":
            if self.public is None:
                return 404, {"message": "public dashboard not found"}
            return 200, dict(self.public)
        if method == "POST":
            self.public = {"uid": "pd1", "accessToken": "a" * 32} | body
            return 200, dict(self.public)
        if method == "PATCH":
            self.public |= body
            return 200, dict(self.public)
        raise AssertionError(method)


def test_publish_creates_enables_or_leaves_alone(publish):
    api = FakeGrafana()
    outcome, config = publish.ensure_public_dashboard(api, "recsys-overview")
    assert outcome == "created" and config["isEnabled"] and config["share"] == "public"
    assert publish.ensure_public_dashboard(api, "recsys-overview")[0] == "unchanged"
    api.public["isEnabled"] = False
    outcome, config = publish.ensure_public_dashboard(api, "recsys-overview")
    assert outcome == "enabled" and api.public["isEnabled"] is True
    assert ("PATCH", "/api/dashboards/uid/recsys-overview/public-dashboards/pd1") in api.calls


def test_publish_reports_api_errors(publish):
    def broken(method, path, body):
        return 500, {"message": "boom"}

    with pytest.raises(RuntimeError, match="HTTP 500"):
        publish.ensure_public_dashboard(broken, "recsys-overview")
