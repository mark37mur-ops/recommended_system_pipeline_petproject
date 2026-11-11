"""Production config as code (stages 6-7), without Docker: what docker-compose.prod.yml,
the Caddyfile and the workflows promise. The real checks are `docker compose config`,
`caddy validate`, actionlint (CI job `infra`) and the local e2e; this pins the decisions a
later edit could quietly undo (a published port, /metrics through Caddy, a write token in
the deploy job)."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CADDYFILE = ROOT / "infra" / "caddy" / "Caddyfile"
WORKFLOWS = ROOT / ".github" / "workflows"


class ComposeLoader(yaml.SafeLoader):
    """SafeLoader that understands Compose's merge-control tags `!reset` and `!override`."""


class Tagged:
    def __init__(self, tag: str, value) -> None:
        self.tag, self.value = tag, value


def _tagged(loader, node):
    if isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    elif isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node, deep=True)
    else:
        value = loader.construct_scalar(node)
    return Tagged(node.tag, value)


ComposeLoader.add_constructor("!reset", _tagged)
ComposeLoader.add_constructor("!override", _tagged)


def load(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=ComposeLoader)  # noqa: S506


@pytest.fixture(scope="module")
def base() -> dict:
    return load(ROOT / "docker-compose.yml")["services"]


@pytest.fixture(scope="module")
def prod() -> dict:
    return load(ROOT / "docker-compose.prod.yml")["services"]


def is_reset(value) -> bool:
    return isinstance(value, Tagged) and value.tag == "!reset"


def test_every_service_is_covered_by_the_override(base, prod):
    assert set(base) <= set(prod)
    assert "caddy" in prod


def test_only_caddy_is_published_on_all_interfaces(base, prod):
    for name, service in base.items():
        override = prod.get(name, {})
        if is_reset(override.get("ports")):
            continue
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name} publishes {port} beyond loopback"
    published = {name for name, s in prod.items() if "ports" in s and not is_reset(s["ports"])}
    assert published == {"caddy"}
    # The databases and the API publish nothing at all in production.
    assert all(is_reset(prod[name]["ports"]) for name in ("postgres", "redis", "api"))


def test_our_images_come_from_ghcr_and_are_never_built(base, prod):
    ours = {name for name, s in base.items() if "build" in s}
    assert ours  # api, airflow-*, mlflow, grafana-init, grafana-publish, loader, trainer
    for name in ours:
        assert is_reset(prod[name].get("build")), f"{name} would still be built on the server"
        image = prod[name]["image"]
        assert image.startswith("${IMAGE_PREFIX:?") and ":${IMAGE_TAG:?" in image, name


def test_logs_rotate_everywhere_and_long_running_services_have_memory_limits(base, prod):
    for name, service in prod.items():
        options = service["logging"]["options"]
        assert service["logging"]["driver"] == "json-file"
        assert options["max-size"] and options["max-file"], name
    long_running = {n for n, s in base.items() if s.get("restart") == "unless-stopped"}
    for name in long_running | {"caddy"}:
        assert prod[name]["deploy"]["resources"]["limits"]["memory"], name


def test_proxy_trust_and_https_settings(prod):
    api = prod["api"]["environment"]
    assert api["COOKIE_SECURE"] == "true"
    assert api["FORWARDED_ALLOW_IPS"] == "${EDGE_SUBNET:-10.203.17.0/24}"
    assert prod["caddy"]["networks"] == ["edge"]
    assert set(prod["api"]["networks"]) == set(prod["grafana"]["networks"]) == {"default", "edge"}
    grafana = prod["grafana"]["environment"]
    assert grafana["GF_SERVER_SERVE_FROM_SUB_PATH"] == "true"
    assert grafana["GF_SERVER_ROOT_URL"].endswith("/grafana/")
    assert "/grafana/api/health" in " ".join(prod["grafana"]["healthcheck"]["test"].value)


def caddy_routes() -> list[tuple[str, str]]:
    """(matcher paths, what the handle block does) in file order."""
    text = CADDYFILE.read_text(encoding="utf-8")
    matchers = dict(re.findall(r"^\s*(@\w+) path (.+)$", text, flags=re.M))
    routes = []
    for name, body in re.findall(r"^\s*handle (@\w+)? ?\{\n(.*?)\n\s*\}", text, re.M | re.S):
        routes.append((matchers.get(name, "*") if name else "*", body.strip()))
    return routes


def test_caddy_blocks_internal_paths_before_proxying_to_the_api():
    routes = caddy_routes()
    assert routes[0] == ("/metrics* /admin*", "respond 404")
    assert routes[1] == (
        "/grafana/public-dashboards/* /grafana/api/public/* /grafana/public/*",
        "reverse_proxy grafana:3000",
    )
    assert routes[2] == ("/grafana /grafana/*", "respond 404")
    assert routes[-1] == ("*", "reverse_proxy api:8000")
    config = CADDYFILE.read_text(encoding="utf-8").splitlines()
    config = [line for line in config if not line.lstrip().startswith("#")]
    assert "{$SITE_ADDRESS} {" in config
    # Caddy must not believe a client's own X-Forwarded-For (it would forge its IP).
    assert not any("trusted_proxies" in line for line in config)


def workflow(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))  # YAML 1.1 reads the key `on` as True
    return data


def test_publish_runs_after_every_check_and_only_it_writes_packages():
    ci = workflow("ci.yml")
    jobs = ci["jobs"]
    publish = jobs["publish"]
    assert set(publish["needs"]) == set(jobs) - {"publish"}
    assert "refs/heads/main" in publish["if"] and "push" in publish["if"]
    assert publish["permissions"]["packages"] == "write"
    assert ci["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for name, job in jobs.items() if name != "publish")
    images = {entry["image"] for entry in publish["strategy"]["matrix"]["include"]}
    assert images == {"api", "airflow", "mlflow"}
    names = next(s for s in publish["steps"] if s.get("id") == "names")["run"]
    assert "${GITHUB_REPOSITORY,,}" in names  # GHCR needs lower case


def test_deploy_is_inert_without_secrets_and_pins_the_host_key():
    deploy = workflow("deploy.yml")
    assert deploy["on"]["workflow_run"]["workflows"] == ["CI"]
    assert deploy["permissions"] == {"contents": "read", "packages": "read"}
    steps = deploy["jobs"]["deploy"]["steps"]
    assert steps[0]["id"] == "server"
    assert "deploy skipped: server not configured" in steps[0]["run"]
    assert all("steps.server.outputs.configured == 'true'" in s["if"] for s in steps[1:])
    scripts = "\n".join(s.get("run", "") for s in steps)
    assert "StrictHostKeyChecking yes" in scripts
    assert re.search(r"StrictHostKeyChecking[ =]no", scripts) is None
    assert "${{" not in scripts  # inputs reach the shell through env only (no injection)
