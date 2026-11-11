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
    """(matcher, what the handle block does) in file order: "path /a /b" or "*" for the
    catch-all, and the block's body with comments and indentation removed."""
    lines = [
        line.strip()
        for line in CADDYFILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    matchers = {}
    for line in lines:
        if found := re.match(r"(@\w+) (\w+ .+)$", line):
            matchers[found[1]] = found[2]
    routes, i = [], 0
    while i < len(lines):
        found = re.fullmatch(r"handle( @\w+)? \{", lines[i])
        if not found:
            i += 1
            continue
        depth, body, i = 1, [], i + 1
        while depth:
            depth += lines[i].count("{") - lines[i].count("}")
            if depth:
                body.append(lines[i])
            i += 1
        name = (found[1] or "").strip()
        routes.append((matchers[name] if name else "*", "\n".join(body)))
    return routes


def test_caddy_blocks_internal_paths_before_proxying_to_the_api():
    routes = caddy_routes()
    # 0: a raw path with `..` or percent-escaped . / \ % never reaches an upstream.
    matcher, action = routes[0]
    assert matcher.startswith("vars_regexp {http.request.orig_uri} ") and action == "respond 400"
    pattern = re.compile(matcher.split("`")[1])
    for bad in ("/grafana/api/health/../public/x", "/static/%2e%2e/metrics", "/a%2Fb", "/a%5cb"):
        assert pattern.search(bad), bad
    for good in ("/", "/movies/search?q=..%2F", "/grafana/public/build/app.1a2b.js"):
        assert not pattern.search(good), good
    assert routes[1] == ("path /metrics* /admin*", "respond 404")
    assert routes[2] == (
        "path /grafana/public-dashboards/* /grafana/api/public/* /grafana/public/*",
        "reverse_proxy grafana:3000",
    )
    assert routes[3] == ("path /grafana /grafana/*", "respond 404")
    # The public /ready answers the API's status code with one word, never the API's body.
    matcher, action = routes[4]
    assert matcher == "path /ready"
    assert action.startswith("reverse_proxy api:8000 {")
    assert 'respond `{"status":"ready"}` 200' in action
    assert 'respond `{"status":"degraded"}` 503' in action
    assert routes[-1] == ("*", "reverse_proxy api:8000")
    assert len(routes) == 6
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


def test_deploy_runs_detached_only_for_the_tip_of_main_and_rolls_back_only_by_hand():
    """s7-review I-1/I-2: the SSH session only starts and polls the deploy; a CI run whose
    commit is no longer the tip of main is skipped; only the manual run may go back."""
    job = workflow("deploy.yml")["jobs"]["deploy"]
    steps = {s["name"]: s for s in job["steps"]}
    version = steps["Version to deploy"]
    assert "commits/main" in version["run"] and "skip=true" in version["run"]
    assert version["env"]["GH_TOKEN"] == "${{ github.token }}"
    gated = [s for s in job["steps"] if s.get("id") not in ("server", "version")]
    assert all("steps.version.outputs.skip == 'false'" in s["if"] for s in gated)
    deploy = next(s for s in job["steps"] if s["name"].startswith("Deploy"))["run"]
    assert "deploy.sh --detach '$id'" in deploy
    assert ".deploy/$id.status" in deploy and '"done 0")' in deploy and '"done 3")' in deploy
    # --allow-older is a constant chosen by the event, never text from the input.
    assert '[ "$GITHUB_EVENT_NAME" != workflow_dispatch ] || flags="--allow-older"' in deploy
    assert job["timeout-minutes"] > 30  # longer than the 30-minute polling deadline
