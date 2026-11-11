"""bootstrap-server.sh's generate_env: the server .env made from .env.example passes the same
secret policy the containers enforce at start (recsys.secrets_check), keeps the template's
structure, sets the production values and never overwrites an existing .env."""

import subprocess
from pathlib import Path

import pytest
from shell_fakes import ROOT, SCRIPTS, find_bash, posix

from recsys.secrets_check import (
    AIRFLOW_FERNET_KEY,
    GRAFANA_SECRETS,
    airflow_problems,
    grafana_problems,
    secret_problem,
)

BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash (Git's bash on Windows)")
EXAMPLE = ROOT / ".env.example"


def generate(target: Path, site: str = "203.0.113.7.sslip.io") -> subprocess.CompletedProcess:
    script = posix(SCRIPTS / "bootstrap-server.sh")
    # Sourcing defines the functions without running main (which needs root).
    command = f'source "{script}" && generate_env "$1" "$2" "$3"'
    return subprocess.run(
        [BASH, "-c", command, "bash", posix(EXAMPLE), posix(target), site],
        capture_output=True,
        text=True,
        timeout=60,
    )


def parse(path: Path) -> dict[str, str]:
    pairs = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            pairs[key] = value
    return pairs


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    target = tmp_path_factory.mktemp("bootstrap") / ".env"
    result = generate(target)
    assert result.returncode == 0, result.stdout + result.stderr
    return target


def test_secrets_pass_the_container_policy(generated):
    env = parse(generated)
    assert airflow_problems(env) == []
    assert grafana_problems(env) == []
    for name in ("ADMIN_TOKEN", "SESSION_SECRET", "POSTGRES_PASSWORD", "APP_DB_PASSWORD"):
        assert secret_problem(name, env[name]) is None
    assert len(env[AIRFLOW_FERNET_KEY]) == 44 and env[AIRFLOW_FERNET_KEY].endswith("=")


def test_every_placeholder_is_replaced_by_a_distinct_url_safe_value(generated):
    template = parse(EXAMPLE)
    env = parse(generated)
    placeholders = [key for key, value in template.items() if value == "change-me"]
    assert set(GRAFANA_SECRETS) <= set(placeholders)
    values = [env[key] for key in placeholders]
    assert len(set(values)) == len(values)
    url_safe = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=")
    assert all(set(value) <= url_safe for value in values)
    assert all("=" not in env[k] for k in placeholders if k != AIRFLOW_FERNET_KEY)


def test_production_values_and_template_structure(generated):
    env = parse(generated)
    assert env["SITE_ADDRESS"] == "203.0.113.7.sslip.io"
    assert env["COOKIE_SECURE"] == "true"
    assert env["TELEGRAM_BOT_TOKEN"] == ""  # optional values stay as in the template
    assert env["IMAGE_PREFIX"] == parse(EXAMPLE)["IMAGE_PREFIX"]
    template_lines = EXAMPLE.read_text().splitlines()
    lines = generated.read_text().splitlines()
    assert len(lines) == len(template_lines)
    for mine, theirs in zip(lines, template_lines, strict=True):
        if theirs.startswith("#") or "=" not in theirs:
            assert mine == theirs  # comments survive for the admin reading .env later
        else:
            assert mine.split("=", 1)[0] == theirs.split("=", 1)[0]


def test_existing_env_is_never_overwritten(tmp_path):
    target = tmp_path / ".env"
    target.write_text("ADMIN_TOKEN=keep-me\n")

    result = generate(target)

    assert result.returncode == 0
    assert target.read_text() == "ADMIN_TOKEN=keep-me\n"
    assert "left as it is" in result.stderr
