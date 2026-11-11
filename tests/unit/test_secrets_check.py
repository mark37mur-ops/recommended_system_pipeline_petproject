"""Fail-fast secrets check (stage 4 review I3): the Airflow containers refuse to start with the
`.env.example` placeholders, short secrets or a malformed Fernet key."""

import base64
import os
from pathlib import Path

import pytest
import yaml

from recsys.secrets_check import (
    AIRFLOW_FERNET_KEY,
    AIRFLOW_SECRETS,
    MIN_SECRET_LENGTH,
    fernet_key_problem,
    main,
    secret_problem,
)

ROOT = Path(__file__).resolve().parents[2]
STRONG = "s" * MIN_SECRET_LENGTH
FERNET = base64.urlsafe_b64encode(os.urandom(32)).decode()


def _env_example() -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.startswith("#")
    )


@pytest.mark.parametrize(
    "value",
    ["change-me", "CHANGE-ME", "changeme", "change-me-" + "x" * MIN_SECRET_LENGTH, "<secret>"],
)
def test_placeholders_are_refused(value):
    assert "placeholder" in secret_problem("X", value)


@pytest.mark.parametrize("value", ["", "x" * (MIN_SECRET_LENGTH - 1), "admin-secret"])
def test_short_or_empty_secrets_are_refused(value):
    problem = secret_problem("X", value)
    assert problem is not None and "X" in problem


def test_strong_secret_is_accepted():
    assert secret_problem("X", STRONG) is None


@pytest.mark.parametrize(
    "value",
    [
        "change-me",
        "",
        "not base64 at all!",
        base64.urlsafe_b64encode(os.urandom(16)).decode(),  # valid base64, wrong length
    ],
)
def test_malformed_fernet_keys_are_refused(value):
    problem = fernet_key_problem("AIRFLOW_FERNET_KEY", value)
    assert problem is not None and "Fernet.generate_key" in problem


def test_valid_fernet_key_is_accepted():
    assert fernet_key_problem("AIRFLOW_FERNET_KEY", FERNET) is None


def _strong_env() -> dict[str, str]:
    return {name: STRONG for name in AIRFLOW_SECRETS} | {AIRFLOW_FERNET_KEY: FERNET}


def test_main_passes_with_strong_values(capsys):
    assert main([], environ=_strong_env()) == 0


def test_main_lists_every_problem_and_never_prints_values(capsys):
    env = _strong_env() | {"AIRFLOW_ADMIN_PASSWORD": "change-me", AIRFLOW_FERNET_KEY: "short"}
    del env["AIRFLOW_JWT_SECRET"]
    assert main([], environ=env) == 1
    err = capsys.readouterr().err
    for name in ("AIRFLOW_ADMIN_PASSWORD", "AIRFLOW_JWT_SECRET", AIRFLOW_FERNET_KEY):
        assert name in err
    assert "AIRFLOW_API_SECRET_KEY" not in err
    assert "short" not in err.replace("shorter", "")  # the secret itself is not echoed


def test_env_example_airflow_values_are_refused():
    """`cp .env.example .env && docker compose up` without editing must not start Airflow."""
    example = _env_example()
    for name in AIRFLOW_SECRETS:
        assert secret_problem(name, example[name]) is not None, name
    assert fernet_key_problem(AIRFLOW_FERNET_KEY, example[AIRFLOW_FERNET_KEY]) is not None
    assert main([], environ=example) == 1


def test_airflow_init_runs_the_check_before_migrating():
    """The check runs at container start (compose config must stay valid with .env.example,
    CI parses it), and every other Airflow container waits for airflow-init."""
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    init = services["airflow-init"]
    command = " ".join(init["command"])
    assert command.index("recsys.secrets_check") < command.index("airflow db migrate")
    assert set(AIRFLOW_SECRETS) | {AIRFLOW_FERNET_KEY} <= set(init["environment"])
    for name in ("airflow-apiserver", "airflow-scheduler", "airflow-dag-processor"):
        assert services[name]["depends_on"]["airflow-init"] == {
            "condition": "service_completed_successfully"
        }
