"""Secrets in `Settings` (review I1, I2): empty = not configured, weak values refuse to start."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys_api.settings import MIN_SECRET_LENGTH, Settings

STRONG = "s" * MIN_SECRET_LENGTH


@pytest.mark.parametrize("field", ["admin_token", "session_secret"])
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_secret_means_not_configured(field, value):
    assert getattr(Settings(**{field: value}), field) is None


@pytest.mark.parametrize("field", ["admin_token", "session_secret"])
@pytest.mark.parametrize(
    "value",
    [
        "change-me",
        "CHANGE-ME",
        "changeme",
        "change-me-" + "x" * MIN_SECRET_LENGTH,  # long, but still the template placeholder
        "x" * (MIN_SECRET_LENGTH - 1),
        "admin-secret",
    ],
)
def test_placeholder_or_short_secret_refuses_to_start(field, value):
    with pytest.raises(ValidationError, match=field.upper()):
        Settings(**{field: value})


def test_weak_secret_from_environment_refuses_to_start(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "change-me")
    monkeypatch.setenv("SESSION_SECRET", STRONG)
    with pytest.raises(ValidationError, match="token_urlsafe"):
        Settings()


def test_strong_secrets_are_accepted():
    settings = Settings(admin_token=STRONG, session_secret=STRONG)
    assert settings.admin_token.get_secret_value() == STRONG
    assert settings.session_secret.get_secret_value() == STRONG


def test_env_example_placeholders_are_refused():
    """`cp .env.example .env && docker compose up` without editing must not start the API."""
    example = Path(__file__).parents[2] / ".env.example"
    values = dict(
        line.split("=", 1)
        for line in example.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.startswith("#")
    )
    for field in ("admin_token", "session_secret"):
        with pytest.raises(ValidationError, match=field.upper()):
            Settings(**{field: values[field.upper()]})
