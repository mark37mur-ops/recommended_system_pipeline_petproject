"""Service configuration from environment variables (`.env` in compose, see `.env.example`)."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# One secrets policy for the API and the Airflow containers (recsys.secrets_check).
from recsys.secrets_check import MIN_SECRET_LENGTH, secret_problem
from recsys.tracking import MODEL_NAME

__all__ = ["MIN_SECRET_LENGTH", "Settings"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", protected_namespaces=())

    database_url: str = "postgresql+psycopg://app:app@localhost:5432/app"
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = MODEL_NAME
    # Where champion artifacts are downloaded before loading (removed right after).
    model_dir: Path = Path(".model-cache")
    # POST /admin/reload-model needs `Authorization: Bearer <token>`; unset or empty ->
    # endpoint off (503).
    admin_token: SecretStr | None = None
    # Signs the anonymous-user cookie; unset or empty -> random per process (cookies die on
    # restart).
    session_secret: SecretStr | None = None
    cookie_secure: bool = False  # true behind HTTPS (stage 6)
    cache_ttl_seconds: int = 600
    rate_limit_per_minute: int = 30  # POST /ratings, per IP and per visitor
    # GET /recommendations, per IP and per visitor: a person clicking around never gets near it
    recommendations_rate_limit_per_minute: int = 120
    max_ratings_per_user: int = 300
    onboarding_size: int = 20
    model_retry_seconds: float = 30.0  # startup: retry MLflow / Postgres this often
    log_level: str = "INFO"
    # json: one JSON object per line on stdout (spec 5.6, what Docker collects); text: humans.
    log_format: Literal["json", "text"] = "json"
    # Write every shown recommendation to recommendation_impressions (spec 5.2), off the
    # request path. Off: no rows, no online metric.
    impressions_enabled: bool = True
    # A (visitor, path, model version, movie) logged within this many seconds is not logged
    # again: the online metric only uses its first impression (s5-review I-4). 0 = log all.
    impressions_dedup_seconds: float = 3600.0

    @field_validator("admin_token", "session_secret", mode="before")
    @classmethod
    def _empty_is_unset(cls, value):
        """`ADMIN_TOKEN=` in an env file means "not configured", never "the empty token"."""
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        return value

    @field_validator("admin_token", "session_secret")
    @classmethod
    def _strong_secret(cls, value: SecretStr | None, info: ValidationInfo) -> SecretStr | None:
        """Refuse to start with a template placeholder or a short secret: either one lets
        anybody forge visitor cookies or call the admin endpoint."""
        if value is None:
            return None
        problem = secret_problem(info.field_name.upper(), value.get_secret_value())
        if problem is not None:
            raise ValueError(problem)
        return value
