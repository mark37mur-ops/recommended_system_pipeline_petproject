"""Service configuration from environment variables (`.env` in compose, see `.env.example`)."""

from pathlib import Path

from pydantic import SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from recsys.tracking import MODEL_NAME

MIN_SECRET_LENGTH = 32
# Template values (`.env.example`) and the usual "fill me in" markers; never real secrets.
PLACEHOLDER_PREFIXES = ("change", "replace", "your-", "<")
GENERATE_HINT = 'python -c "import secrets; print(secrets.token_urlsafe(32))"'


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
    rate_limit_per_minute: int = 30
    max_ratings_per_user: int = 300
    onboarding_size: int = 20
    model_retry_seconds: float = 30.0  # startup: retry MLflow / Postgres this often
    log_level: str = "INFO"

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
        raw = value.get_secret_value()
        name = info.field_name.upper()
        if raw.strip().lower().startswith(PLACEHOLDER_PREFIXES):
            raise ValueError(f"{name} is a placeholder; generate one with: {GENERATE_HINT}")
        if len(raw) < MIN_SECRET_LENGTH:
            raise ValueError(
                f"{name} must be at least {MIN_SECRET_LENGTH} characters; "
                f"generate one with: {GENERATE_HINT}"
            )
        return value
