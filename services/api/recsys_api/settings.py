"""Service configuration from environment variables (`.env` in compose, see `.env.example`)."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from recsys.tracking import MODEL_NAME


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", protected_namespaces=())

    database_url: str = "postgresql+psycopg://app:app@localhost:5432/app"
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = MODEL_NAME
    # Where champion artifacts are downloaded before loading (removed right after).
    model_dir: Path = Path(".model-cache")
    # POST /admin/reload-model needs `Authorization: Bearer <token>`; unset -> endpoint off.
    admin_token: SecretStr | None = None
    # Signs the anonymous-user cookie; unset -> random per process (cookies die on restart).
    session_secret: SecretStr | None = None
    cookie_secure: bool = False  # true behind HTTPS (stage 6)
    cache_ttl_seconds: int = 600
    rate_limit_per_minute: int = 30
    max_ratings_per_user: int = 300
    onboarding_size: int = 20
    model_retry_seconds: float = 30.0  # startup: retry MLflow / Postgres this often
    log_level: str = "INFO"
