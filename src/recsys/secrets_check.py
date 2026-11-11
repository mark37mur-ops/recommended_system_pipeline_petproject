"""Refuse to start with weak secrets from `.env` (stage 4 review I3).

One policy for the API settings and the Airflow containers: a secret must not be a template
placeholder ("change-me" and friends) and must have at least MIN_SECRET_LENGTH characters.
The Fernet key must be what Airflow expects (urlsafe base64 of 32 bytes).

The compose service `airflow-init` runs this before `airflow db migrate`; every other Airflow
container waits for it to succeed, so `cp .env.example .env && docker compose up` without
editing brings Airflow down with the list of offending variables (never their values):
    python -m recsys.secrets_check            # Airflow (default)
    python -m recsys.secrets_check grafana    # Grafana (compose service grafana-init, stage 5)
It checks at container start, not at `docker compose config` time, so CI can still validate
the compose file with `.env.example`.
"""

import base64
import binascii
import os
import sys
from collections.abc import Mapping, Sequence

MIN_SECRET_LENGTH = 32
# Template values (`.env.example`) and the usual "fill me in" markers; never real secrets.
PLACEHOLDER_PREFIXES = ("change", "replace", "your-", "<")
GENERATE_HINT = 'python -c "import secrets; print(secrets.token_urlsafe(32))"'
FERNET_HINT = (
    'python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
)

# Names as in `.env` (compose passes them to airflow-init under the same names).
AIRFLOW_SECRETS = (
    "AIRFLOW_ADMIN_PASSWORD",
    "AIRFLOW_JWT_SECRET",
    "AIRFLOW_API_SECRET_KEY",
    "AIRFLOW_DB_PASSWORD",
)
AIRFLOW_FERNET_KEY = "AIRFLOW_FERNET_KEY"
# Stage 5: the Grafana admin login, the key Grafana encrypts data source passwords with in its
# own database (its built-in default is a publicly known string), and the read-only
# grafana_reader role of the app database.
GRAFANA_SECRETS = ("GRAFANA_ADMIN_PASSWORD", "GRAFANA_SECRET_KEY", "GRAFANA_DB_PASSWORD")


def secret_problem(name: str, value: str | None) -> str | None:
    """Why `value` is not acceptable as the secret `name`, or None if it is."""
    raw = (value or "").strip()
    if raw.lower().startswith(PLACEHOLDER_PREFIXES):
        return f"{name} is a placeholder; generate one with: {GENERATE_HINT}"
    if len(raw) < MIN_SECRET_LENGTH:
        return (
            f"{name} must be at least {MIN_SECRET_LENGTH} characters; "
            f"generate one with: {GENERATE_HINT}"
        )
    return None


def fernet_key_problem(name: str, value: str | None) -> str | None:
    """Same check as `cryptography.fernet.Fernet(key)`, without needing cryptography."""
    try:
        key = base64.urlsafe_b64decode((value or "").strip().encode())
    except (binascii.Error, ValueError):
        key = b""
    if len(key) != 32:
        return (
            f"{name} is not a Fernet key (urlsafe base64 of 32 bytes); "
            f"generate one with: {FERNET_HINT}"
        )
    return None


def airflow_problems(environ: Mapping[str, str]) -> list[str]:
    problems = [secret_problem(name, environ.get(name)) for name in AIRFLOW_SECRETS]
    problems.append(fernet_key_problem(AIRFLOW_FERNET_KEY, environ.get(AIRFLOW_FERNET_KEY)))
    return [p for p in problems if p is not None]


def grafana_problems(environ: Mapping[str, str]) -> list[str]:
    problems = [secret_problem(name, environ.get(name)) for name in GRAFANA_SECRETS]
    return [p for p in problems if p is not None]


GROUPS = {"airflow": airflow_problems, "grafana": grafana_problems}


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    argv = list(argv or [])
    group = argv[0] if argv else "airflow"
    if group not in GROUPS or len(argv) > 1:
        print(f"usage: python -m recsys.secrets_check [{'|'.join(GROUPS)}]", file=sys.stderr)
        return 2
    problems = GROUPS[group](os.environ if environ is None else environ)
    if not problems:
        print(f"secrets check ({group}): ok", file=sys.stderr)
        return 0
    print("refusing to start, fix these in .env:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
