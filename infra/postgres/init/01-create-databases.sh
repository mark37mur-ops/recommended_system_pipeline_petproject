#!/bin/sh
# One Postgres instance, one database (and role) per component (spec 1):
#   app    - movies, users, ratings (the API and the loader)
#   mlflow - MLflow backend store
#   airflow - Airflow metadata (stage 4); skipped when AIRFLOW_DB_PASSWORD is empty
# Runs automatically on the first start with an empty data volume. Idempotent: the compose
# job `db-init` runs it again against the running server on every `docker compose up`
# (PGHOST=postgres), which adds databases missing on an existing volume and re-applies the
# role passwords from .env.
set -eu

create_db() {
  name="$1"
  password="$2"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
       -v name="$name" -v password="$password" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'name', :'password')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'name') \gexec
SELECT format('ALTER ROLE %I PASSWORD %L', :'name', :'password') \gexec
SELECT format('CREATE DATABASE %I OWNER %I', :'name', :'name')
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'name') \gexec
SQL
}

create_db app "$APP_DB_PASSWORD"
create_db mlflow "$MLFLOW_DB_PASSWORD"
if [ -n "${AIRFLOW_DB_PASSWORD:-}" ]; then
  create_db airflow "$AIRFLOW_DB_PASSWORD"
fi
