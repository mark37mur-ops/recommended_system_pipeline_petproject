#!/bin/sh
# One Postgres instance, one database (and role) per component (spec 1):
#   app    - movies, users, ratings (the API and the loader)
#   mlflow - MLflow backend store
#   airflow - Airflow metadata (stage 4). Compose always sets AIRFLOW_DB_PASSWORD (`:?`);
#             the "skip when empty" branch below is only for running this script by hand.
# plus one role without a database of its own:
#   grafana_reader - Grafana's read-only view of `app` (stage 5). It may only connect to `app`
#             (CONNECT and TEMPORARY are revoked from PUBLIC on app, mlflow and airflow),
#             every transaction is read-only and every statement stops after 5 s (a heavy
#             dashboard query cannot hurt the API). Which tables it may SELECT is decided by
#             the app's schema code (recsys.db.schema.DASHBOARD_OBJECTS): those objects are
#             created by the API after this script, so the API grants them at start-up.
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

create_reader() {
  name="$1"
  password="$2"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
       -v name="$name" -v password="$password" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'name', :'password')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'name') \gexec
SELECT format('ALTER ROLE %I LOGIN PASSWORD %L', :'name', :'password') \gexec
SELECT format('ALTER ROLE %I SET default_transaction_read_only = on', :'name') \gexec
SELECT format('ALTER ROLE %I SET statement_timeout = %L', :'name', '5s') \gexec
SELECT format('GRANT CONNECT ON DATABASE app TO %I', :'name') \gexec
SQL
}

# Role defaults (read-only, timeout) can be overridden by the session itself; the real fence
# is privileges. PUBLIC (every role) may by default CONNECT to every database and create
# TEMPORARY tables: take that away, so the reader reaches only `app` (granted above) and
# creates nothing there. The owners (app, mlflow, airflow) keep all rights on their own
# databases. Idempotent, re-applied by db-init on every `docker compose up`.
lock_down_public() {
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<'SQL'
SELECT format('REVOKE CONNECT, TEMPORARY ON DATABASE %I FROM PUBLIC', datname)
FROM pg_database WHERE datname IN ('app', 'mlflow', 'airflow') \gexec
SQL
}

if [ -n "${GRAFANA_DB_PASSWORD:-}" ]; then
  create_reader grafana_reader "$GRAFANA_DB_PASSWORD"
fi
lock_down_public
