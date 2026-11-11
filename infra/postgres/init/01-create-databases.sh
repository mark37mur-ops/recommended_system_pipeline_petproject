#!/bin/sh
# One Postgres instance, one database (and role) per component (spec 1):
#   app    - movies, users, ratings (the API and the loader)
#   mlflow - MLflow backend store
#   airflow - stage 4 hook: created only when AIRFLOW_DB_PASSWORD is set
# Runs automatically on the first start with an empty data volume. Idempotent, so it can be
# re-run by hand on an existing volume (e.g. to add the airflow database in stage 4):
#   docker compose exec postgres sh /docker-entrypoint-initdb.d/01-create-databases.sh
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
