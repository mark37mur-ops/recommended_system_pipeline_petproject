# syntax=docker/dockerfile:1
# MLflow tracking server + Model Registry with a Postgres backend store and proxied artifact
# storage (--serve-artifacts). Same MLflow version as uv.lock, plus the psycopg 3 driver.
FROM python:3.12-slim
ARG MLFLOW_VERSION=3.18.0
ARG PSYCOPG_VERSION=3.3.8
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" "psycopg[binary]==${PSYCOPG_VERSION}" \
    && useradd --create-home --uid 10001 mlflow \
    && mkdir -p /mlartifacts && chown mlflow /mlartifacts
USER mlflow
EXPOSE 5000
