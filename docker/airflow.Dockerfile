# syntax=docker/dockerfile:1
# Airflow image for the pipelines (spec 4.1): official Airflow + the recsys package.
#
# Why two virtualenvs in one image: Airflow 3.3's constraints pin SQLAlchemy 2.0, pandas 3.0,
# pyarrow 25, while uv.lock (shared with the API image) has SQLAlchemy 2.1, pandas 3.1,
# pyarrow 26. Installing recsys into Airflow's environment would either break Airflow or force
# a second, different set of versions for training. So Airflow keeps its own environment and
# recsys gets /opt/recsys-venv, built from the very same uv.lock as the API: the model is
# trained with exactly the library versions it is served with. DAG tasks run there through
# @task.external_python. No DockerOperator: that would need the Docker socket (spec 4.1).

# Airflow 3.3.2 (2026-09-17), not 3.4.0 (5 days old at the time of writing). slim = core +
# a handful of providers, no 80 provider packages we do not use.
ARG AIRFLOW_IMAGE=apache/airflow:slim-3.3.2-python3.12

# ----- stage 1: the recsys virtualenv ---------------------------------------------------------
# Built on the Airflow image itself, so the venv points at the interpreter that exists in the
# final image (/usr/python/bin/python3.12), byte for byte the same Python.
FROM ${AIRFLOW_IMAGE} AS recsys-venv
USER root
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/python/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/recsys-venv \
    UV_CACHE_DIR=/root/.cache/uv \
    VIRTUAL_ENV=
WORKDIR /build
# Dependencies first (cached layer), then the package itself. `--group drift` adds Evidently
# for the drift_report task (stage 5); the API image does not need it and does not get it.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group drift --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group drift --no-editable

# ----- stage 2: Airflow + the venv ------------------------------------------------------------
FROM ${AIRFLOW_IMAGE}
ARG AIRFLOW_VERSION=3.3.2
USER root
# libgomp1: OpenMP runtime that implicit (ALS) and LightGBM need; the slim image lacks it.
# /data: mount point of the shared volume with run folders and the raw dataset. Created here
# and owned by airflow (uid 50000, gid 0), so a fresh named volume inherits that ownership.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data/runs /data/raw \
    && chown -R airflow:0 /data \
    && chmod -R g+rwX /data
COPY --from=recsys-venv /opt/recsys-venv /opt/recsys-venv
USER airflow
# The slim image has no Postgres driver for Airflow's own metadata DB: psycopg2 for the sync
# engine, asyncpg for the async one Airflow 3 opens as well. Installed with the official
# constraints, so pip cannot move any Airflow dependency.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" psycopg2-binary asyncpg \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt"
# DAGs are baked in, not bind-mounted: code and DAGs ship together under one image tag (CD,
# spec 6.4), and a Windows bind mount would be slow and come with odd permissions.
COPY --chown=airflow:0 airflow/dags /opt/airflow/dags
ARG GIT_SHA=unknown
# OPENBLAS/MKL=1: implicit parallelises with its own threads; nested BLAS threads would only
# oversubscribe the CPU (same as the API image). GIT_SHA ends up as a tag of every MLflow run.
ENV RECSYS_PYTHON=/opt/recsys-venv/bin/python \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet \
    RUNS_DIR=/data/runs \
    RAW_DIR=/data/raw \
    GIT_SHA=$GIT_SHA
