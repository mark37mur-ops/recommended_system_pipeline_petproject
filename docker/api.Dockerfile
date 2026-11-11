# syntax=docker/dockerfile:1
# API image: the recsys package with its core dependencies only (mlflow-skinny client, no
# dev tools, no full MLflow server) + the FastAPI service. Also used for one-off jobs
# (MovieLens loader, training against the MLflow server) via the compose `tools` profile.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable

FROM python:3.12-slim
# libgomp1: OpenMP runtime for LightGBM and implicit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data && chown app /data
COPY --from=builder /opt/venv /opt/venv
COPY services/api /app
ARG GIT_SHA=unknown
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MODEL_DIR=/tmp/models \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet \
    FORWARDED_ALLOW_IPS=127.0.0.1 \
    GIT_SHA=$GIT_SHA
WORKDIR /app
USER app
EXPOSE 8000
# --no-access-log: the app writes one JSON line per request itself (route template, status,
# duration; spec 5.6), uvicorn's text access log would duplicate it with raw URLs.
CMD ["uvicorn", "--factory", "recsys_api.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-server-header", "--no-access-log"]
