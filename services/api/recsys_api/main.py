"""FastAPI application (spec 3.1). Run with `uvicorn --factory recsys_api.main:create_app`.

Startup never blocks on MLflow or Postgres: the app serves immediately (popularity /
degraded `/ready`) while a background thread creates the schema, loads the movie catalog and
the champion model, retrying every `model_retry_seconds` until both succeed.

Monitoring (stage 5): `/metrics` for Prometheus (spec 5.1, `recsys_api.metrics`), every shown
recommendation queued for `recommendation_impressions` (spec 5.2, `recsys_api.impressions`)
and JSON logs on stdout (spec 5.6, `recsys.logs`). /metrics is served on the API port; in
production Caddy must not proxy it to the internet (stage 6).
"""

import hmac
import logging
import secrets
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from recsys.db.schema import make_engine
from recsys.logs import configure_logging
from recsys.models.als import pin_blas_threads
from recsys.tracking import champion_version, load_champion
from recsys_api.cache import KeyValueStore, RateLimiter, RecommendationCache, RedisStore
from recsys_api.catalog import CatalogHolder
from recsys_api.identity import COOKIE_NAME, CookieSigner
from recsys_api.impressions import ImpressionLogger, ImpressionObserver, RecentImpressions
from recsys_api.metrics import ApiMetrics, MetricsMiddleware, MetricsObserver
from recsys_api.model import ModelHolder
from recsys_api.observer import CompositeObserver, Observer
from recsys_api.schemas import (
    MovieList,
    MovieOut,
    RatingIn,
    RatingOut,
    RecommendationsOut,
    ReloadOut,
)
from recsys_api.service import (
    MAX_K,
    DatabaseUnavailable,
    RateLimited,
    RatingQuotaExceeded,
    RecommendationService,
    UnknownMovie,
)
from recsys_api.settings import Settings

log = logging.getLogger("recsys_api")

HERE = Path(__file__).parent
STATIC_PREFIX = "/static"
COOKIE_MAX_AGE = 365 * 24 * 3600
RELOAD_STATUS_CODES = {"loaded": 200, "unchanged": 200, "failed": 502, "busy": 409}


UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def route_uvicorn_logs() -> None:
    """uvicorn gives its loggers text handlers before it builds the app; send them through the
    root (JSON) handler instead, so its start-up and error lines are JSON too. A logger uvicorn
    switched off has no handlers and does not propagate (`--no-access-log`, see
    docker/api.Dockerfile: MetricsMiddleware writes one line per request instead, with the
    route template rather than the raw URL); it stays off."""
    for name in UVICORN_LOGGERS:
        uvicorn_log = logging.getLogger(name)
        if uvicorn_log.handlers:
            uvicorn_log.handlers.clear()
            uvicorn_log.propagate = True


def _configure_logging(level: str, fmt: str) -> None:
    if configure_logging(level, fmt):
        route_uvicorn_logs()
    logging.getLogger("recsys_api").setLevel(level)


class Warmup:
    """Background start-up: schema + catalog, then the model; retried until both are done."""

    def __init__(self, repo, catalog: CatalogHolder, models: ModelHolder, retry: float) -> None:
        self.repo, self.catalog, self.models, self.retry = repo, catalog, models, retry
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def attempt(self) -> bool:
        if len(self.catalog.current) == 0:
            try:
                self.repo.ensure_schema()
                size = self.catalog.refresh(self.repo)
                log.info("movie catalog loaded: %d movies", size)
            except DatabaseUnavailable as exc:
                log.warning("catalog not loaded yet (database unavailable): %s", exc)
        if self.models.current is None:
            self.models.reload()
        return len(self.catalog.current) > 0 and self.models.current is not None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self.attempt():
                    return
            except Exception:  # noqa: BLE001 - never let the warm-up thread die silently
                log.exception("warm-up attempt failed")
            self._stop.wait(self.retry)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="warmup", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _model_seed(recommender, n: int = 5) -> list[int]:
    """The model's own most-rated movies: smoke-test input while the catalog is not loaded."""
    feats = getattr(recommender, "movie_feats", None)
    if feats is None or "movie_n_ratings" not in feats:
        return []
    return [int(m) for m in feats["movie_n_ratings"].nlargest(n).index]


def create_app(
    settings: Settings | None = None,
    *,
    repo=None,
    store: KeyValueStore | None = None,
    loader: Callable | None = None,
    probe: Callable[[], str | None] | None = None,
    observer: Observer | None = None,
    background: bool = True,
) -> FastAPI:
    """Wire the app. Arguments other than `settings` replace real Postgres / Redis / MLflow
    (tests); `background=False` runs one synchronous warm-up attempt instead of the thread."""
    settings = settings or Settings()
    _configure_logging(settings.log_level, settings.log_format)

    if repo is None:
        from recsys_api.repository import PgRepository

        engine = make_engine(settings.database_url, pool_size=10, max_overflow=10, pool_timeout=5)
        repo = PgRepository(engine)
    if store is None:
        store = RedisStore(settings.redis_url)
    if loader is None:
        loader = partial(
            load_champion, settings.mlflow_tracking_uri, settings.model_name, settings.model_dir
        )
    if probe is None:
        probe = partial(champion_version, settings.mlflow_tracking_uri, settings.model_name)

    metrics = ApiMetrics()
    impressions = (
        ImpressionLogger(repo.insert_impressions, metrics=metrics)
        if settings.impressions_enabled
        else None
    )
    observers: list[Observer] = [MetricsObserver(metrics)]
    if impressions is not None:
        recent = RecentImpressions(window=settings.impressions_dedup_seconds)
        observers.append(ImpressionObserver(impressions, recent=recent))
    if observer is not None:
        observers.append(observer)
    observer = CompositeObserver(observers)

    catalog = CatalogHolder()

    def validate(recommender) -> None:
        """Smoke prediction before a new model goes live; raises -> old model stays."""
        seed = catalog.current.popular(5) or _model_seed(recommender)
        if not seed:
            raise ValueError("no movies to validate the model on (catalog not loaded)")
        if not recommender.recommend_new_user([(m, 5) for m in seed], k=10):
            raise ValueError(f"smoke prediction returned no recommendations for {seed}")

    models = ModelHolder(loader, probe, validate=validate, observer=observer)
    service = RecommendationService(
        repo=repo,
        catalog=lambda: catalog.current,
        models=models,
        cache=RecommendationCache(store, ttl=settings.cache_ttl_seconds),
        limiter=RateLimiter(store, limit=settings.rate_limit_per_minute),
        observer=observer,
        max_ratings=settings.max_ratings_per_user,
    )
    # Every response with a cookie writes impressions and may run an inference: a script must
    # not call /recommendations without bound (s5-review I-4). Own counter, generous budget;
    # like the ratings limiter it lets everything through while Redis is down (spec 3.5).
    recommendations_limiter = RateLimiter(
        store, limit=settings.recommendations_rate_limit_per_minute
    )
    if settings.session_secret is not None:
        secret = settings.session_secret.get_secret_value()
    else:
        log.warning("SESSION_SECRET is not set: visitor cookies will not survive a restart")
        secret = secrets.token_urlsafe(32)
    signer = CookieSigner(secret)
    warmup = Warmup(repo, catalog, models, settings.model_retry_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pin_blas_threads()  # once per process, before any model is used
        if impressions is not None:
            impressions.start()
        if background:
            warmup.start()
        else:
            warmup.attempt()
        yield
        warmup.stop()
        if impressions is not None:
            impressions.close()

    app = FastAPI(title="MovieLens recommender", version="0.1.0", lifespan=lifespan)
    app.add_middleware(MetricsMiddleware, metrics=metrics, mounts=(STATIC_PREFIX,))
    app.mount(STATIC_PREFIX, StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    app.state.service, app.state.models, app.state.catalog = service, models, catalog
    app.state.metrics, app.state.impressions = metrics, impressions

    def current_user(request: Request) -> int | None:
        return signer.verify(request.cookies.get(COOKIE_NAME))

    def client_ip(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def movie_list(movies, user_id: int | None) -> MovieList:
        mine = service.user_ratings(user_id) if movies else {}
        return MovieList(items=[MovieOut.of(m, mine.get(m.movie_id)) for m in movies])

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request):
        return templates.TemplateResponse(
            request, "index.html", {"min_ratings": service.min_personal}
        )

    @app.get("/movies/onboarding", response_model=MovieList)
    def onboarding(request: Request) -> MovieList:
        movies = catalog.current.onboarding(settings.onboarding_size)
        return movie_list(movies, current_user(request))

    @app.get("/movies/search", response_model=MovieList)
    def search(
        request: Request, q: Annotated[str, Query(min_length=2, max_length=100)]
    ) -> MovieList:
        return movie_list(catalog.current.search(q), current_user(request))

    @app.post("/ratings", response_model=RatingOut)
    def post_rating(body: RatingIn, request: Request, response: Response) -> RatingOut:
        user_id = current_user(request)
        try:
            result = service.rate(user_id, client_ip(request), body.movie_id, body.rating)
        except UnknownMovie as exc:
            raise HTTPException(422, f"unknown movie_id {body.movie_id}") from exc
        except RateLimited as exc:
            raise HTTPException(
                429, "too many ratings, try again in a minute", headers={"Retry-After": "60"}
            ) from exc
        except RatingQuotaExceeded as exc:
            limit = settings.max_ratings_per_user
            raise HTTPException(403, f"rating limit of {limit} per user reached") from exc
        except DatabaseUnavailable as exc:
            raise HTTPException(503, "ratings are temporarily unavailable") from exc
        if result.user_id != user_id:
            response.set_cookie(
                COOKIE_NAME,
                signer.sign(result.user_id),
                max_age=COOKIE_MAX_AGE,
                httponly=True,
                samesite="lax",
                secure=settings.cookie_secure,
            )
        return RatingOut(movie_id=result.movie_id, rating=result.rating, n_ratings=result.n_ratings)

    @app.get("/recommendations", response_model=RecommendationsOut)
    def recommendations(
        request: Request, k: Annotated[int, Query(ge=1, le=MAX_K)] = 10
    ) -> RecommendationsOut:
        user_id = current_user(request)
        keys = [f"recs:ip:{client_ip(request)}"] + (
            [f"recs:user:{user_id}"] if user_id is not None else []
        )
        if not recommendations_limiter.allow(keys):
            raise HTTPException(
                429, "too many requests, try again in a minute", headers={"Retry-After": "60"}
            )
        rec = service.recommend(user_id, k)
        current = catalog.current
        items = [MovieOut.of(current.get(m)) for m in rec.movie_ids if m in current]
        return RecommendationsOut(
            items=items,
            path=rec.path,
            model_version=rec.model_version,
            n_ratings=rec.n_ratings,
            min_ratings=service.min_personal,
            cached=rec.cached,
        )

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        body, content_type = metrics.render()
        return Response(body, headers={"Content-Type": content_type})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        model = models.current
        try:
            repo.ping()
            database = "ok"
        except DatabaseUnavailable:
            database = "unavailable"
        size = len(catalog.current)
        ok = model is not None and database == "ok" and size > 0
        body = {
            "status": "ready" if ok else "degraded",
            "model": {
                "loaded": model is not None,
                "version": model.version if model else None,
                "run_id": model.run_id if model else None,
                "loaded_at": model.loaded_at if model else None,
                "last_error": models.last_error,
            },
            "database": database,
            "redis": "ok" if store.ping() else "unavailable",
            "catalog_size": size,
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.post("/admin/reload-model", response_model=ReloadOut)
    def reload_model(request: Request, force: bool = False) -> JSONResponse:
        expected = settings.admin_token.get_secret_value() if settings.admin_token else ""
        if not expected:  # never compare against an empty token: that would accept "Bearer "
            raise HTTPException(503, "ADMIN_TOKEN is not configured")
        scheme, _, given = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(given.encode(), expected.encode()):
            raise HTTPException(401, "bad admin token", headers={"WWW-Authenticate": "Bearer"})
        try:
            catalog_size: int | None = catalog.refresh(repo)
        except DatabaseUnavailable:
            catalog_size = None
        result = models.reload(force=force)
        body = ReloadOut(
            status=result.status,
            version=result.version,
            previous_version=result.previous_version,
            error=result.error,
            catalog_size=catalog_size,
        )
        return JSONResponse(body.model_dump(), status_code=RELOAD_STATUS_CODES[result.status])

    metrics.precreate_error_series(
        sorted({route.path for route in app.routes if isinstance(route, APIRoute)})
    )
    return app
