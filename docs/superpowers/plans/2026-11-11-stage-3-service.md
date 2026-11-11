# Этап 3: онлайн-сервис (FastAPI + Postgres + Redis + MLflow server в compose). План реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Посетитель открывает страницу, оценивает ≥5 фильмов и сразу получает персональные рекомендации (fold-in ALS → LightGBM), всё поднимается одной командой `docker compose up -d --build`: Postgres (БД `app` и `mlflow`), Redis, MLflow server (Postgres backend, `--serve-artifacts`), API. MovieLens 1M загружается в Postgres идемпотентным CLI, чемпион обучается и регистрируется на сервере MLflow одной командой, API подхватывает его без перезапуска (`POST /admin/reload-model`).

**Architecture:**
- `src/recsys/db/` — общая для API и Airflow (этап 4) часть: схема БД `app` на SQLAlchemy Core (`schema.py`) и загрузчик MovieLens (`load_movielens.py`, CLI). Здесь же правило «веб-id не пересекаются с ML-1M».
- `services/api/recsys_api/` — FastAPI-приложение (spec 6.3: `services/api/` — роуты, шаблоны, статика). Слои: `settings.py` (env) → `repository.py` (SQL к `app`) / `cache.py` (Redis: кэш, rate limit, «предохранитель») / `model.py` (держатель модели, атомарная замена) → `catalog.py` (каталог в памяти, поиск, онбординг, холодный старт) → `service.py` (выбор пути `full | cold_start | fallback`, кэш, деградация) → `main.py` (`create_app`, роуты, lifespan, фоновая загрузка) + `templates/`, `static/`.
- Бизнес-логика не знает про FastAPI, Redis и Postgres: зависит от протоколов (`Repository`, `KeyValueStore`, загрузчик модели). Unit-тесты подставляют фейки, интеграционные — настоящие Postgres и Redis.
- Модель приходит **только из MLflow Registry** (`champion`), API никогда не обращается к ALS-факторам пользователя по id: любой пользователь с ≥5 оценками идёт через fold-in по своим оценкам из Postgres.

**Tech Stack:** Python 3.12, FastAPI 0.162 + Starlette, uvicorn, Jinja2, Pydantic 2 + pydantic-settings, SQLAlchemy 2.1 (Core) + psycopg 3, redis-py, mlflow-skinny 3.18 (клиент в образе API), mlflow 3.18 (сервер и локальный sqlite-store), Docker Compose v5, Postgres 16, Redis 7.4, pytest + TestClient (httpx).

**Spec:** `docs/superpowers/specs/2026-11-11-movielens-recsys-design.md` (секции 1, 2.4, 3, 6.2, 6.3, 6.5)

## Global Constraints

- Всё из планов этапов 0–2 в силе: `src/`-layout, зависимости только через `uv add` (+ `uv.lock` в git), ruff (line-length 100, py312, E,F,I,B,UP), код и коммиты на английском, коммиты заканчиваются пустой строкой и `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Не пушим, README не создаём.
- На `C:` ничего большого: кэши uv на `D:`, модель в API скачивается в явный каталог (`MODEL_DIR`), а не в системный temp.
- Unit-тесты (по умолчанию) не требуют Docker, сети, Postgres, Redis или MLflow-сервера. Интеграционные помечены `@pytest.mark.integration` и пропускаются, если не заданы `TEST_DATABASE_URL` и `TEST_REDIS_URL`. Интеграционные тесты отказываются работать с БД, имя которой не содержит `test` (они делают `drop_all`).
- Тесты не трогают `./mlruns` (только временные sqlite-хранилища pytest).
- Наружу (на хосте) открыт только порт приложения; Postgres, Redis, MLflow привязаны к `127.0.0.1` (spec 6.2). Секреты только в `.env` (в `.gitignore`), в git — `.env.example`.
- Целевая задержка `GET /recommendations` p95 < 100 мс (spec 3.5).

## Review Focus (топ-5 способов ошибиться)

1. **Гонки при горячей замене модели и при потоках.** Ссылка на модель меняется одним присваиванием, запрос берёт ссылку один раз; ошибка загрузки оставляет старую модель; две перезагрузки одновременно не идут; `threadpool_limits` больше не вызывается на каждый запрос (BLAS ограничивается один раз при загрузке модели). Тесты в Task 2 и Task 4.
2. **Несвежий кэш.** Новая оценка должна инвалидировать `recs:{user_id}:{model_version}`, включая гонку «запрос посчитал рекомендации по старым оценкам и записал их в кэш после новой оценки». Смена версии модели автоматически даёт промах. Тесты в Task 4.
3. **Коллизия веб-пользователей с ML-1M и подделка cookie.** Веб-id выдаются из отдельной последовательности (≥ 1 000 000), CHECK в БД связывает `source` и диапазон id; cookie подписана HMAC, чужой id подставить нельзя; веб-пользователь всегда идёт через fold-in. Тесты в Task 3, 4, 7.
4. **Деградация.** MLflow недоступен → popularity + `/ready` = 503 (degraded); ошибка модели → popularity; Postgres недоступен → запись 503, чтение из кэша или popularity; Redis недоступен → без кэша и без rate limit, но без таймаута на каждом запросе. Тесты в Task 4, 5, 7.
5. **Лимиты и валидация.** 30 оценок/мин на пользователя **и** на IP, ≤300 оценок на пользователя (атомарно, под блокировкой строки пользователя), оценка — строго целое 1..5 (pyfunc принимает только int), фильм должен существовать. Тесты в Task 4, 5, 7.

---

## Файловая структура (что добавится)

```
src/recsys/
├── db/
│   ├── schema.py              # metadata: movies, users, ratings; WEB_USER_ID_START; create_schema(engine)
│   └── load_movielens.py      # load_movielens(engine, raw_dir) -> LoadStats; CLI (--download)
├── models/als.py              # (изм.) pin_blas_threads(), один раз на процесс
├── models/pyfunc.py           # (изм.) требования: mlflow-skinny
├── tracking.py                # (изм.) load_champion(dst_dir) без глобального состояния, champion_version()
└── pipelines/train.py         # (изм.) --tmp-dir
services/api/recsys_api/
├── settings.py  repository.py  cache.py  model.py  catalog.py  service.py  identity.py
├── schemas.py   observer.py    main.py
├── templates/index.html
└── static/app.js  static/style.css
docker/api.Dockerfile  docker/mlflow.Dockerfile  .dockerignore
infra/postgres/init/01-create-databases.sh
infra/scripts/latency.py        # p95 по ~100 запросам (stdlib)
docker-compose.yml  .env.example
tests/unit/test_api_*.py        # бизнес-логика и HTTP с фейками
tests/integration/conftest.py  tests/integration/test_db.py  tests/integration/test_api_flow.py
.github/workflows/ci.yml        # + job integration (service containers) + job docker build
```

---

### Task 1: Зависимости, конфигурация pytest/ruff

**Files:** `pyproject.toml`, `uv.lock`, `.gitignore`

- Основная зависимость `mlflow` → `mlflow-skinny` (клиент: tracking REST, registry, pyfunc). Полный `mlflow` (sqlite/SQLAlchemy store, сервер) — в группе `mlflow-store`; `[tool.uv] default-groups = ["dev", "mlflow-store"]`, так что `uv sync` локально и в CI ставит всё как раньше, а образ API ставит `--no-default-groups`.
- `uv add fastapi uvicorn jinja2 pydantic-settings sqlalchemy "psycopg[binary]" redis`; `uv add --dev httpx`.
- pytest: `pythonpath = ["services/api", "tests/unit"]`, маркер `integration`, `filterwarnings` для двух сторонних предупреждений (SQLAlchemy `noload` внутри mlflow sqlalchemy_store и pandas `Operation with range` внутри `mlflow.models.utils`) — точечно по сообщению и модулю.
- ruff: `known-first-party = ["recsys", "recsys_api"]`.
- [x] `uv lock --check`, весь набор зелёный, `pytest -q` без предупреждений. Commit `chore: service deps, mlflow-skinny core dependency, pytest config`.

### Task 2: Долги этапа 2 в `recsys`

**Files:** `src/recsys/models/als.py`, `src/recsys/tracking.py`, `src/recsys/models/pyfunc.py`, `src/recsys/pipelines/train.py`; тесты `tests/unit/test_als.py`, `tests/unit/test_tracking.py`

**Interfaces:**
```python
# recsys.models.als
def pin_blas_threads() -> None
    # BLAS = 1 поток на весь процесс (threadpoolctl без контекстного менеджера), после этого
    # методы ALS больше не оборачивают каждый вызов в threadpool_limits (это гонка между потоками).
# recsys.tracking
def champion_version(tracking_uri: str | None = None, name: str = MODEL_NAME) -> str | None
def load_champion(tracking_uri=None, name=MODEL_NAME, dst_dir: Path | None = None) -> ChampionModel
    # alias разрешается один раз; артефакты models:/{name}/{version} качаются в уникальный
    # подкаталог dst_dir (по умолчанию mlruns/tmp/champion), модель читается в память,
    # подкаталог удаляется в finally. MlflowClient(tracking_uri=..., registry_uri=...) и
    # download_artifacts(tracking_uri=..., registry_uri=...): mlflow.set_tracking_uri не вызывается.
def register_and_promote(..., *, beats_popularity: bool) -> RegistrationOutcome   # обязательный keyword
# recsys.pipelines.train: CLI --tmp-dir (по умолчанию mlruns/tmp) — в контейнере /tmp/recsys
```
- [x] Тесты (RED → GREEN): `test_pin_blas_threads_skips_per_call_limits` (после `pin_blas_threads` `threadpool_limits` в `candidates_for_ratings` не вызывается — monkeypatch на исключение), `test_load_champion_does_not_touch_global_tracking_uri`, `test_load_champion_downloads_into_dst_dir_and_cleans_up`, `test_champion_version_none_without_alias`, обновлённый `test_load_champion_resolves_alias_once_and_loads_that_exact_version`, `test_register_and_promote_requires_explicit_beats_popularity`, `test_pip_requirements_pin_mlflow_skinny`, `test_cli_tmp_dir_option`.
- [x] Commit `fix: pin BLAS once per process, side-effect-free champion loading, explicit popularity gate`.

### Task 3: Схема БД `app` и загрузчик MovieLens

**Files:** `src/recsys/db/schema.py`, `src/recsys/db/load_movielens.py`; тесты `tests/unit/test_db_schema.py` (без БД), `tests/integration/test_db.py`

**Interfaces:**
```python
WEB_USER_ID_START = 1_000_000
SOURCES = ("movielens", "web")
metadata: MetaData
movies(movie_id INT PK, title TEXT, year SMALLINT NULL, genres TEXT  -- "Action|Comedy")
users(user_id INT PK, source TEXT, gender CHAR(1) NULL, age SMALLINT NULL, occupation SMALLINT NULL,
      zip TEXT NULL, created_at TIMESTAMPTZ DEFAULT now(),
      CHECK source IN SOURCES, CHECK (source = 'web') = (user_id >= WEB_USER_ID_START))
ratings(user_id INT FK users, movie_id INT FK movies, rating SMALLINT CHECK 1..5, source TEXT,
        rated_at TIMESTAMPTZ, updated_at TIMESTAMPTZ, PK (user_id, movie_id), INDEX (movie_id))
web_user_id_seq START WITH WEB_USER_ID_START
def create_schema(engine) -> None                 # идемпотентно (create_all)
def make_engine(url: str, **kwargs) -> Engine     # pool_pre_ping, connect_timeout
# load_movielens
@dataclass LoadStats(movies: int, users: int, ratings: int)
def movie_rows(movies_df) -> list[dict]; user_rows(users_df) -> list[dict]   # чистые, unit-тесты
def load_movielens(engine, raw_dir: Path) -> LoadStats   # upsert (ON CONFLICT), оценки через COPY в temp-таблицу
CLI: python -m recsys.db.load_movielens [--data-dir data/raw/ml-1m] [--download] [--database-url URL]
     (по умолчанию $DATABASE_URL)
```
- [x] Unit: `test_movie_rows_join_genres_and_keep_year_null`, `test_user_rows_mark_source_movielens`, `test_web_ids_start_above_movielens_range`. Integration: `test_create_schema_is_idempotent`, `test_web_user_sequence_starts_at_web_range`, `test_check_rejects_web_user_in_movielens_range`, `test_loader_is_idempotent` (два прогона на синтетическом raw dir → одинаковые числа, source=movielens).
- [x] Commit `feat(db): app schema and idempotent MovieLens loader`.

### Task 4: Бизнес-логика сервиса (без FastAPI)

**Files:** `services/api/recsys_api/{catalog,identity,cache,model,service,observer}.py`; тесты `tests/unit/test_api_catalog.py`, `test_api_identity.py`, `test_api_cache.py`, `test_api_model.py`, `test_api_service.py`, фейки в `tests/unit/api_fakes.py`

**Interfaces:**
```python
# catalog.py
@dataclass(frozen=True) Movie(movie_id, title, year, genres: tuple[str, ...], n_ratings)
def display_title(raw: str) -> str            # "Matrix, The" -> "The Matrix"
class Catalog:
    from_movies(movies) ; get(movie_id) ; __contains__ ; __len__ ; by_popularity
    popular(k, exclude) -> list[int]
    search(q, limit=20) -> list[Movie]        # casefold-подстрока, сначала совпадение с начала, затем по популярности
    onboarding(n=20, pool=500) -> list[Movie] # round-robin по жанрам (жанры по суммарной популярности)
    genre_popular(ratings, k, pool=500) -> list[int]   # 1–4 оценки: жанры оценок ≥4, вес = число таких оценок
# identity.py — подписанная cookie
class CookieSigner(secret): sign(user_id) -> str ; verify(value) -> int | None   # HMAC-SHA256, compare_digest
# cache.py
class KeyValueStore(Protocol): get_many, set, delete, incr_with_ttl ...
class RedisStore(url, timeout=0.2s, cooldown=5s)  # ошибка → «предохранитель»: Redis пропускается cooldown секунд
class RecommendationCache(store, ttl=600): get(user_id, version) ; put(...) ; invalidate(user_id, version)
    # ключ recs:{user_id}:{model_version}; в значении — номер поколения оценок пользователя
    # (rv:{user_id}, INCR при каждой оценке), запись с устаревшим поколением = промах
class RateLimiter(store, limit=30, window=60): allow(keys) -> bool   # fixed window; Redis недоступен -> True
# model.py
@dataclass(frozen=True) LoadedModel(recommender, version, run_id, loaded_at)
class ModelHolder(loader, version_probe):
    current -> LoadedModel | None             # атомарное чтение ссылки
    reload(force=False) -> ReloadResult       # status: loaded | unchanged | failed | busy; старая модель остаётся при ошибке
# service.py
MIN_PERSONAL_RATINGS = 5 ; MAX_K = 50
@dataclass Recommendation(movie_ids, path: "full"|"cold_start"|"fallback", model_version, n_ratings, cached)
class RecommendationService(repo, catalog_ref, holder, cache, observer):
    recommend(user_id | None, k) -> Recommendation
    # 0 оценок -> популярное (cold_start); 1–4 -> популярное в любимых жанрах (cold_start);
    # ≥5 -> модель (fold-in) с добивкой популярным (full); нет модели / ошибка -> популярное (fallback, не кэшируется);
    # Postgres недоступен: кэш, иначе популярное (fallback)
    rate(user_id | None, ip, movie_id, rating) -> RatingResult   # лимиты, создание веб-пользователя, upsert, инвалидация кэша
# observer.py — хуки для этапа 5 (Prometheus, impressions), по умолчанию no-op
```
- [x] Тесты (фейки: `FakeRepository`, `FakeStore`, `StubRecommender`): каталог (display title, поиск, онбординг покрывает разные жанры и детерминирован, genre_popular, popular с exclude), cookie (round-trip, подделка, чужой секрет, мусор), кэш (hit/miss, TTL передан, другая версия = промах, гонка поколений), rate limiter (31-й запрос отклонён, отдельные окна для user и IP, Redis down → разрешено), предохранитель Redis (после ошибки store не вызывается cooldown секунд), `ModelHolder` (загрузка, unchanged, ошибка оставляет старую, busy при параллельной перезагрузке, ссылка не «рвётся»), сервис (все пути таблицы 2.4, fallback при ошибке модели, Postgres down → кэш/популярное, оценённое не рекомендуется, добивка до k, веб-пользователь идёт через `recommend_new_user`).
- [x] Commit `feat(api): service core — catalog, cold start, cache, rate limits, model holder`.

### Task 5: FastAPI-приложение и страница

**Files:** `services/api/recsys_api/{settings,schemas,main}.py`, `templates/index.html`, `static/{app.js,style.css}`; тесты `tests/unit/test_api_http.py`

**Endpoints (spec 3.1, без `/metrics` — этап 5):**
| Метод и путь | Контракт |
|---|---|
| `GET /` | HTML (Jinja2), онбординг, поиск, звёзды, блок рекомендаций |
| `GET /movies/onboarding` | `{"items": [Movie + user_rating]}` (20 фильмов) |
| `GET /movies/search?q=` | `q` 2..100 символов, до 20 фильмов |
| `POST /ratings` | тело `{"movie_id": int, "rating": int 1..5}` (strict int) → `{"movie_id","rating","n_ratings"}`; 422 — валидация/нет фильма; 429 — rate limit; 403 — лимит 300; 503 — Postgres недоступен. Ставит cookie `uid` (HttpOnly, SameSite=Lax) новому посетителю |
| `GET /recommendations?k=10` | `k` 1..50 → `{"items","path","model_version","n_ratings","min_ratings","cached"}` |
| `GET /health` | 200 `{"status":"ok"}` |
| `GET /ready` | 200 `{"status":"ready",...}` если модель загружена, БД и каталог доступны; иначе 503 `{"status":"degraded",...}` с причинами |
| `POST /admin/reload-model?force=false` | `Authorization: Bearer $ADMIN_TOKEN`; 200 `loaded/unchanged`, 502 `failed` (старая модель осталась), 409 `busy`, 401 неверный токен, 503 токен не настроен |

Lifespan: `pin_blas_threads()`, `create_schema`, фоновая нить грузит каталог и модель с повтором каждые `MODEL_RETRY_SECONDS`, пока не получится (сервис стартует сразу, до этого — popularity/деградация). `uvicorn --factory recsys_api.main:create_app`.
- [x] Тесты (TestClient + фейки): все контракты выше, cookie выдаётся один раз и подпись проверяется, подделанная cookie → новый пользователь, 5 оценок → `path=full`, повтор → `cached=true`, новая оценка → `cached=false`, 31-я оценка за минуту → 429, лимит 300 (в тесте маленький), `/ready` 503 без модели и 200 с моделью, admin reload (401/503/200/502), страница рендерится и содержит ссылки на статику.
- [x] Commit `feat(api): FastAPI app, endpoints and demo page`.

### Task 6: Postgres-репозиторий и интеграционные тесты API

**Files:** `services/api/recsys_api/repository.py`; `tests/integration/conftest.py`, `tests/integration/test_api_flow.py`

**Interfaces:**
```python
class PgRepository(engine):
    ping() ; load_movies() -> list[Movie] (с числом оценок) ; user_ratings(user_id) -> list[tuple[int, int]]
    create_web_user() -> int        # nextval(web_user_id_seq)
    upsert_rating(user_id, movie_id, rating, max_ratings) -> int   # SELECT ... FOR UPDATE строки пользователя,
        # квота считается под блокировкой; raises UserNotFound / RatingQuotaExceeded
```
- [x] Integration (`TEST_DATABASE_URL`, `TEST_REDIS_URL`): новый посетитель → 5 оценок → `full` (модель на синтетике, fold-in), кэш в настоящем Redis (`recs:{uid}:{ver}` есть, TTL ≤ 600), новая оценка инвалидирует, повторная оценка обновляет запись (unique), rate limit 429, квота 403, Redis недоступен (неверный порт) → оценки и рекомендации работают без кэша/лимитов, веб-id ≥ 1 000 000.
- [x] Commit `feat(api): Postgres repository and integration tests`.

### Task 7: Docker, compose, `.env.example`

**Files:** `docker/api.Dockerfile`, `docker/mlflow.Dockerfile`, `.dockerignore`, `docker-compose.yml`, `.env.example`, `infra/postgres/init/01-create-databases.sh`

- `api.Dockerfile`: multi-stage, uv 0.13.7, `uv sync --frozen --no-default-groups --no-editable` в `/opt/venv` (только основные зависимости: mlflow-skinny, без dev/полного mlflow), runtime `python:3.12-slim` + `libgomp1`, пользователь без root, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MODEL_DIR=/tmp/models`, `uvicorn --factory`.
- `mlflow.Dockerfile`: `python:3.12-slim` + `mlflow==3.18.0` (как в lock) + `psycopg[binary]`.
- compose: `postgres` (один инстанс; init-скрипт создаёт роли и БД `app`, `mlflow`; блок `airflow` создаётся, если задан `AIRFLOW_DB_PASSWORD` — крючок этапа 4), `redis` (без persistence, maxmemory + LRU), `mlflow` (`--backend-store-uri postgresql+psycopg://…/mlflow --serve-artifacts --artifacts-destination /mlartifacts`, `MLFLOW_SERVER_ALLOWED_HOSTS`), `api`; профиль `tools`: `loader` и `trainer` (тот же образ API). Healthchecks, `depends_on: condition`, именованные тома `pgdata`, `mlartifacts`, порты внутренних сервисов на `127.0.0.1`.
- Одна команда обучения и регистрации против сервера: `docker compose --profile tools run --rm trainer`.
- [x] `docker compose config -q`, сборка обоих образов. Commit `feat(docker): api and mlflow images, compose stack`.

### Task 8: CI

**Files:** `.github/workflows/ci.yml`
- Job `test` без изменений. Job `integration`: service containers `postgres:16-alpine` и `redis:7.4-alpine` с healthcheck, `TEST_DATABASE_URL`, `TEST_REDIS_URL`, `uv run pytest -m integration -v`. Job `docker`: `docker compose config -q` (с `.env.example`) и `docker build` обоих образов (без push — этап 7).
- [x] Commit `ci: integration tests with Postgres/Redis services and image builds`.

### Task 9: End-to-end в Docker и замер задержки

```bash
cp .env.example .env   # + сгенерировать секреты
docker compose up -d --build
docker compose --profile tools run --rm loader     # MovieLens -> Postgres
docker compose --profile tools run --rm trainer    # train + register champion на сервере MLflow
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" localhost:8000/admin/reload-model
# онбординг -> 5 оценок -> /recommendations (path=full) -> повтор (cached=true) -> 31 оценка (429) -> /ready
python infra/scripts/latency.py --n 100       # p95 cold (без кэша) и warm
docker compose down                           # тома сохраняются
```
- [x] Транскрипт ключевых выводов — в отчёт этапа.

---

## Decisions

1. **`services/api/recsys_api/`, а не `src/recsys/api/`.** Spec 6.3 разделяет ML-пакет (`src/recsys`) и сервис (`services/api/`). Пакет сервиса называется `recsys_api` (уникальное имя, не `app`), в образ копируется как каталог и импортируется через `PYTHONPATH=/app`; в тестах — `pythonpath = ["services/api", …]`. Общая для API и Airflow часть БД (схема, загрузчик) — в `src/recsys/db/`, потому что DAG этапа 4 её переиспользует.
2. **Миграции: идемпотентный `create_all`, без Alembic.** Схема маленькая, до v2 ожидаются только новые таблицы (`recommendation_impressions`, `drift_metrics` на этапе 5) — `create_all` их добавит. Alembic нужен при первом `ALTER` существующей таблицы; путь перехода — `alembic stamp` на текущей схеме. Последовательность веб-id тоже объект metadata.
3. **Веб-id из отдельной последовательности `web_user_id_seq START 1 000 000`**, плюс CHECK `(source = 'web') = (user_id >= 1 000 000)`: коллизия с ML-1M (1..6040) невозможна даже при ошибке в коде. Веб-пользователь всегда обслуживается fold-in по своим оценкам из Postgres (`recommend_new_user`), сохранённые ALS-факторы по id не используются ни для кого.
4. **Анонимный пользователь = подписанная cookie `uid` (`{id}.{hmac}`),** `SESSION_SECRET` из env. Пользователь создаётся при первой оценке (не при просмотре страницы — боты не плодят строки). Подпись нужна, чтобы нельзя было оценивать от чужого id; HMAC без БД позволяет отдать рекомендации из кэша при недоступном Postgres. Неизвестный БД id (например, после сброса БД) → новый пользователь и новая cookie.
5. **Каталог фильмов в памяти API** (3883 строки + число оценок): поиск, онбординг, холодный старт и popularity работают без запроса к БД и при недоступном Postgres. Обновляется при старте, при `reload-model` и фоновым повтором, пока пуст.
6. **Кэш хранит топ-50 (MAX_K), ответ режется до `k`,** потому что ключ spec (`recs:{user_id}:{model_version}`) не содержит `k`. Защита от гонки: в значении лежит поколение оценок пользователя (`rv:{user_id}`, `INCR` на каждую оценку); при чтении `MGET` ключа и поколения за один запрос. Fallback-ответы не кэшируются (чтобы восстановление модели сразу было видно). Без модели версия в ключе — `none`.
7. **Rate limit — fixed window 60 с в Redis** (`INCR` + `EXPIRE` в pipeline) по ключам пользователя и IP. Простое и дешёвое решение; на границе окна возможно до 2× лимита за короткое время — для защиты от мусора достаточно. Redis недоступен → лимит отключён (spec 3.5). Квота 300 оценок проверяется в Postgres под `SELECT … FOR UPDATE` строки пользователя (повторная оценка того же фильма квоту не тратит). Коды: 429 — частота, 403 — квота.
8. **«Предохранитель» для Redis:** короткий таймаут сокета (200 мс) и после ошибки Redis пропускается 5 с, чтобы при упавшем Redis не платить таймаут на каждом запросе.
9. **Инференс сериализован `threading.Lock`, BLAS = 1 поток на процесс** (`pin_blas_threads()` при старте + `OPENBLAS_NUM_THREADS=1` в образе). Один fold-in + ранжирование ~20–30 мс; сериализация даёт предсказуемую задержку без переподписки CPU и без вопросов о потокобезопасности `implicit`/LightGBM. Пропускная способность промахов ~30–40 rps на процесс, остальное закрывает кэш; если понадобится больше — несколько воркеров uvicorn.
10. **Перезагрузка модели синхронная для вызывающего (`POST /admin/reload-model` ждёт результата),** но выполняется в пуле потоков, так что остальные запросы обслуживаются. Airflow (этап 4) получает в ответе итог (`loaded`/`unchanged`/`failed`). Если версия чемпиона не изменилась — `unchanged` без скачивания (`?force=true` перезагружает). Параллельная перезагрузка → 409.
11. **`/ready` = 503 при деградации** (нет модели, нет БД или пустой каталог) с причинами в теле; healthcheck контейнера смотрит `/health` (процесс жив), поэтому деградировавший API не перезапускается в цикле, а smoke-тест деплоя (этап 7) увидит деградацию.
12. **mlflow-skinny в образе API.** Основная зависимость пакета — `mlflow-skinny` (REST-клиент tracking/registry, pyfunc); полный `mlflow` (SQLAlchemy store для локального sqlite, сервер) — в группе `mlflow-store`, входящей в группы по умолчанию. Образ API ставит только основные зависимости. `requirements.txt` модели фиксирует `mlflow-skinny==…` (ставится и с полным `mlflow`).
13. **Модель в compose регистрируется заново на сервере MLflow**, а не переносится из локального `mlruns/` (там абсолютные пути `D:\`). `trainer` = образ API + `python -m recsys.pipelines.train --data-dir /data/raw/ml-1m --tmp-dir /tmp/recsys`, `MLFLOW_TRACKING_URI=http://mlflow:5000`. Эксперимент на сервере создаётся с `mlflow-artifacts:/`, клиенты грузят артефакты по HTTP, общих томов нет (spec 4.3). Источник данных этапа 3 — raw-файлы; выгрузка из Postgres (с веб-оценками) — задача `extract_ratings` этапа 4.
14. **MLflow REST-клиент в API с ограниченными ретраями** (`MLFLOW_HTTP_REQUEST_MAX_RETRIES=2`, `MLFLOW_HTTP_REQUEST_TIMEOUT=10`): по умолчанию клиент ретраит минутами, а перезагрузка должна быстро сообщать об ошибке.
15. **Popularity для fallback и холодного старта — по всем оценкам в Postgres** (MovieLens + веб), из каталога в памяти; совпадает с бейзлайном spec 2.5 по смыслу (самые оцениваемые фильмы).
16. **Онбординг:** round-robin по жанрам (порядок — по суммарному числу оценок жанра), в каждом жанре самый популярный ещё не выбранный фильм из топ-500. Жанры в ML-1M перечислены по алфавиту, поэтому «первый жанр» фильма для разнообразия не годится.
17. **Холодный старт 1–4 оценки:** жанры фильмов с оценкой ≥4, вес жанра = число таких оценок; кандидаты — топ-500 популярных, счёт = сумма весов совпавших жанров, при равенстве — популярность. Если оценок ≥4 нет — просто популярное без оценённого.

### Решения, принятые при реализации

18. **`scikit-learn` — явная основная зависимость.** `LambdaRanker` использует sklearn-API LightGBM (`LGBMRanker`), а `scikit-learn` раньше приходил только транзитивно через полный `mlflow`. В образе с `mlflow-skinny` обучение падало (`LightGBMError: scikit-learn is required`) — найдено на e2e, CI-job `docker` теперь проверяет `LGBMRanker()` в образе.
19. **MLflow server: `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`, `--workers=1`.** MLflow 3.18 по умолчанию поднимает ~9 процессов huey для GenAI-джобов (~2.1 ГБ RSS); для tracking + registry они не нужны. Итог ~380 МБ (spec 6.1 закладывает ~0.3 ГБ).
20. **`_current_champion` считает «нет чемпиона» только по кодам not-found** (`RESOURCE_DOES_NOT_EXIST`, `INVALID_PARAMETER_VALUE`). Раньше любая ошибка реестра означала «чемпиона нет»: API писал неверную причину, а `register_and_promote` при недоступном реестре продвинул бы модель без сравнения.
21. **`load_champion` сам разрешает место хранения версии** (`get_model_version_download_uri` → для MLflow 3 `models:/m-…` → `get_logged_model().artifact_location`) и качает его с явными URI: `download_artifacts("models:/name/v", tracking_uri=…)` в mlflow 3.18 всё равно ходит в глобальный registry URI (`ModelsArtifactRepository._get_model_uri_infos`).
22. **Загрузчик в compose скачивает ML-1M в именованный том `movielens`** (общий для `loader` и `trainer`, на этапе 4 его же подключит Airflow), а не в bind-mount `./data`: на VPS не будет проблем с правами каталога хоста. Каталог `/data` в образе принадлежит пользователю `app`, поэтому новый том наследует владельца.
23. **Порты внутренних сервисов на хосте настраиваются в `.env`** (`POSTGRES_PORT`, `REDIS_PORT`, `MLFLOW_PORT`): на машине разработки порт 5432 занят нативным Postgres 18, локально использован `POSTGRES_PORT=15432`.
24. **`httpx2` вместо `httpx` для TestClient** — Starlette 1.10 объявила `httpx` устаревшим для `starlette.testclient`.
25. **MLflow-клиент в API: `MAX_RETRIES=1`, `BACKOFF_FACTOR=1`, `TIMEOUT=10`.** При остановленном MLflow `reload-model` возвращает 502 примерно за 16 с (время уходит на DNS-ошибку контейнера), старая модель продолжает работать.
26. **`RATE_LIMIT_PER_MINUTE` пробрасывается в compose** (по умолчанию 30) — только для замера задержки (`infra/scripts/latency.py` создаёт 100 посетителей с одного IP).
