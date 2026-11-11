# Этап 4: оркестрация в Airflow (`bootstrap_data`, `retrain_daily`). План реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Система переобучается сама. В Airflow UI (127.0.0.1:8080) видны два DAG: `bootstrap_data` (ручной: скачать ML-1M → проверить MD5 → загрузить в Postgres `app`) и `retrain_daily` (`@daily`: выгрузка оценок из Postgres, включая веб-оценки → валидация → отчёт о дрифте → split → ALS → ранжировщик → оценка challenger, champion и базовых линий на одном сплите → решение → регистрация, alias `champion`, перезагрузка API, либо запись причины отказа). Всё поднимается тем же `docker compose up -d --build`.

**Architecture:**
- Вся логика в пакете `recsys`, DAG-файлы только связывают задачи (spec 4.1). Каждая задача DAG — вызов одной функции `recsys.pipelines.retrain.<task>(run_dir)` или `recsys.pipelines.bootstrap.<task>(...)`. Функции принимают и возвращают пути, пишут результаты в `/data/runs/{run_id}/` (Parquet, JSON, бандлы моделей). Через XCom идут только строка `run_dir` и id ветки.
- Образ Airflow = `apache/airflow:slim-3.3.2-python3.12` + драйвер Postgres для метаданных + **отдельный venv `/opt/recsys-venv`** с `recsys`, собранный `uv sync --frozen` из того же `uv.lock`, что и образ API. Задачи запускаются через `@task.external_python(python="/opt/recsys-venv/bin/python")` и `@task.branch_external_python`. В окружение самого Airflow `recsys` не ставится (Decision 2).
- Состояние между запусками (сколько оценок было в прошлый раз, доля веб-оценок, распределение оценок) хранится в новой таблице `retrain_runs` БД `app`. Её пишет `validate_data`, читают правило «−5%» и `drift_report`.
- Один MLflow run на запуск DAG. Его id лежит в `run_dir/mlflow_run.json`, и каждая задача логирует в тот же run, ретраи переиспользуют его.
- `drift_report` на этапе 4 — лёгкая JSON-сводка (`recsys.drift.build_drift_report`). На этапе 5 эту функцию заменит Evidently (Decision 5).

**Tech Stack:** Airflow 3.3.2 (LocalExecutor, SimpleAuthManager, TaskFlow API, standard provider 1.19), Postgres 16 (метаданные в БД `airflow`), MLflow 3.18 server, uv 0.13.7, Python 3.12, pytest.

**Spec:** `docs/superpowers/specs/2026-11-11-movielens-recsys-design.md`, секции 1, 2, 4 (основная), 6.3, 6.5.

## Global Constraints

- Всё из планов этапов 0–3 в силе: `src/`-layout, зависимости только через `uv add`/`uv lock` (+ `uv.lock` в git), ruff (line-length 100, py312, E,F,I,B,UP), код и коммиты на английском, каждый коммит заканчивается пустой строкой и `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Не пушим, `--no-verify` не используем, README не создаём.
- Unit-тесты не требуют Docker, сети, Postgres, Airflow и MLflow-сервера: состояние пишется во временную SQLite, MLflow — во временный sqlite-store, API — тестовый HTTP-сервер на 127.0.0.1. Интеграционные помечены `integration`. DAG-тест пропускается, если `airflow` не импортируется, и запускается внутри образа Airflow (локально и в CI).
- На `C:` ничего большого: кэши uv на `D:`, данные и временные файлы задач лежат в `/data` внутри контейнера (том Docker на D:).
- Память: WSL ограничен 6 ГБ. Контейнеры Airflow в простое ≤ ~1,5 ГБ суммарно (один воркер api-server, 1 процесс парсинга, `parallelism=2`). Замеряется `docker stats --no-stream`.
- Порт UI Airflow привязан к `127.0.0.1` (spec 6.2). Секреты только в `.env`, в `.env.example` — плейсхолдеры.
- Существующий стек (postgres, redis, mlflow, api, профиль `tools`) продолжает работать, `python -m recsys.pipelines.train` тоже.

## Review Focus (топ-5 способов ошибиться)

1. **Идемпотентность и ретраи.** Повтор задачи с тем же `run_id` даёт тот же результат: снапшот замораживается первой успешной выгрузкой, запись атомарная (временная папка → `os.replace`), обучение детерминировано (сиды), MLflow run переиспользуется, регистрация не создаёт вторую версию для того же run, alias ставится повторно без вреда. Тесты в Task 3–5.
2. **Честное сравнение challenger и champion.** Оба оцениваются **на одном тестовом сплите текущего запуска** и **одним путём (fold-in)**, тем же, по которому API обслуживает веб-посетителей. Champion загружается из Registry, его ALS не переобучается. Порог: NDCG@10 ≥ champion − 0.005 и coverage −10% максимум. Ошибка реестра ≠ «чемпиона нет». Тесты в Task 4.
3. **Валидация падает громко.** NULL, оценка вне 1..5, дубликаты `(user_id, movie_id)`, падение числа оценок > 5% относительно прошлого запуска (с тем же флагом `include_web`), пустой снапшот: задача падает с перечнем всех проблем, данные в `retrain_runs` не записываются. Каждое правило покрыто тестом.
4. **Граница двух окружений Python.** DAG-файлы не импортируют `recsys`, функции для `external_python` самодостаточны (импорт внутри тела). В образе: `airflow` в своём venv, `recsys` в `/opt/recsys-venv`, конфликт версий (SQLAlchemy 2.0 vs 2.1, pandas 3.0 vs 3.1) не возникает. DAG-тест проверяет импорт DAG, id задач и рёбра. Отдельный smoke-тест в образе проверяет, что `recsys` импортируется в venv.
5. **Перезагрузка API не откатывает промоушен.** Alias уже стоит, источник правды — Registry. Если API недоступен, задача `reload_api` падает после ретраев (видно красным в UI), модель остаётся продвинутой. Проверяется, что API загрузил именно ту версию, которую продвинули. Тесты с тестовым HTTP-сервером.

---

## Файловая структура (что добавится)

```
airflow/dags/
├── bootstrap_data.py            # ручной DAG: download -> verify_checksum -> load_postgres
└── retrain_daily.py             # @daily DAG, ветвление promote / skip
docker/airflow.Dockerfile        # apache/airflow slim + psycopg2 + /opt/recsys-venv
src/recsys/
├── drift.py                     # лёгкий отчёт о дрифте (крючок для Evidently, этап 5)
├── db/schema.py                 # + таблица retrain_runs
├── db/extract.py                # снапшот ratings/movies/users из БД app
├── db/run_state.py              # record_run / previous_run
├── data/download.py             # fetch_archive + extract_archive (download_ml1m их композиция)
├── pipelines/run_dir.py         # раскладка /data/runs/{run_id}, атомарная запись
├── pipelines/bootstrap.py       # задачи bootstrap_data
├── pipelines/retrain.py         # задачи retrain_daily + CLI для ручного прогона
├── pipelines/reload.py          # POST /admin/reload-model (stdlib urllib)
├── pipelines/train.py           # рефакторинг: fit_history_als + evaluate_variants
└── tracking.py                  # + ensure_run, register_version, set_champion
infra/postgres/init/01-create-databases.sh   # airflow БД обязательна; db-init в compose
tests/unit/test_run_dir.py, test_drift.py, test_run_state.py, test_retrain_tasks.py,
           test_reload_api.py, test_bootstrap.py
tests/integration/test_extract.py
tests/dags/test_dag_integrity.py # только внутри образа Airflow
```

---

### Task 1: План (этот файл) — коммит

### Task 2: Строительные блоки (TDD)

**Files:** `src/recsys/pipelines/run_dir.py`, `src/recsys/drift.py`, `src/recsys/db/schema.py`, `src/recsys/db/run_state.py`, `src/recsys/db/extract.py`, `src/recsys/data/download.py`; тесты `test_run_dir.py`, `test_drift.py`, `test_run_state.py`, `test_download.py`, `tests/integration/test_extract.py`.

```python
# run_dir.py
safe_run_id(run_id: str) -> str                       # "manual__2026-11-11T10:00:00+00:00" -> "manual__2026-11-11T10-00-00-00-00"
class RunDir: root: Path; extract/split/als/ranker/model dirs; *.json paths
    for_run(run_id, runs_root=None) -> RunDir          # runs_root: $RUNS_DIR, default /data/runs
atomic_dir(target: Path) -> ContextManager[Path]      # пишем в target.tmp-xxxx, затем os.replace
write_json_atomic(path, obj) / read_json(path)

# db/schema.py: retrain_runs(run_id PK, include_web, n_ratings, n_web_ratings, n_users,
#                            data_snapshot, rating_counts JSON, recorded_at)
# db/run_state.py
RunRecord (dataclass), record_run(engine, record), previous_run(engine, run_id, include_web) -> RunRecord | None
# db/extract.py
extract_snapshot(engine, include_web=True) -> tuple[ratings, movies, users]
    # ratings: user_id, movie_id, rating, timestamp(epoch s), source; movies как load_movies;
    # users: только movielens (у веб-пользователей демография NaN, как в API)
# drift.py
summarize_ratings(ratings) -> dict                    # n_ratings, n_web, web_share, n_users, rating_share{1..5}
build_drift_report(current: dict, previous: dict | None) -> dict   # Stage 5: Evidently
# download.py
fetch_archive(dest_dir, url, md5) -> Path; extract_archive(archive, dest_dir) -> Path
```

Тесты: `safe_run_id` без `:`/`+`/`/`; `atomic_dir` не оставляет половинчатого результата при исключении и заменяет старый; `previous_run` игнорирует текущий run и другой `include_web`, берёт самый свежий; `record_run` повторно — upsert; drift считает доли и дельты, `previous=None` → без дельт; extract (integration) возвращает веб и movielens, `include_web=False` отбрасывает веб, timestamp = epoch, жанры — списки.

### Task 3: Задачи `retrain_daily` (TDD)

**Files:** `src/recsys/pipelines/retrain.py`, `src/recsys/pipelines/train.py` (рефакторинг), `src/recsys/tracking.py`; тест `tests/unit/test_retrain_tasks.py`.

```python
extract_ratings(run_id, database_url=None, include_web=True, runs_root=None) -> str  # run_dir
validate_data(run_dir, database_url=None, tracking_uri=None, max_drop=0.05) -> str
drift_report(run_dir, database_url=None, tracking_uri=None) -> str
split_data(run_dir) -> str
train_als(run_dir, tracking_uri=None, config=TrainConfig()) -> str
train_ranker(run_dir, tracking_uri=None, config=TrainConfig()) -> str
evaluate(run_dir, tracking_uri=None, model_name=MODEL_NAME, config=TrainConfig()) -> str
decide_promotion(run_dir, tracking_uri=None) -> "promote" | "skip"   # id задачи-ветки
register_challenger(run_dir, tracking_uri=None, model_name=MODEL_NAME) -> str   # задача promote
set_champion_alias(run_dir, tracking_uri=None, model_name=MODEL_NAME) -> str
log_reason(run_dir, tracking_uri=None) -> str
# tracking.py
ensure_run(run_dir_json: Path, experiment, tags) -> str   # MLflow run id, переиспользуется
register_version(model_uri, run_id, tags, name) -> str     # идемпотентно по run_id
set_champion(name, version, expected_previous) -> None     # падает, если champion сдвинулся
# train.py
fit_history_als(split, config) -> ALSCandidateGenerator
evaluate_variants(split, movies, users, ranker, als, config) -> TrainResult
```

`evaluate` пишет `evaluate.json`: метрики popularity / als_only / als_ranker (известные пользователи), `challenger_foldin` и `champion_foldin` (fold-in на истории тестовых пользователей), выбранный вариант, `beats_popularity`, `model_uri` залогированной модели, версия champion'а на момент оценки. `decide_promotion` сравнивает fold-in с fold-in через `tracking.decide_promotion` (допуск 0.005, coverage −10%).

Тесты (синтетика ~60 пользователей, sqlite-state и sqlite-MLflow во tmp): каждое правило валидации падает отдельно (NULL, 0 и 6, дубликат, падение на 6% при допуске 5%, пустой снапшот), падение на 4% проходит, при падении запись в `retrain_runs` не появляется; полный прогон validate → … → set_champion_alias без чемпиона → promote; повтор каждой задачи с тем же run → те же файлы/метрики, одна версия в реестре; второй запуск при наличии чемпиона: в MLflow есть `champion_*` метрики, ветка выбирается правилом; ветки `decide_promotion` (нет чемпиона, хуже чемпиона, coverage упал, не лучше popularity); `set_champion_alias` падает, если alias сдвинули после оценки; `log_reason` пишет теги; `web_share` в тегах > 0, если в снапшоте есть веб.

### Task 4: `reload_api` и `bootstrap` (TDD)

**Files:** `src/recsys/pipelines/reload.py`, `src/recsys/pipelines/bootstrap.py`; тесты `test_reload_api.py`, `test_bootstrap.py`.

```python
reload_api(run_dir, api_url=None, admin_token=None, timeout=60) -> str
    # POST {api_url}/admin/reload-model, Bearer; 200 loaded|unchanged и version == продвинутой -> ok
    # 401/409/502/5xx/сетевой сбой -> ReloadError (ретраи Airflow), alias не трогаем
download(raw_root) -> str ; verify_checksum(archive) -> str ; load_postgres(archive, database_url) -> dict
```

Тесты: тестовый HTTP-сервер проверяет заголовок и путь; `loaded` нужной версии → ok; `unchanged` той же версии → ok; другая версия → ошибка; 502/409/401 → ошибка с текстом ответа; недоступный порт → ошибка; пустой токен → ошибка до запроса. Bootstrap: `file://` архив, MD5 не совпал → ошибка; повтор не качает заново.

### Task 5: DAG-файлы и DAG-тест

**Files:** `airflow/dags/bootstrap_data.py`, `airflow/dags/retrain_daily.py`, `tests/dags/test_dag_integrity.py`.

- TaskFlow API, `@task.external_python(python=RECSYS_PYTHON, expect_airflow=False)` с импортом внутри тела; ветка — `@task.branch_external_python`; `skip` — `EmptyOperator`.
- `default_args`: `retries=2`, `retry_delay=2 мин`; DAG: `catchup=False`, `max_active_runs=1`, `dagrun_timeout`, `params={"include_web": True}`, `render_template_as_native_obj=True`.
- Тест: `DagBag(include_examples=False)` без ошибок импорта; ровно два DAG; id задач и рёбра совпадают со spec 4.2; `retries == 2`, `catchup is False`, `max_active_runs == 1`, расписание `@daily` / `None`; внутри образа `/opt/recsys-venv/bin/python -c "import recsys.pipelines.retrain"` проходит.

### Task 6: Образ Airflow, compose, `.env`

**Files:** `docker/airflow.Dockerfile`, `docker-compose.yml`, `.env.example`, `.dockerignore`, `infra/postgres/init/01-create-databases.sh`; локальный `.env` (не в git).

- Builder-стадия на том же базовом образе: `uv sync --frozen --no-default-groups` в `/opt/recsys-venv` (python из образа, `UV_PYTHON_DOWNLOADS=never`). Финальная стадия: `libgomp1`, `psycopg2-binary` с constraints Airflow, `/data` принадлежит `airflow:0`, DAG-файлы копируются в образ.
- Compose: `db-init` (one-shot, тот же init-скрипт против работающего Postgres: на существующем томе создаёт БД `airflow`), `airflow-init` (`airflow db migrate`), `airflow-apiserver` (UI на `127.0.0.1:${AIRFLOW_PORT:-8080}`, пароль админа из `.env`), `airflow-scheduler` (LocalExecutor, задачи выполняются здесь), `airflow-dag-processor`. Healthcheck у каждого. Тома `airflow-data:/data`, `airflow-logs`.
- `.env.example`: `AIRFLOW_DB_PASSWORD`, `AIRFLOW_FERNET_KEY`, `AIRFLOW_JWT_SECRET`, `AIRFLOW_API_SECRET_KEY`, `AIRFLOW_ADMIN_PASSWORD`, `AIRFLOW_PORT`.

### Task 7: CI

Новый job `airflow`: сборка образа Airflow, `docker run` с примонтированным репозиторием: `pip install pytest` + `python -m pytest tests/dags`; проверка, что в venv задач `import recsys.pipelines.retrain` работает и `sqlalchemy` там 2.1, а в окружении Airflow 2.0. Существующие jobs без изменений.

### Task 8: E2E и отчёт

`docker compose up -d --build` → `bootstrap_data` (на загруженном томе идемпотентен) → несколько веб-оценок через API → `retrain_daily` → состояния задач, метрики MLflow (challenger vs champion), ветка, `/ready` с новой версией, тег `web_share > 0`, `docker stats --no-stream`, `docker compose down` (тома сохраняются). Логи в `.superpowers/sdd/overnight/s4-e2e/`. Отчёт в `.superpowers/sdd/overnight/stage-4-report.md`.

---

## Decisions

1. **Airflow 3.3.2, а не 3.4.0.** 3.4.0 вышел 5 дней назад (2026-11-06), 3.3.2 — патч-релиз от 2026-09-17. Для учебного проекта на один сервер свежие баги минорного релиза не нужны. Обновление — смена одного `ARG`. *Если неправ:* нет новых фич 3.4, обновление стоит одной пересборки.
2. **Отдельный venv `/opt/recsys-venv` + `@task.external_python` (вариант b).** Constraints Airflow 3.3.2 для py3.12 фиксируют SQLAlchemy 2.0.52, pandas 3.0.5, pyarrow 25.0.1, numpy 2.5.3, а у нас в `uv.lock` SQLAlchemy 2.1.8, pandas 3.1.1, pyarrow 26.0.1, numpy 2.5.4. Установка в окружение Airflow либо ломает Airflow (SQLAlchemy 2.1), либо требует даунгрейда наших зависимостей и второго набора версий, отличного от образа API. Отдельный venv собирается из того же `uv.lock`, поэтому обучение в Airflow идёт на тех же версиях, что и инференс в API. `external_python` предпочтён `BashOperator`: остаётся TaskFlow API, возвращаемое значение сразу попадает в XCom, а ветвление делает `@task.branch_external_python`. *Цена ошибки:* каждый запуск задачи — новый процесс Python (+1–3 с на импорт), функции в DAG не могут ссылаться на внешние имена (импорт внутри тела).
3. **SimpleAuthManager (по умолчанию в Airflow 3), а не FAB.** Один администратор, UI доступен только через SSH-туннель (spec 6.2). FAB добавил бы Flask-AppBuilder, свою БД пользователей и ~150 МБ RSS. Пароль задаётся в `.env` (`AIRFLOW_ADMIN_PASSWORD`). api-server перед стартом пишет файл паролей, Airflow его читает и ничего не генерирует. *Цена:* пароль лежит открытым текстом в файле внутри контейнера, Airflow пишет в лог предупреждение про «production-shaped» развёртывание. Для нескольких пользователей или SSO нужен FAB или Keycloak.
4. **Состояние между запусками — таблица `retrain_runs` в БД `app`, а не манифест прошлой папки.** Postgres — источник правды (spec 1), папки `/data/runs` можно чистить. Строка пишется только после успешной валидации, upsert по `run_id`. «Прошлый запуск» — самый свежий **другой** run с тем же `include_web`: обучение без веб-данных не должно ложно срабатывать на правиле −5%. *Цена:* лишняя таблица в `app`, `create_all` создаёт её без миграции.
5. **`drift_report` на этапе 4 — JSON-сводка, Evidently на этапе 5.** `recsys.drift.build_drift_report(current, previous)` — единственная точка замены. Сводка: `n_ratings`, `n_web_ratings`, `web_share`, `n_users`, доли оценок 1..5 (всё и только веб), те же числа прошлого запуска и дельты. Пишется в `run_dir/drift.json`, в MLflow (артефакт `drift/drift.json`, метрики `drift_*`) и в лог задачи. Задача не падает на дрифте: это сигнал, а не тревога (spec 5.3). *Цена:* до этапа 5 нет статистических тестов и отчёта по жанрам и годам.
6. **Снапшот замораживается первой успешной выгрузкой.** Таблица `ratings` меняется каждую минуту, повторная выгрузка через час дала бы другие данные. Если в `run_dir/extract/` уже лежит полный снапшот (запись атомарная), `extract_ratings` его переиспользует. Остальные задачи всегда пересчитывают результат и детерминированы (сиды). *Цена:* чтобы перевыгрузить данные для того же run, нужен новый run (или удалить папку), «Clear» в UI переиспользует снапшот.
7. **Challenger и champion сравниваются по fold-in на одном сплите.** Champion загружается из Registry и получает историю тестовых пользователей текущего снапшота через `recommend_for_ratings`. Его ALS не переобучается, это и есть «fold-in на новом снапшоте». Challenger для решения оценивается так же, через fold-in, а не через свои обученные факторы, потому что API обслуживает веб-посетителей именно так. Метрики popularity / ALS-only / ALS+ranker по факторам логируются для сравнения базовых линий (spec 2.5). *Цена:* fold-in для ~6000 пользователей добавляет к `evaluate` ~10–30 с. Сравнение «по факторам» показало бы challenger чуть лучше, чем он работает в сервисе.
8. **Ветки:** `promote` регистрирует версию (теги: метрики, снапшот, вариант, причина), `set_champion_alias` переносит alias, `reload_api` перезагружает API. `skip` — `EmptyOperator`, видимый в UI, `log_reason` пишет причину в MLflow run и в лог. Отклонённый challenger **не регистрируется**: в Registry только модели, прошедшие проверку, отклонённая остаётся залогированной моделью в run, её можно зарегистрировать руками. *Цена:* в Registry не видно, сколько кандидатов отвергнуто (это видно в экспериментах MLflow по тегу `promoted=false`).
9. **`set_champion_alias` с защитой от гонки.** Если alias сдвинулся между `evaluate` и `set_champion_alias` (ручной `trainer`, второй DAG), задача падает и не перезаписывает чужое решение. `max_active_runs=1` исключает гонку внутри Airflow. *Цена:* ручное вмешательство во время запуска требует перезапуска DAG.
10. **`reload_api` при сбое падает после ретраев, промоушен не откатывается.** Alias — источник правды. API без перезагрузки продолжает отдавать прошлую версию, подхватит новую при рестарте или следующем вызове reload. Красная задача в UI видна, тихое предупреждение пропустили бы. Успех — только `loaded`/`unchanged` с `version` равной продвинутой. *Цена:* DAG-run красный, хотя модель продвинута. Это осознанно: «модель продвинута, но не обслуживается» требует внимания.
11. **Отдельный том `airflow-data` (`/data`), а не том `movielens` из этапа 3.** Тот принадлежит uid 10001 (пользователь `app` образа API), Airflow работает под uid 50000. `bootstrap_data` один раз скачает 6 МБ архива в свой том (повтор проверяет MD5 и не качает). *Цена:* архив хранится в двух томах.
12. **DAG-файлы копируются в образ**, а не монтируются с хоста. Код и DAG выкатываются вместе по тегу образа (spec 6.4), bind-mount с Windows медленный и с другими правами. *Цена:* правка DAG требует `docker compose up -d --build`.
13. **`db-init` в compose.** Init-скрипт Postgres выполняется только на пустом томе, а у нас том уже с данными этапа 3. One-shot сервис запускает тот же идемпотентный скрипт против работающего сервера при каждом `up`: создаёт БД `airflow`, если её нет, и синхронизирует пароли ролей с `.env`. *Цена:* +1 короткоживущий контейнер при старте.
14. **Новые DAG стоят на паузе при создании** (по умолчанию в Airflow). Пока `bootstrap_data` не отработал, `retrain_daily` упал бы на пустом снапшоте. После bootstrap один раз: `airflow dags unpause retrain_daily`. *Цена:* лишний шаг при первом запуске.
15. **`retries=2` для всех задач, включая `validate_data`.** Spec 4.1. Детерминированная ошибка валидации повторится три раза и упадёт, это лишние ~4 минуты, зато транзиентный сбой БД при чтении `retrain_runs` ретраится. Задачи вне Airflow-окружения не могут бросить `AirflowFailException`.

### Решения и находки при реализации

16. **Airflow 3 открывает и синхронный, и асинхронный движок метаданных.** Для psycopg2-URL он строит `postgresql+asyncpg://…`, и `airflow db migrate` падал с `No module named 'asyncpg'`. В образ ставятся `psycopg2-binary` и `asyncpg` (оба с constraints Airflow).
17. **`run_id` нельзя называть аргументом TaskFlow-задачи.** Это ключ контекста Airflow. Ошибка (`The key 'run_id' in args is a part of kwargs and therefore reserved`) возникает только при выполнении, DagBag её не видит. Аргумент переименован в `airflow_run_id`, DAG-тест сверяет сигнатуры всех задач с `KNOWN_CONTEXT_KEYS`. Первый запуск в e2e упал именно на этом, ретрай после исправления прошёл (try 3).
18. **`DagBag(include_examples=…)` в Airflow 3.3 убран**, `dagbag.get_dag()` ходит в БД метаданных. Тест берёт `dagbag.dags[...]`, примеры выключены через `AIRFLOW__CORE__LOAD_EXAMPLES=false` образа.
19. **Порядок в `retrain_runs` строго монотонный.** На Windows две быстрые вставки получали одинаковый `recorded_at`, тест «предыдущего запуска» был нестабилен (1 из ~6 прогонов). Теперь `recorded_at = max(now, last + 1 мкс)`.
20. **Шум в логах задач:** `ModuleNotFoundError: No module named 'pendulum'`. Это проба самого `ExternalPythonOperator` (`python -c "import pendulum"`): по ней он решает, какие ключи контекста можно передать. В recsys-venv pendulum нет, проба печатает трейсбек, оператор продолжает работу. На результат это не влияет.
21. **Снятие с паузы создаёт scheduled-run** за последний интервал (`catchup=False` → ровно один). Ручной запуск, сделанный в это же время, ждёт его из-за `max_active_runs=1`. В e2e это и было видно: оба запуска прошли друг за другом.
22. **Дисперсия ранжировщика сравнима с допуском.** На одном и том же снапшоте два запуска дали одинаковые метрики (детерминизм подтверждён). Но добавление 1558 веб-оценок (0,16% данных) сдвинуло NDCG@10 ALS+ranker с 0.0793 до 0.0834, а ALS-only почти не изменился (0.0668 → 0.0672). При допуске 0.005 модель, обученная без веб-данных и худшая на 0.0037, тоже была продвинута. Так и задумано spec 4.2, но допуск стоит пересмотреть после нескольких недель истории метрик (этап 5).

### Решения по итогам ревью (`.superpowers/sdd/overnight/s4-review.md`)

23. **Запуск `include_web=false` по умолчанию только оценочный (ревью I1).** Он проходит всю цепочку, сравнивается с champion'ом и логируется в MLflow, но `decide_promotion` всегда уходит в `skip` с причиной `include_web=false runs are evaluation-only …`. Почему: такая модель измерена без веб-посетителей, единственной живой аудитории API, а допуск 0.005 «ради свежести» рассчитан на модели одного рода на одних данных. Без этого правила любой эксперимент «а без веба?» молча заменял живую модель (так в e2e champion'ом стала v5). Повышение возможно только через явный параметр DAG `allow_promotion=true` (инцидент «веб-данные отравлены»). В этом случае действует обычное правило spec 4.2 с допуском: человек уже принял решение, а шлюзы popularity и coverage остаются. Для `include_web=true` параметр ничего не меняет. *Цена:* в инциденте нужен ручной запуск с двумя параметрами. Отклонённая оценочная модель, как и любая отклонённая (Decision 8), не регистрируется.
24. **Происхождение модели видно в Registry.** У каждой версии есть теги `include_web`, `n_web_ratings`, `allow_promotion` и `data_snapshot`. У версий от stage-3 `trainer` стоят `include_web=false` и `n_web_ratings=0`, потому что он читает сырые файлы MovieLens.
25. **Флаги разбираются строго (`retrain.parse_flag`, ревью M6).** Принимаются bool, 0/1 и строки `true/false/1/0/yes/no/on/off` в любом регистре. Всё остальное даёт `ValueError`, а не `bool("False") is True`. `render_template_as_native_obj=True` остаётся, и DAG-тест проверяет, что он включён. *Цена:* опечатка в conf ручного запуска роняет задачу, а не трактуется молча.
26. **Регистрация идемпотентна по реестру, а не по тегу run (ревью I2).** `register_version` сначала ищет `search_model_versions("name='…' and run_id='…'")` и переиспользует найденную версию. Затем всегда дописывает все теги и ставит `registered_version` на run. Ретрай после падения в любой точке (после `register_model`, между тегами) доводит регистрацию до конца, а не создаёт версию N+1. Тест симулирует обрыв на втором теге.
27. **Секреты Airflow проверяются при старте контейнера (ревью I3).** `airflow-init` сначала запускает `python -m recsys.secrets_check` из recsys-venv и только потом `airflow db migrate`. Проверяются `AIRFLOW_ADMIN_PASSWORD`, `AIRFLOW_JWT_SECRET`, `AIRFLOW_API_SECRET_KEY` и `AIRFLOW_DB_PASSWORD`: не плейсхолдер (`change…`, `replace…`, `your-…`, `<…`) и не короче 32 символов. Политика та же, что у настроек API, код общий (`recsys.secrets_check.secret_problem`). `AIRFLOW_FERNET_KEY` должен быть urlsafe-base64 от 32 байт (та же проверка, что в `Fernet(key)`, без зависимости от cryptography). Остальные контейнеры Airflow ждут `service_completed_successfully`, поэтому весь Airflow не поднимается, а в логе перечислены имена переменных без значений. Проверка идёт при старте, а не через `:?` в compose, поэтому `docker compose config` с `.env.example` в CI остаётся валидным. *Цена:* пароли Postgres для `app` и `mlflow` этой проверкой не покрыты (оставлено, см. отчёт фиксера).
28. **Границы правила включительные с эпсилоном 1e-9 (ревью M1).** `0.2 * 0.9 = 0.18000000000000002`, и coverage ровно −10% раньше отклонялся. Граничные случаи NDCG −0.005 и coverage −10% закреплены тестами.
29. **Повтор после promote согласован (ревью M2, M3).** Если в `promotion.json` уже стоит `promoted: true`, `evaluate` не пересчитывает и не перелогирует модель (иначе модель сравнивалась бы сама с собой). `decide_promotion` возвращает `promote` с причиной `already promoted vN in this run (…)`, `register_challenger` не перезаписывает `promotion.json`, а перенос alias остаётся no-op. `log_reason` пишет текущий alias из Registry, а не версию, которую видел `evaluate`.
30. **Аварийный выход из правил количества и контроль веб-оценок (ревью M5).** Новое правило: если в прошлом запуске `include_web=true` были веб-оценки, а теперь их 0, это ошибка валидации. Потерю всех веб-оценок (~0.2% данных) правило −5% не видит, а это единственные невосстановимые данные (spec 6.2). Параметр DAG `allow_count_drop=true` отключает на один запуск оба правила количества, −5% и «веб пропал» (после сознательной чистки). Запуск записывается в `retrain_runs` и становится новой базой. Без этого после чистки падал бы каждый следующий запуск: упавшие запуски не записываются, база не сдвигается. Храповик из нескольких −4% подряд по-прежнему проходит, это буква spec.
31. **Папка чужого run не переиспользуется (ревью M9).** `safe_run_id` не инъективен (`a:b` и `a-b`). `extract_ratings` сверяет `run_id` и `include_web` из `extract.json` и падает при несовпадении.
32. **CI-job `airflow`: pytest ставится с constraints Airflow 3.3.2** и не может сдвинуть `packaging`/`pluggy` в тестируемом окружении. `timeout-minutes: 30`.
