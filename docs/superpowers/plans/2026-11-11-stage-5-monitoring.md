# Этап 5: мониторинг (Prometheus, Grafana, логи показов, Evidently, алерты). План реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Три уровня мониторинга из spec 5 работают локально в том же `docker compose up -d --build`:
1. **Сервис жив и быстр?** API отдаёт `/metrics` (метрики spec 5.1), Prometheus их собирает, на дашборде Grafana видны RPS, p95, доля ошибок, доля fallback, hit rate кэша, оценки в час, версия и возраст модели.
2. **Рекомендации полезны?** Каждый показ рекомендаций пишется в `recommendation_impressions` (без задержки ответа). Онлайн-метрика (доля показанного, что оценили в течение часа, и средняя оценка) считается SQL-представлением и видна в Grafana.
3. **Данные не уплыли?** Задача `drift_report` в `retrain_daily` строит отчёт Evidently (MovieLens против веб-оценок за 7 дней, плюс дрифт рекомендаций), кладёт HTML в MLflow, `drift_share` в MLflow и в Postgres (`drift_metrics`) для Grafana.
Пять алертов spec 5.5 заведены в Grafana Alerting с контакт-точкой Telegram (токен из `.env`; без токена контакт-точки нет, алерты видны только в UI). Публичная read-only копия дашборда — Grafana public dashboard по ссылке без логина.

**Architecture:**
- **API.** Новый модуль `recsys_api/metrics.py`: свой `CollectorRegistry` (не глобальный, чтобы тесты и несколько `create_app` не конфликтовали), чистый ASGI-middleware для `http_requests_total{endpoint,status}` и `http_request_duration_seconds{endpoint}` (endpoint = шаблон маршрута из `scope["route"]`, неизвестный путь → `unmatched`), и `MetricsObserver` поверх существующего крючка `Observer` (этап 3 оставил его ровно для этого). `recsys_api/impressions.py`: `ImpressionLogger` — ограниченная очередь + фоновый поток, пишет пачками через `PgRepository.insert_impressions`. Отказ БД или переполнение очереди → показы теряются и считаются в метрике, ответ не страдает. `CompositeObserver` объединяет метрики и показы.
- **Схема `app`.** Новые таблицы через тот же `create_all` (`recsys.db.schema`): `recommendation_impressions`, `drift_metrics`, `run_metrics`. Плюс два представления (Postgres-only DDL в `create_schema`): `recommendation_outcomes` (первый показ пары user–movie + оценил ли в течение часа) и `online_quality_daily`. Роль `grafana_reader` создаёт init-скрипт Postgres, а `create_schema` выдаёт ей `SELECT` только на объекты из `DASHBOARD_OBJECTS` (представления и две таблицы метрик, не сырые `ratings`/`users`).
- **Дрифт.** `recsys/drift.py`: Evidently импортируется лениво (в образе API его нет). Два набора данных: «оценки» (rating, genre, release_year, user_activity) и «рекомендации» (popularity, genre, release_year), reference — MovieLens из снапшота, current — веб-оценки за 7 дней до момента выгрузки и показы `path=full` за те же 7 дней. Мало данных → отчёт «insufficient data», `drift_share = NULL`, задача не падает. Лёгкая сводка этапа 4 (дельты к прошлому запуску) остаётся в `drift.json`.
- **NDCG для Grafana.** `evaluate` дополнительно пишет свои метрики в `run_metrics(run_id, metric, value, recorded_at)`: Grafana читает Postgres, а не внутреннюю схему MLflow.
- **Prometheus + Grafana в compose.** `infra/prometheus/prometheus.yml` (scrape api и себя, retention 7 дней / 512 МБ), `infra/grafana/provisioning/{datasources,dashboards,alerting}`, `infra/grafana/dashboards/recsys.json`, `infra/grafana/telegram.yml` (подключается стартовым скриптом, только если заданы оба `TELEGRAM_*`). `grafana-init` (образ API) проверяет секреты Grafana той же политикой `recsys.secrets_check`, `grafana-publish` (образ API) идемпотентно включает public dashboard через HTTP API Grafana и печатает ссылку.
- **Логи.** `recsys/logs.py`: JSON-форматтер (stdlib) — одна строка JSON на событие в stdout. API настраивает им корень и логгеры uvicorn; access-лог uvicorn выключен, вместо него middleware пишет одну строку на запрос (метод, шаблон маршрута, статус, длительность).

**Tech Stack:** prometheus-client 0.26, Prometheus v3.15.1, Grafana OSS 13.2.5, Evidently 0.7.23 (группа зависимостей `drift`), Postgres 16, FastAPI, Airflow 3.3.2 (без изменений DAG-графа).

**Spec:** `docs/superpowers/specs/2026-11-11-movielens-recsys-design.md`, секция 5 (основная), 1, 6.

## Global Constraints

- Всё из планов этапов 0–4 в силе: `src/`-layout, зависимости через `uv add`/`uv lock` (+ `uv.lock` в git), ruff (line-length 100, py312, E,F,I,B,UP), код и коммиты на английском, комментарии в конфигах объясняют «почему». Каждый коммит заканчивается пустой строкой и `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Не пушим, `--no-verify` не используем, README не создаём.
- Unit-тесты без Docker и сети (SQLite, фейки). Интеграционные (`integration`) — против Postgres/Redis compose (`POSTGRES_PORT=15432` локально). DAG-тесты — в образе Airflow.
- Память: весь стек ≤ ~4,5 ГБ из 6 ГБ WSL (`docker stats --no-stream`). Grafana без предустановки плагинов (`GF_PLUGINS_PREINSTALL_DISABLED`): по умолчанию Grafana 13 при старте качает ~180 МБ плагинов и занимает ~660 МБ RAM, без них ~250 МБ.
- Все порты на `127.0.0.1`. `/metrics` в compose отдаётся на порту API; **на этапе 6 Caddy обязан закрыть `/metrics` снаружи** (`respond /metrics 404`), Prometheus ходит к API внутри сети compose.
- Живой champion v6 не трогаем руками; новый запуск `retrain_daily` может продвинуть новую версию по правилу spec 4.2.
- Горячий путь API не ждёт ни Postgres (показы), ни Prometheus.

## Review Focus (топ-5 способов ошибиться)

1. **Кардинальность меток.** `endpoint` — шаблон маршрута (`/movies/search`, а не `/movies/search?q=…`), неизвестные пути сводятся к одному значению `unmatched`, `/static/*` — к `/static`. `status` — код ответа. `model_info{version}` держит ровно одну серию: при смене версии старая метка удаляется. Тест прогоняет 100 случайных путей и проверяет число серий.
2. **Показы не ломают и не тормозят ответ.** Запись в фоне, пачками; очередь ограничена; падение БД, ошибка SQL, переполнение очереди → ответ 200 как обычно, потеря видна в `impressions_dropped_total`/`impressions_write_errors_total`. Тест с «зависшим» репозиторием проверяет, что `/recommendations` отвечает сразу.
3. **Онлайн-метрика считает то, что заявлено.** Один показ на пару (user, movie) — первый; «оценил в течение часа» = `shown_at ≤ rated_at < shown_at + 1h`; повторные показы одного фильма не умножают знаменатель. Интеграционный тест на Postgres с ручным расчётом.
4. **Дрифт не роняет переобучение.** Нет веб-оценок, мало веб-оценок, нет показов, нет таблицы показов, Evidently упал на данных → отчёт со статусом, `drift_share` NULL, задача зелёная. Evidently в образе API отсутствует, его импорт не должен ломать `recsys.pipelines.*`.
5. **Публичный доступ только на чтение и только к нужному.** Public dashboard исполняет лишь запросы своих панелей (не произвольный SQL), а сама роль Postgres `grafana_reader` read-only (`default_transaction_read_only`, `statement_timeout`) и видит только `DASHBOARD_OBJECTS`. Пароль админа Grafana и `GF_SECURITY_SECRET_KEY` проходят ту же проверку на плейсхолдер/длину, что и остальные секреты.

---

## Файловая структура (что добавится)

```
src/recsys/logs.py                         # JSON-логи (stdlib), общий для API и задач
src/recsys/drift.py                        # + Evidently: датасеты, отчёт, insufficient data
src/recsys/db/schema.py                    # + recommendation_impressions, drift_metrics, run_metrics,
                                           #   представления, гранты grafana_reader
src/recsys/db/monitoring.py                # запись drift_metrics / run_metrics, чтение показов
src/recsys/secrets_check.py                # + группа grafana
src/recsys/pipelines/retrain.py            # drift_report на Evidently, evaluate -> run_metrics
services/api/recsys_api/metrics.py         # реестр, middleware, MetricsObserver
services/api/recsys_api/impressions.py     # ImpressionLogger (очередь + поток)
services/api/recsys_api/observer.py        # + CompositeObserver
services/api/recsys_api/{main,model,repository,settings}.py
infra/prometheus/prometheus.yml
infra/grafana/start.sh                     # собирает provisioning, Telegram только при заданном токене
infra/grafana/telegram.yml                 # контакт-точка + политика уведомлений
infra/grafana/provisioning/datasources/datasources.yml
infra/grafana/provisioning/dashboards/dashboards.yml
infra/grafana/provisioning/alerting/rules.yml
infra/grafana/dashboards/recsys.json
infra/grafana/publish_dashboard.py         # включает public dashboard (stdlib urllib)
infra/scripts/traffic.py                   # генератор трафика для e2e
infra/postgres/init/01-create-databases.sh # + роль grafana_reader
tests/unit/test_api_metrics.py, test_impressions.py, test_logs.py, test_monitoring_config.py
tests/unit/test_drift.py, test_retrain_tasks.py, test_secrets_check.py (дополнены)
tests/integration/test_impressions_db.py
```

### Task 1: План (этот файл) — коммит

### Task 2: JSON-логи и метрики API (TDD)
- [ ] `recsys/logs.py`: `JsonFormatter` (ts, level, logger, message, extra-поля, exc), `configure_json_logging(level)`. Тест: строка — валидный JSON, extra и исключение попадают в поля.
- [ ] `recsys_api/metrics.py`: `ApiMetrics(registry)` с метриками spec 5.1 + `model_created_timestamp_seconds`, `impressions_*`; `MetricsMiddleware` (ASGI); `MetricsObserver`. Метки предзаполнены (path, result, source), чтобы `rate()` видел нули.
- [ ] `ModelHolder`/`load_champion`: время создания версии в Registry (`created_at`) → `model_created_timestamp_seconds`.
- [ ] `GET /metrics` (`text/plain; version=0.0.4`), `Settings.log_format` (`json`|`text`).
- [ ] Тесты: имена/типы/метки, endpoint = шаблон, 404 → `unmatched`, 500 → `status="500"`, `model_info` одна серия после reload, кэш hit/miss, `ratings_received_total{source="web"}`.

### Task 3: Показы и онлайн-метрика (TDD)
- [ ] Схема: `recommendation_impressions(id, user_id, movie_id, position, model_version, path, shown_at)` + индексы; представления; `DASHBOARD_OBJECTS` + гранты.
- [ ] `PgRepository.insert_impressions(rows)`; `ImpressionLogger(record/close)`.
- [ ] Unit: пачки, неблокирующая запись при «зависшем» sink, переполнение, ошибки sink, flush при закрытии, анонимные (без user_id) не пишутся.
- [ ] Integration: вставка, онлайн-метрика по представлению против ручного расчёта, грант `grafana_reader` (если роль есть).

### Task 4: Evidently в `drift_report` (TDD)
- [ ] `drift.py`: `rating_frames`, `recommendation_frames`, `evidently_report(reference, current, …)` → `{status, drift_share, columns, html}`; сводка этапа 4 под именем `summary_report`.
- [ ] `retrain.drift_report`: HTML-артефакты, метрики `drift_share`/`prediction_drift_share`, теги статуса, upsert в `drift_metrics`. `evaluate` → `run_metrics`.
- [ ] Тесты: явный дрифт → share > 0; одинаковые распределения → 0; мало данных → NULL и HTML «insufficient data»; ошибка Evidently → статус `error`, задача не падает; повтор задачи — одна строка в `drift_metrics`.

### Task 5: Prometheus, Grafana, алерты, compose
- [ ] `prometheus.yml`, provisioning, дашборд, правила алертов, `telegram.yml`, `start.sh`, `publish_dashboard.py`.
- [ ] compose: `prometheus`, `grafana-init`, `grafana`, `grafana-publish`; `api` ждёт `db-init`; `.env.example` (+ `GRAFANA_*`, `TELEGRAM_*`); init-скрипт с ролью `grafana_reader`.
- [ ] Образ Airflow: `uv sync --group drift`. Образ API: `--no-access-log`.
- [ ] Тест конфигов: JSON/YAML парсятся, все uid источников данных существуют, панели spec 5.4 на месте, каждая метрика из выражений дашборда и алертов есть в реестре API, пять правил spec 5.5 с порогами.

### Task 6: CI
- [ ] docker job: `promtool check config` в образе Prometheus. Остальное (pytest видит новые тесты, образ Airflow собирается с drift) — без новых job'ов.

### Task 7: E2E и отчёт
`docker compose up -d --build`, трафик скриптом (несколько посетителей, включая холодный старт), выдержка `/metrics`, targets Prometheus `up`, health источников Grafana и запросы панелей через `/api/ds/query`, `retrain_daily` → HTML Evidently в MLflow + `drift_share` в Postgres, строки показов + онлайн-метрика, правила алертов через API, срабатывание «API недоступен» при остановленном API, `docker stats --no-stream`, `docker compose down` (тома остаются). Логи — `.superpowers/sdd/overnight/s5-e2e/`, отчёт — `.superpowers/sdd/overnight/stage-5-report.md`.

---

## Decisions

1. **Evidently 0.7.23 в отдельной группе зависимостей `drift`.** Нужен только задаче `drift_report`, а тянет ~40 пакетов (plotly, statsmodels, litestar, nltk…). Группа входит в `default-groups` (локальная разработка, CI), образ Airflow ставит её явно (`--group drift`), образ API — нет. Импорт ленивый, внутри функции. *Цена ошибки:* если кто-то импортирует Evidently на уровне модуля в общем коде, сломается образ API; от этого защищает тест «`recsys.pipelines.retrain` импортируется без evidently».
2. **Evidently 0.7.23 и pandas 3: шим одной функции.** Evidently при импорте тянет `pandas.core.dtypes.common.infer_dtype_from_object`, которую pandas 3 удалил; используется она только в legacy-тесте `TestColumnsType`, который мы не вызываем. `recsys.drift` перед импортом подставляет эквивалент (`pandas_dtype(x).type`). Откат pandas до 2.x ради отчёта сломал бы общий для API и обучения `uv.lock`, третий venv в образе Airflow — дороже. *Цена:* при обновлении Evidently/pandas шим может стать лишним или недостаточным; unit-тест реально строит отчёт Evidently и упадёт первым.
3. **Методы дрифта зафиксированы: Wasserstein (normed) для чисел, Jensen-Shannon для категорий, порог 0.1** (умолчания Evidently для reference > 1000 строк). Без фиксации Evidently на маленьких выборках переключается на p-value тесты, и смысл `drift_share` менялся бы от размера данных. Reference — случайная выборка 50 000 строк MovieLens (сид 0): отчёт строится за секунды и весит единицы МБ. *Цена:* distance-метрики на малых current шумные, поэтому нижний порог `MIN_CURRENT_ROWS = 50`.
4. **Признаки на строку оценки:** `rating` (категория), `genre` (категория: один жанр фильма, выбранный случайно с сидом, так что многожанровый фильм вносит каждый жанр с весом 1/n), `release_year` (число), `user_activity` (число оценок автора в снапшоте). **Дрифт рекомендаций:** reference — фильмы, которые оценивала аудитория MovieLens, current — показы `path=full` за 7 дней; признаки `popularity` (число оценок фильма в снапшоте), `genre`, `release_year`. Это сравнение «что советуем» с «что смотрела исходная аудитория». Показы холодного старта исключены: это фиксированное правило, а не модель. *Цена:* «рекомендации против потребления» ожидаемо дрейфуют (модель любит популярное), поэтому `prediction_drift_share` хранится отдельно и в `drift_share` не входит.
5. **Окно current — 7 дней до `extracted_at` снапшота**, а не до «сейчас»: ретрай задачи считает то же самое (spec 4.1). Показы читаются из Postgres в окне до того же момента. Для `include_web=false` веб-оценок в снапшоте нет → дрифт оценок «insufficient data», дрифт рекомендаций считается.
6. **`drift_metrics` — отдельная таблица (spec 1), одна строка на run:** `drift_share`, `prediction_drift_share` (NULL при нехватке данных), статусы, размеры выборок, `details` JSON. Upsert по `run_id` (повтор задачи не дублирует строку). Grafana рисует `drift_share` по `computed_at`.
7. **NDCG для дашборда — таблица `run_metrics` в `app`, а не чтение БД `mlflow`.** Внутренняя схема MLflow не контракт и меняется между версиями; `evaluate` и так знает метрики. *Цена:* метрики дублируются (MLflow — источник правды для экспериментов, `run_metrics` — витрина для Grafana).
8. **Возраст модели = время создания версии в Registry (`model_created_timestamp_seconds`), а не `model_loaded_timestamp_seconds`.** Время загрузки сбрасывается при каждом рестарте API и скрыло бы сломанное переобучение; время создания версии растёт, пока не появится новый champion. `model_loaded_timestamp_seconds` остаётся (spec 5.1) и показывается на дашборде. *Цена:* алерт сработает и тогда, когда переобучение работает, но 36 ч подряд отклоняет challenger'а — это тоже «модель устарела», повод посмотреть.
9. **Показы: только посетители с `user_id`.** Без cookie показ нельзя связать с будущей оценкой. Кэшированные ответы — тоже показы (их увидели). В таблицу добавлена колонка `path` (сверх spec 5.2): онлайн-метрику модели честно считать по `full`, а не по популярному холодному старту.
10. **Онлайн-метрика по первому показу пары (user, movie).** Страница обновляет рекомендации после каждой оценки, один фильм показывается много раз; считать каждый показ значило бы раздувать знаменатель. Окно: `shown_at ≤ rated_at < shown_at + 1 h`. Если оценку потом изменили, `rated_at` сдвигается и показ может выпасть из окна — консервативно. *Цена:* показы моложе часа ещё «не созрели» и занижают долю на дашборде за последний час.
11. **Запись показов — очередь на 10 000 строк + фоновый поток, пачки до 500 строк или раз в секунду.** Ни ретраев, ни буфера на диске: при недоступной БД пачка теряется и считается в `impressions_write_errors_total`. Показы — аналитика, а не транзакция. *Цена:* при падении Postgres теряются показы за время простоя.
12. **Публичная копия — Grafana public dashboard (`/public-dashboards/<token>`), а не анонимный Viewer.** Анонимный Viewer может слать в `/api/ds/query` любой SQL от имени источника данных; public dashboard исполняет только сохранённые запросы своих панелей. File-provisioning public dashboards не поддерживает, поэтому one-shot `grafana-publish` включает его через HTTP API (идемпотентно) и печатает ссылку. На этапе 6 Caddy проксирует наружу только `/public-dashboards/`, `/api/public/` и статику. *Цена:* лишний one-shot контейнер; токен ссылки генерирует Grafana, при пересоздании тома Grafana ссылка меняется.
13. **Роль `grafana_reader`: init-скрипт создаёт её (пароль `GRAFANA_DB_PASSWORD`, `default_transaction_read_only=on`, `statement_timeout=5s`), гранты выдаёт `create_schema`** на объекты `DASHBOARD_OBJECTS` (оба представления, `drift_metrics`, `run_metrics`), если роль существует. Объекты создаёт приложение позже init-скрипта, поэтому гранты живут рядом со схемой. API теперь ждёт `db-init`, чтобы роль существовала к его первому `create_schema`. *Цена:* код схемы знает имя роли мониторинга.
14. **Telegram подключается, только если заданы `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID`.** Grafana 13 не стартует с пустым `bottoken` в provisioning (проверено: `could not find Bot Token in settings`). `infra/grafana/start.sh` копирует provisioning во временный каталог и добавляет `telegram.yml` (контакт-точка + политика) только при обоих значениях. Без них подключается `telegram-off.yml` (сброс политики и удаление контакт-точки, оставшейся от прошлого старта: Grafana хранит provisioned-объекты в своей БД и не удаляет их вместе с файлом). Политика по умолчанию в Grafana 13 ведёт в получателя `empty` без интеграций: алерты срабатывают и видны в UI, ничего не отправляется. *Цена:* обёртка над entrypoint образа Grafana.
15. **Секреты Grafana — та же политика `recsys.secrets_check`** (группа `grafana`: `GRAFANA_ADMIN_PASSWORD`, `GRAFANA_SECRET_KEY`, `GRAFANA_DB_PASSWORD`). Проверяет one-shot `grafana-init` на образе API, Grafana ждёт его успеха. `GF_SECURITY_SECRET_KEY` задан явно: умолчание Grafana — публично известная строка, которой шифруются пароли источников данных в её БД.
16. **Пороги алертов (spec 5.5):** `up{job="api"} < 1` 2 мин (нет данных = тревога); доля 5xx > 5% по окну 5 мин; p95 > 300 мс по окну 10 мин (пользовательские маршруты, без `/metrics`, `/health`, `/ready`, `/admin/*`); доля fallback > 20% по окну 10 мин; возраст модели > 36 ч. Для долей при нулевом трафике (0/0) состояние Normal (нет данных = OK). Гистограмма имеет бакет ровно 0.3 с, чтобы p95 около порога не интерполировался между 0.25 и 0.5.
17. **`/metrics` на основном порту API (spec 3.1), один процесс uvicorn** — обычный реестр prometheus_client без multiprocess-режима. Закрыть `/metrics` снаружи — задача Caddy на этапе 6.
18. **JSON-логи только в API.** Задачи Airflow пишут в журнал задачи, который читают в UI Airflow; там формат Airflow удобнее. *Цена:* логи пайплайна не в JSON (централизованный сбор — v2, Loki).
19. **Grafana 13.2.5 и Prometheus v3.15.1.** Обе версии вышли больше двух недель назад (2026-10-27 и 2026-10-21); 13.3.0 — первый минорный релиз своей ветки, 13.2.x уже на пятом патче.

### Решения и находки при реализации

20. **Год выпуска не того фильма в `drift._movie_table` (баг, пойман тестом).** `pd.DataFrame({...}, index=movie_id)` выравнивал Series годов по старому индексу 0..n, и фильм получал год соседа. Теперь значения передаются `.to_numpy()`; тест сверяет год конкретных фильмов.
21. **FastAPI 0.162 не кладёт `scope["route"]` для смонтированных приложений** (только для APIRoute), поэтому `/static/*` метился как `unmatched`. Middleware получает список префиксов монтирования (`/static`).
22. **`--no-access-log` uvicorn выключает access-лог через `handlers=[]` и `propagate=False`.** Первая версия перенаправления логов uvicorn в JSON включала его обратно. Теперь перенаправляются только логгеры с обработчиками; тест закрепляет, что выключенный access-лог остаётся выключенным. Прогресс-бар MLflow при загрузке модели (tqdm в stderr, не JSON) выключен `MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false`.
23. **Память Grafana.** `docker stats` показывал ~620 МБ, из них ~400 МБ — активный page cache отображённых бинарников (cgroup `file`), RSS ~225–250 МБ; после пересоздания контейнера 251 МБ. Лимиты памяти — в `docker-compose.prod.yml` (этап 6).
24. **`docker compose up -d --wait` возвращает 1**, если one-shot контейнер (`grafana-publish`, `db-init`, `grafana-init`, `airflow-init`) завершился, даже с кодом 0. Стек при этом поднят целиком; в e2e это видно в `01-compose-up.log`. Для скриптов: `up -d` и проверка `/ready`.
25. **Public dashboard не раскрывает запросы.** `/api/public/dashboards/<token>` отдаёт панели без `expr`/`rawSql`, запросы панели выполняются только через `/api/public/dashboards/<token>/panels/<id>/query` и требуют `timeRange` в теле. `/api/ds/query` без логина — 401.
26. **Первый реальный дрифт: `drift_share = 1.0` и `prediction_drift_share = 1.0`.** 1647 веб-оценок за 7 дней — тестовые посетители этапов 3–5, которые оценивают онбординг-фильмы (популярные, высокие оценки, активность 2–20 оценок против 20–2314 у MovieLens). Это ожидаемо по spec 5.3 и не тревога.
