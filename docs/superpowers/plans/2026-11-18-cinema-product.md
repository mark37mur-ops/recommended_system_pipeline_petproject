# «Кино на вечер» Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать одобренный киноинтерфейс с первыми пятью звёздными оценками, колодой намерений и настоящими постерами, сохранив cookie и пользовательские данные.

**Architecture:** Сохранить FastAPI/Jinja2/vanilla JS и существующий fold-in. Намерения хранить отдельно в movie_feedback, обогащение — в movie_metadata; TMDB вызывается только отдельным CLI. Три независимых кодера работают по общим контрактам; интегратор единолично меняет общую схему, сериализацию и HTTP wiring.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, SQLAlchemy Core, PostgreSQL 16, Redis, Jinja2, JavaScript/CSS, uv/pytest/ruff; браузерная проверка desktop/mobile.

**Spec:** `docs/superpowers/specs/2026-11-17-cinema-product-design.md`. Дизайн одобрен пользователем; новый brainstorming не требуется. Уточнение пользователя имеет приоритет: валидный Read Access Token уже предоставлен отдельно.

## Global Constraints

- «Первые минимум пять разных фильмов оцениваются звёздами 1–5»; сохранить `MIN_PERSONAL_RATINGS=5`.
- «Засчитываются только успешно сохранённые оценки разных movie_id»; повторная оценка не повышает прогресс.
- «Хочу посмотреть» не преобразуется в 5, «Не интересно» не преобразуется в 1. Только звёзды попадают в ratings/обучение.
- «Пропуск ... скрывает карточку только в текущей сессии»: только память текущей страницы, без SQL, Redis, localStorage/sessionStorage и нового skip TTL. Reload может вернуть пропущенное.
- «Работать с существующей подписанной cookie»; не принимать user_id клиента. Сохранить SESSION_SECRET, формат/имя/срок cookie, web-id boundary, квоту 300 оценок и существующие лимиты 30 mutations/min, 120 recommendations/min по IP/visitor.
- «Новые поля — отдельная movie_metadata ... не менять существующие таблицы и не переписывать исходные названия». Миграция только CREATE новых таблиц, без ALTER/drop/перезагрузки датасета.
- «Фильтровать исключения и на cache-hit, и на свежей выдаче, и в fallback». Не возвращать отвергнутое ради десяти мест. Импрессии — только для окончательно возвращённых ids.
- «На пользовательских экранах отсутствуют названия алгоритмов, модели, версии, кэш, NDCG, названия датасетов и инфраструктуры». Служебные API/метрики сохранить.
- Каталог не расширять из TMDB. Секрет `D:\secrets\movielens-recsys\tmdp_token.txt` (именно tmdp) не читать при планировании, не печатать, не помещать в git/браузер/артефакты.
- Текущее разрешение — только этот план: не кодировать, не коммитить, не push, не менять сервер. Выпуск ниже — будущий этап с отдельным разрешением.

## Review Focus

1. Старая cookie после обновления и сбоя истории не должна сбрасывать прогресс или незаметно заводить нового пользователя — Tasks 2,4,6.
2. Одновременные rating/intent и возврат Redis не должны оставлять противоречивое намерение или оживлять отвергнутое — Tasks 2,6.
3. Более 50 исключений, пустая колода и недоступный Postgres не должны приводить к повтору запрещённых фильмов — Task 2.
4. Одноимённые ремейки, альтернативные названия и отсутствие года не должны давать чужой постер — Tasks 3,7.
5. Таймаут записи, вертикальный скролл и опоздавший ответ поиска не должны совершать лишнее действие или менять карточку — Tasks 5,6.

## Проверенная отправная точка

Прочитаны service/repository/main/schemas/catalog/settings/cache, DB schema, UI, pyproject, Compose/Dockerfile/CI/deploy.sh и существующие тесты/fixtures. Сейчас recommend отдаёт cache-hit до чтения БД; fallback при ошибке БД возвращает popular без исключений. upsert_rating уже блокирует web-user и защищает квоту. MovieOut не содержит постеров; UI показывает технические подписи. Warmup создаёт схему в фоне: этого недостаточно для миграции до обновлённого API. Integration fixture использует явный TRUNCATE без новых FK-таблиц. API image включает src и services/api, поэтому CLI из src/recsys попадёт в image.

Read-only git status: присутствует незакоммиченная спецификация; не перезаписывать её. Для read-only git использован `git -c safe.directory=D:/movielens-recsys ...`, без изменения global config. Проверки приложения при подготовке плана не запускались.

## Файлы, владельцы и зависимости

Все пути относительно `D:\movielens-recsys`.

| Scope | Владелец | Создать / изменить |
|---|---|---|
| Схема и HTTP интеграция | I | `src/recsys/db/schema.py`, `services/api/recsys_api/{schemas,main}.py`, `tests/unit/{api_fakes,test_db_schema,test_api_http}.py`, `tests/integration/conftest.py` |
| Feedback + выдача | A | `services/api/recsys_api/{service,repository}.py`, новые `tests/unit/test_api_feedback.py`, `tests/integration/test_feedback_db.py`; существующие `tests/unit/{test_api_service,test_api_cache}.py` |
| Постеры и metadata | B | новые `src/recsys/{movie_metadata,enrich_movies}.py`, `services/api/recsys_api/metadata.py`, `tests/unit/{test_movie_metadata,test_enrich_movies}.py`, `tests/integration/test_movie_metadata_db.py` |
| UI | C | `services/api/recsys_api/templates/index.html`, `services/api/recsys_api/static/{app.js,style.css}`, новые `static/tmdb-logo.svg`, `tests/unit/test_cinema_page.py` |
| Сквозные проверки | I | новый `tests/integration/test_cinema_flow.py`; screenshots/logs вне публичных assets |
| Подготовка выпуска | I | `infra/scripts/deploy.sh`, `tests/unit/test_deploy_script.py` |

Граф: **Task 1 (I)** → **Tasks 2 (A), 3 (B), 5 (C) параллельно** → **Task 4 (I)** → **Task 6** → **Task 7** → **Task 8** → **Task 9**. UI до Task 4 использует тестовые JSON по контракту. Общие файлы меняет только I; A/B/C передают ему необходимые изменения. Не менять identity/model/training/settings без отдельной выявленной необходимости. Не добавлять runtime HTTP dependency: CLI использует stdlib urllib.request с внедряемым транспортом.

## Общий контракт: закрепить до параллельной работы

### SQL

- `movie_feedback`: user_id Integer NOT NULL FK users.user_id ON DELETE CASCADE; movie_id Integer NOT NULL FK movies.movie_id; state Text NOT NULL CHECK IN ('watchlist','dismissed'); updated_at DateTime(timezone=True) NOT NULL default func.now(); composite PK `(user_id,movie_id)` named `movie_feedback_pkey`. Ни skip, ни rating.
- `movie_metadata`: movie_id Integer PK FK movies.movie_id; tmdb_id Integer NOT NULL; poster_path Text nullable; title_ru Text nullable; source Text NOT NULL CHECK source='tmdb'; updated_at DateTime(timezone=True) NOT NULL default func.now(). Upsert, исходный movies.title неизменён; unmatched не удаляет прежний match.
- Обе таблицы в общей metadata определяет I. A/B только импортируют их. create_schema/create_all запускается повторяемо; ни одной старой таблицы не пересоздавать. Fixtures добавляют новые таблицы в явный TRUNCATE, иначе FK заблокируют очистку родителей.

### Python

- A сохраняет `RecommendationService.rate(...) -> RatingResult`, `recommend(user_id: int | None, k: int) -> Recommendation`.
- A в service.py: `FeedbackState = Literal['watchlist','dismissed']`; frozen `FeedbackResult(user_id:int, created_user:bool, movie_id:int, state:FeedbackState)`; exception `MovieAlreadyRated`.
- A добавляет методы service: `set_feedback(user_id:int|None, ip:str, movie_id:int, state:FeedbackState)->FeedbackResult`; `remove_watchlist(user_id:int|None, ip:str, movie_id:int)->None`; `feedback(user_id:int|None)->dict[int,FeedbackState]`; `rating_history(user_id:int|None)->dict[int,int]`. Последний строгий: БД failure не превращается в пустую историю; старый user_ratings helper оставить совместимым.
- A расширяет Repository/PgRepository: `user_feedback(user_id:int)->dict[int,FeedbackState]`; `upsert_feedback(user_id:int,movie_id:int,state:FeedbackState)->None`; `delete_watchlist(user_id:int,movie_id:int)->None`. Writes используют тот же users row lock, что upsert_rating. Intent на уже оценённый фильм → MovieAlreadyRated; rating атомарно удаляет intent. delete удаляет только state=watchlist. user_feedback возвращает ordered dict: updated_at DESC, movie_id ASC.
- B в `src/recsys/movie_metadata.py`: frozen `MovieMetadata(movie_id:int,tmdb_id:int,poster_path:str|None,title_ru:str|None,source:str='tmdb')`; frozen `CatalogEntry(movie_id:int,title:str,year:int|None,n_ratings:int)`; `normalize_title(title:str)->str`; `match_movie(title:str,year:int|None,candidates:list[dict])->dict|None`; `poster_url(poster_path:str|None)->str|None`; `load_metadata(engine:Engine)->dict[int,MovieMetadata]`; `upsert_metadata(engine:Engine,item:MovieMetadata)->None`; `load_enrichment_catalog(engine:Engine)->list[CatalogEntry]`.
- B в `services/api/recsys_api/metadata.py`: `MetadataHolder.current:dict[int,MovieMetadata]`; `refresh(engine:Engine)->int`; `get(movie_id:int)->MovieMetadata|None`; `search(query:str,limit:int=20)->list[int]`. Refresh атомарно заменяет snapshot; при ошибке старый сохраняется, изначально пустой допустим.
- I расширяет create_app keyword аргументом `metadata_holder:MetadataHolder|None=None`. Production metadata refresh callback подключить к warmup/admin reload; tests inject holder без реальной БД. PgRepository.load_movies и Movie.title B не меняет.
- I: `MovieOut.of(movie:Movie,user_rating:int|None=None,metadata:MovieMetadata|None=None)->MovieOut`. Новое поле только `poster_url:str|None`; title=непустой title_ru либо прежний display_title, year/genres исходные. Все movie responses используют один serializer.

### HTTP

`MovieOut={movie_id:int,title:str,year:int|null,genres:string[],user_rating:int|null,poster_url:string|null}`; `MovieList={items:MovieOut[]}`.

| Request | Response / поведение |
|---|---|
| `POST /ratings {movie_id,rating}` | прежний 200 `{movie_id,rating,n_ratings}`, удалить intent в транзакции, invalidate cache |
| `GET /ratings` | 200 `{items:MovieOut[],n_ratings:int,min_ratings:5}`, вся история с user_rating; anonymous empty, известный пользователь/DB down →503 |
| `GET /watchlist` | 200 MovieList, только собственные watchlist в repo order; anonymous empty, DB down →503 |
| `POST /feedback {movie_id,state}` | 200 `{movie_id,state}`, state только watchlist/dismissed; первый anonymous mutation создаёт web-user/cookie как ratings |
| `DELETE /watchlist/{movie_id}` | idempotent 204, только своё watchlist; anonymous no-op без создания пользователя |
| `GET /movies/onboarding` | прежний MovieList + metadata; UI заменяет «Не смотрел» локально из пула |
| `GET /movies/search?q=...` | прежние ограничения 2–100; I объединяет Catalog.search и MetadataHolder.search по русскому названию, dedupe по movie_id, прежние результаты первыми |
| `GET /recommendations?k=1..50` | прежний JSON с technical fields для совместимости; UI их не показывает, server skip param отсутствует |

I определяет FeedbackIn/FeedbackOut/RatingsOut в schemas.py. Новые тела extra='forbid', StrictInt>0; user_id/skip запрещены. Errors: 422 invalid/unknown movie; 429 shared mutation limiter + Retry-After:60; 403 только rating quota; 409 intent на rated; 503 DB. Общий cookie setter rating/feedback с прежними параметрами; GET не создаёт пользователей. Существующие same-origin/security ограничения сохранить.

## Task 1: Общие таблицы и сериализация — I

**Files:** schema.py, schemas.py, test_db_schema.py, test_api_http.py, integration/conftest.py из scope I.
**Interfaces:** SQL/MovieOut/Pydantic контракт выше; prerequisite A/B/C.

- [ ] Написать `test_feedback_and_metadata_are_additive`: новые PK/FK/checks совпадают с контрактом, старые columns/constraints неизменны. `test_feedback_input_rejects_user_id_skip_and_non_integer`: reject 1.0/true/'1'/user_id/state='skip'. `test_movieout_metadata_optional`: старый Movie даёт poster_url=None.
- [ ] `uv run pytest tests/unit/test_db_schema.py tests/unit/test_api_http.py -v`: сохранить red по отсутствующим таблицам/полям.
- [ ] Добавить только новые таблицы и согласованные models/serializer; exact-key assertion существующего HTTP теста расширить только на poster_url. Обновить явный TRUNCATE fixture новыми FK-таблицами.
- [ ] Повторить тот же test command, сохранить green. Передать готовые имена A/B/C; дальнейшие изменения общей зоны — только I.

## Task 2: Feedback, транзакции и безопасная выдача — A

**Files:** scope A. **Interfaces:** service/Repository методы выше; зависит от Task 1. Общий api_fakes обновляет I по переданным A сигнатурам; A до этого использует subclass fake в собственном test файле.

- [ ] Failing unit tests: `test_intent_never_changes_ratings_or_model_input` (ratings до/после равны, модель получает только звёзды); `test_feedback_upsert_switches_state` (одна строка); `test_rating_clears_intent`; `test_remove_watchlist_only`; `test_rated_movie_rejects_intent`; `test_history_failure_is_not_empty`; `test_feedback_limits_and_web_id_guard`. Invalid movie не создаёт user.
- [ ] Параметризовать `test_exclusions_on_every_path`: cache-hit/cold-start/full/model-missing/model-error; ids уникальны, rated+watchlist+dismissed отсутствуют; observer получает ровно финальные response ids. `test_more_than_max_k_exclusions`: >50 exclusions и достаточно разрешённых → k; весь каталог запрещён →[].
- [ ] `test_db_down_does_not_serve_unknown_exclusions`: identified user без читаемых ratings/feedback → пустой fallback с n_ratings=None, без кэширования/популярных/старых cached ids; anonymous cold-start прежний. UI отличает недоступность от «На сегодня всё».
- [ ] `uv run pytest tests/unit/test_api_feedback.py -v`: зафиксировать red. Реализовать методы и общий финальный фильтр перед _serve: актуальные ratings/feedback читаются даже на cache-hit; после фильтра добрать разрешённым popular за пределами первых MAX_K. _compute/жанровые веса зависят только от звёзд.
- [ ] После всех успешных mutations invalidate существующий generation cache; сохранить dirty recovery и защиту stale put. Skips в cache не добавлять. Новый запрос после подтверждённой mutation обязан учитывать exclusion; уже начатый concurrent read может отражать предыдущий snapshot.
- [ ] Failing Postgres tests: `test_concurrent_rating_feedback_rating_wins` — intent первым затем удалён либо rating первым и intent→409; `test_failed_rating_rolls_back_feedback_delete`; `test_user_isolation_and_fk`; `test_retry_upsert_idempotent`. В одной транзакции users lock → web-source check → rating/feedback write; quota как прежде.
- [ ] Реализовать, запустить unit tests A и `uv run pytest tests/integration/test_feedback_db.py -v` с disposable test Postgres/Redis, сохранить green. Skipped integration не считать прохождением.

## Task 3: Изолированное обогащение и metadata — B

**Files:** scope B. **Interfaces:** functions/holder выше; CLI `python -m recsys.enrich_movies --token-file PATH --limit N [--priority-ids ID,ID] [--refresh] [--dry-run]`, DATABASE_URL из окружения без печати DSN. Зависит от Task 1; параллелен A/C.

- [ ] Failing tests `test_match_rejects_remake_year`, `test_ambiguous_or_missing_year_unmatched`, `test_article_and_alternative_title_match`, `test_unsafe_poster_path_rejected`, `test_source_title_unchanged`. Normalize case/пробелы/пунктуацию/trailing article/aliases; единственный exact normalized title или original_title/alias + точный release year; иначе None, не первый result.
- [ ] poster_url допускает только `/[A-Za-z0-9_-]+.(jpg|jpeg|png|webp)` и строит `https://image.tmdb.org/t/p/w500{path}`; reject scheme,//,traversal,query,foreign host. Базовый scope без proxy.
- [ ] CLI tests с fake transport: timeout=10s, max 3 attempts; 429/5xx backoff 1/2s и bounded Retry-After<=30s; 401/403 без retry; <=2 requests/sec. limit 1..4000 default100 ограничивает число фильмов; <=3*N total attempts, без pagination. Повтор без refresh не трогает matched rows, dry-run без DB write, ошибки nonzero с безопасным summary. Bearer не попадает в stdout/stderr/exceptions.
- [ ] `uv run pytest tests/unit/test_movie_metadata.py tests/unit/test_enrich_movies.py -v`: сохранить red. Реализовать functions/SQL/holder/CLI stdlib transport. TMDB search/movie query/year/language=ru-RU; детали API сверить с актуальной официальной документацией перед реализацией. Секрет только внутри CLI в bearer header; никакого внешнего поиска по пользовательскому request.
- [ ] Queue: явно переданные priority ids onboarding сначала, затем весь каталог по n_ratings DESC/movie_id ASC. I получает ids из `/movies/onboarding`; B не импортирует API Catalog из src CLI. Нет priority ids → popular order. Отдельный тест pins порядок.
- [ ] Integration: repeated upsert одна row, FK, неизменный movies.title, пустая metadata, refresh error сохраняет snapshot. Запустить unit tests B и `uv run pytest tests/integration/test_movie_metadata_db.py -v`, сохранить green.

## Task 4: HTTP wiring и восстановление истории — I

**Files:** main.py, schemas.py, api_fakes.py, test_api_http.py. **Interfaces:** Tasks 1–3; без переименования контракта C.

- [ ] Failing HTTP tests: existing cookie GET /ratings восстанавливает фильмы вне onboarding; first feedback ставит прежнюю cookie; user_id/skip/forged MovieLens id отвергаются; watchlist isolation/delete repeat204; rated intent409; history failure503 вместо нуля.
- [ ] `test_metadata_in_every_movie_response`: onboarding/search/recommendations/ratings/watchlist содержат trusted poster_url/null и локализованное title; поиск по исходному и русскому имени, dedupe. Запустить `uv run pytest tests/unit/test_api_http.py -v`, сохранить red.
- [ ] Реализовать endpoints, общий serializer/cookie setter/error mapping/shared mutation limiter. Watchlist order не пересортировывать. Strict rating_history используется для restore.
- [ ] Подключить DI holder, production metadata-refresh callback к warmup/admin reload. Порядок schema→catalog→metadata→model; metadata error сохраняет fallback title/poster, не блокирует startup навсегда. Tests без engine получают injected holder; не обращаться к выдуманному engine fake repo.
- [ ] `uv run pytest tests/unit/test_api_http.py tests/unit/test_api_identity.py tests/unit/test_api_service.py tests/unit/test_api_cache.py -v`: green и прежние служебные JSON/readiness/reload/metrics сохранены.

## Task 5: Киноинтерфейс — C

**Files:** scope C. **Interfaces:** HTTP контракт, mock JSON до Task 4; зависит от Task 1, параллелен A/B.

- [ ] Failing page tests: title «Кино на вечер»; «Найдём кино на вечер»; «Оцените хотя бы 5 фильмов, которые смотрели»; три пункта навигации; footer dialog «О сервисе»; exact notice ниже; нет technical copy в rendered HTML/title/meta. `uv run pytest tests/unit/test_cinema_page.py -v` → red.
- [ ] Реализовать тёмный кинотеатр/красный акцент, постеры 2:3, desktop top nav/mobile bottom nav с safe-area; loading/error/empty состояния. Startup сначала GET /ratings: server distinct count; >=5 сразу подборка, <5 restore stars/progress; 503 retry без ложного сброса. CTA «Подобрать кино» только >=5, можно оценить больше.
- [ ] Onboarding «Не смотрел» заменяет локально из пула; при исчерпании поиск. Deck k=50, локальные seen/skipped Sets текущей страницы; skips без HTTP mutation/хранилищ/TTL. После подтверждённых mutations refetch/fill. Если остались только skipped, показать «На сегодня всё», поиск и возврат к оценкам, без бесконечных refetch.
- [ ] Intent POST /feedback; «Уже смотрел» раскрывает звёзды POST /ratings. Pending блокирует все действия текущей карточки; advance только после success. Network/timeout/429/409/503 сохраняет карточку и retry. Rating немедленно refetch рекомендации, игнорировать устаревшие ответы; sync watchlist/history после success. n_ratings=None → временная недоступность, не exhausted.
- [ ] Свайп: pointer capture; commit только abs(dx)>=80 CSS px и abs(dx)>=1.5*abs(dy); vertical/cancel без mutation; touch-action:pan-y. Left/Right=отказ/сохранение, Space=skip только при фокусе колоды, не search/dialog/stars. Кнопки эквивалентны; focus visible, keyboard stars, reduced-motion без движения.
- [ ] Image error один раз заменяет локальным DOM placeholder title/year без сломанной иконки/рекурсии. Официальный SVG логотип получить из официальных TMDB brand assets. Footer dialog содержит источники и “This product uses the TMDB API but is not endorsed or certified by TMDB.” Отдельное техноменю не делать.
- [ ] Search debounce250ms, sequence/abort обновлять даже при очистке; старый response не возвращает cleared results. Названия/ошибки через textContent, не innerHTML. Page tests → green; поведение подтвердить браузером в Task 6, не grep JS реализации.

## Task 6: Сквозные данные и desktop/mobile — I

**Files:** новый tests/integration/test_cinema_flow.py; дефекты исправляют владельцы.
**Interfaces:** реальные SQL/endpoints/Redis, прежний CookieSigner/SESSION_SECRET.

- [ ] Failing `test_existing_cookie_survives_additive_upgrade`: disposable DB со старым subset таблиц, populated users/movies/ratings и подписанной cookie → create_schema дважды → rows/ids/ratings прежние; app restart с тем же secret восстанавливает историю. Не запускать destructive fixtures на production; имя test DB обязательно содержит test.
- [ ] `test_five_unique_ratings_and_intent_flow`: 4 distinct+повтор=4, пятая=5; intent не меняет rating count/model input; reload history/watchlist; remove unrated watchlist возвращает в pool; rating удаляет intent; второй visitor изолирован; cache hit и Redis outage/recovery исключения сохраняют. Выполнить `uv run pytest tests/integration/test_cinema_flow.py -v` на test Postgres/Redis15, сохранить red/green.
- [ ] Локальный запуск `uv run uvicorn --factory recsys_api.main:create_app --host 127.0.0.1 --port 8000` с локальной test configuration и стабильным SESSION_SECRET, без вывода секретов. Browser errors через interception/offline, не отключение production DB.
- [ ] Desktop1440×900 и mobile390×844: новый visitor, 0→4→5, повтор, >5, поиск, «Не смотрел», CTA; reload восстанавливает stars/progress; существующая cookie с >=5 сразу в колоду. Скриншоты onboarding/deck/watchlist/history, нет horizontal overflow и перекрытия bottom nav.
- [ ] На обоих размерах: три действия, pending double-click=одна mutation, network reject/retry, «Уже смотрел», watchlist remove, history edit, exhausted queue, failed history, no model/DB unavailable; keyboard/touch/vertical scroll без feedback/reduced-motion. Reload очищает только skip, intent/ratings остаются.
- [ ] Poster load/error, null year, длинное/HTML-подобное title, stale search после очистки, hidden panels не перехватывают focus. Проверить network assertions; screenshot один не доказывает поведение. При отсутствии browser automation выполнить ручные сценарии и записать результаты без заявления автоматизации.
- [ ] Проверить visible text/aria/title/meta: нет ALS, LightGBM, NDCG, model version, cache, MovieLens и русских технических аналогов; технический raw JSON допустим. Сохранить реальные результаты.

## Task 7: Отдельная приёмка настоящих постеров — I+B

**Interfaces:** CLI Task 3, реальный TMDB; этот этап не выполняется в planning запросе. Заглушки не закрывают acceptance.

- [ ] При разрешённом исполнении CLI сам безопасно читает token file; не shell substitution/не Get-Content секрета. Локальная команда: `uv run python -m recsys.enrich_movies --token-file 'D:\secrets\movielens-recsys\tmdp_token.txt' --limit 100 --priority-ids <onboarding_ids>`. Повторный запуск без дублей/лишних HTTP, затем ограниченные пакеты по каталогу.
- [ ] Вручную сверить title/year/tmdb_id и изображение всех 20 onboarding карточек и минимум 10 разных персональных карточек после пяти тестовых звёздных оценок. Все sampled posters настоящие и соответствующие; missing match не подменять чужой картинкой. Неудача sample оставляет gate открытым.
- [ ] Браузер: у этих 30+ изображений naturalWidth>0, HTTPS image.tmdb.org, нет image/network ошибок; таблица movie_id/title/year/match и desktop/mobile screenshots без секретов. Отдельно воспроизвести корректный fallback одного изображения.
- [ ] Перед публикацией отдельно проверить картинки из VPS и клиентского браузера. В текущем запросе VPS действия не разрешены: future gate. Если прямой host недоступен, остановить выпуск и отдельно спланировать proxy по movie_id с host allowlist/timeout/size limit; не открывать произвольный URL endpoint.

## Task 8: Logic + quality reviews и финальная верификация — I

**Interfaces:** весь diff и evidence Tasks 1–7; исправления у владельцев.

- [ ] Параллельно запустить независимых logic/quality агентов. Logic: intent≠rating, locking/quota/isolation, exclusions/cache/fallback/exhaustion, impressions, additive migration/old cookies. Quality: contracts/ownership, accessibility/swipes/pending/errors, matching/secrets, responsive/technical copy. Если agent tools недоступны, честно отметить ограничение, self-review не называть независимым.
- [ ] Review findings проверить по источникам/воспроизведению; исправления через failing regression→green; изменения общего контракта синхронизирует I.
- [ ] `uv run ruff check .`; `uv run ruff format --check .`; `uv run pytest -v`; `uv run pytest -m integration -v` с настоящими disposable Postgres/Redis. Сохранить exit codes и pass/fail/skip counts. Skipped integration не считать green. Для новых API библиотек использовать context7 при доступности либо официальную документацию.
- [ ] Выполнить существующие CI build/import/config/shellcheck/actionlint checks из .github/workflows/ci.yml. В частности `docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.prod.yml --profile tools config -q`; API image импортирует service/CLI, static assets присутствуют. Не запускать publish без разрешённого push.
- [ ] Plan completion gate: каждой Task сопоставить log/test/screenshot либо pending с причиной. Poster live/VPS/deploy не выдавать за выполненные по unit tests.

## Task 9: Подготовка деплоя и будущий выпуск — I

**Files:** infra/scripts/deploy.sh, tests/unit/test_deploy_script.py. Подготовка тестируется до server mutation; фактический выпуск только после отдельного разрешения. Этот план не разрешает commit/push.

- [ ] Failing harness tests `test_schema_runs_before_new_api`, `test_schema_failure_restores_previous_ref_tag`, `test_smoke_failure_rolls_back_without_dropping_tables`. Сейчас deploy делает pull→up→smoke; новый additive preflight должен быть между pull и up.
- [ ] После compose pull вызвать `compose run --rm --no-deps api python -c 'import os; from recsys.db.schema import create_schema, make_engine; e=make_engine(os.environ["DATABASE_URL"]); create_schema(e); e.dispose()'`, затем прежний up/smoke. Новый image tag, existing env; DB заранее доступна. Не loader/bootstrap_data, не вывод DSN/secret. При failure прежний rollback ref/tag, созданные таблицы оставить.
- [ ] `uv run pytest tests/unit/test_deploy_script.py -v` и shellcheck → green. На disposable Compose populated DB проверить preflight до старта нового API, повторяемость create_all и неизменность старых rows/sequence. Старый image должен работать с добавленными таблицами.
- [ ] До разрешения server mutation подготовить release ref/image sha tag, previous known-good sha, CI/reviews/browser evidence; backup существующим infra/scripts/backup.sh и проверенный restore в отдельной DB. SESSION_SECRET/volumes/cookie domain сохранить; latest не rollback target. Backup/restore на сервере тоже требуют разрешения выпуска.
- [ ] После отдельного разрешения использовать существующий workflow/deploy.sh без обхода CI/hooks. Выполнить additive preflight, CLI enrichment отдельным контейнером с read-only token file, безопасно доставленным вне git. Windows path в VPS контейнере не существует; не помещать token в image/env dumps.
- [ ] Public health/ready/recommendations, старая cookie/history/feedback, VPS+client posters, desktop/mobile. Неуспех data/poster/smoke gate → прежний sha/tag/ref rollback, повторная проверка cookie/history. Без drop новых таблиц/down -v/удаления новых данных. Старый UI после rollback не поддерживает intent: явно сообщить это ограничение, feedback сохранить до исправленного релиза.

## Самопроверка плана

- Spec coverage: onboarding/return/history — Tasks4–6; deck/skip/signals —2,5,6; navigation —5; metadata/posters/attribution —3,5,7; limits/data/impressions —1,2,4,6; reviews/CI/rollback —8,9.
- Имена таблиц/методов/routes едины; общей зоной владеет I, поэтому A/B/C не пересекаются в schema/main/serializer/fakes.
- Пять Review Focus рисков получили конкретные проверки. UI поведение подтверждает браузер, реальные постеры — отдельная живая приёмка.
- В текущем задании создаётся только этот Markdown; шаги реализации и выпуска не выполняются.
