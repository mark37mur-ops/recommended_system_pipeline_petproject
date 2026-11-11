# Этапы 6–7: прод-окружение (Caddy, HTTPS, лимиты, бэкапы) и CD с откатом. План реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** всё, что нужно для публичного деплоя на один VPS (spec 6.1–6.4), собрано, проверено локально и готово к включению. Сервера пока нет (покупка VPS — решение пользователя), поэтому:
1. `docker-compose.prod.yml` поверх `docker-compose.yml` поднимает тот же стек «как на сервере»: образы из GHCR вместо сборки, Caddy — единственный вход на 80/443, наружу только `/` (приложение без `/metrics` и `/admin/*`) и публичный дашборд Grafana под `/grafana/`, лимиты памяти, ротация логов, реальные IP клиентов за прокси.
2. На каждый push в `main` после зелёных тестов CI собирает образы `api`, `airflow`, `mlflow` с тегами `sha-<7>` и `latest` и пушит их в GHCR (это заработает сразу после push в `main`).
3. `deploy.yml` после успешного CI на `main` выкатывает тег на сервер по SSH (`infra/scripts/deploy.sh`: pull → up → ожидание `/ready` → откат на прошлый тег при неудаче). Пока секретов `DEPLOY_*` нет, workflow пишет «deploy skipped: server not configured» и завершается успешно: `main` остаётся зелёным без сервера.
4. `bootstrap-server.sh` (идемпотентная подготовка чистого Ubuntu: пользователь, ufw, SSH только по ключу, Docker, клон репозитория, `.env` со сгенерированными секретами, cron бэкапа) и `backup.sh` (pg_dump `app` + `mlflow` и артефакты MLflow в датированный архив, хранить N последних).

**Architecture:**
- **Compose override.** `docker-compose.prod.yml` меняет только то, что отличает сервер: `image: ${IMAGE_PREFIX}-{api,airflow,mlflow}:${IMAGE_TAG}` + `build: !reset null` (сервер ничего не собирает); `ports: !reset []` у Postgres, Redis и API (наружу их видно только через Caddy или никак), админские UI (MLflow, Airflow, Grafana, Prometheus) остаются на `127.0.0.1` — это и есть точка входа SSH-туннеля из spec 6.2; `deploy.resources.limits.memory` по замерам этапа 5; `logging: json-file` с `max-size`/`max-file`; сеть `edge` с фиксированной подсетью, в ней только Caddy, API и Grafana — `FORWARDED_ALLOW_IPS` API равен этой подсети; Grafana `GF_SERVER_ROOT_URL=https://${SITE_ADDRESS}/grafana/` + `GF_SERVER_SERVE_FROM_SUB_PATH=true`; `COOKIE_SECURE=true`.
- **Caddy** (`infra/caddy/Caddyfile`, образ `caddy:2.11.9-alpine`): адрес сайта `{$SITE_ADDRESS}` (`<ip>.sslip.io` → сертификат Let's Encrypt; `localhost` → локальный CA Caddy для проверки на своей машине). `/metrics*` и `/admin*` → 404, `/grafana/` → только `public-dashboards/*`, `api/public/*`, статика `public/*`, остальное под `/grafana` → 404; всё прочее → `api:8000`. Заголовки безопасности, `encode zstd gzip`. Сертификаты на томе `caddy-data` (не перевыпускаются при каждом пересоздании контейнера и не упираются в лимиты Let's Encrypt).
- **CI → GHCR.** Новый job `publish` в `ci.yml` (`needs:` все тестовые job'ы, только `push` в `main`), matrix по трём образам, `docker/build-push-action` с кэшем `type=gha`, `GITHUB_TOKEN` с `packages: write` только в этом job'е. Имя образа — `ghcr.io/${GITHUB_REPOSITORY,,}-<name>` (GHCR требует нижний регистр).
- **CD.** `deploy.yml`: `workflow_run` от CI (`main`, `push`, `success`) + `workflow_dispatch` с входом `tag` (ручной откат). Шаг-проверка секретов → summary «skipped» либо SSH с закреплённым `known_hosts` (секрет `DEPLOY_KNOWN_HOSTS`), `docker login ghcr.io` на сервере короткоживущим `GITHUB_TOKEN` (через stdin, после деплоя `logout`), `deploy.sh <tag> <git sha>`.
- **deploy.sh** (на сервере): синхронизирует checkout репозитория с выкатываемым коммитом (compose-файлы и конфиги `infra/` должны совпадать с кодом образа), пишет `IMAGE_TAG` в `.env`, `pull` + `up -d --remove-orphans`, ждёт `GET /ready` через Caddy с таймаутом, при неудаче возвращает прошлый тег и прошлый коммит, `up -d` и выход с ошибкой. После успеха удаляет образы проекта старше предыдущего тега (диск).
- **Валидация в CI** — новый job `infra`: `docker compose config -q` для dev и prod с `.env.example`, `caddy validate`, `shellcheck`. Тесты `deploy.sh`/`backup.sh`/генерации `.env` — обычные pytest в `tests/unit` с поддельными `docker`/`curl`/`git` на `PATH` (bash есть в CI на ubuntu и в git-bash на Windows).

**Tech Stack:** Docker Compose (`!reset`), Caddy 2.11.9, GitHub Actions (`docker/build-push-action@v7`, `docker/login-action@v4`, `docker/setup-buildx-action@v4`), GHCR, bash, shellcheck v0.11.0, OpenSSH, ufw.

**Spec:** `docs/superpowers/specs/2026-11-11-movielens-recsys-design.md`, секция 6 (6.1–6.4), заметки для прода из этапа 5 (`.superpowers/sdd/overnight/stage-5-report.md`, п. 9; `s5-fix-report.md`, п. 3).

## Global Constraints

- Всё из планов этапов 0–5 в силе: код, комментарии, коммиты на английском; каждый коммит заканчивается пустой строкой и `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Не пушим, `--no-verify` не используем, README не пишем (это этап 8).
- Dev-стек (`docker compose up -d --build`) не меняется ни на байт поведения: всё прод-специфичное — только в `docker-compose.prod.yml`. Существующие job'ы CI не меняют поведения.
- Файлы, которые исполняются в Linux (`*.sh`, compose, `.env.example`), — LF (`.gitattributes`), проверка `git ls-files --eol`.
- Секреты GitHub и настройки репозитория не трогаем. Без секретов деплой — no-op с зелёным статусом.
- Комментарии в Caddyfile, compose, workflow и скриптах объясняют «почему» (пользователь учит инфраструктуру).
- Наружу (0.0.0.0) публикуется только Caddy. Docker пишет свои правила iptables в обход ufw, поэтому опубликованный на 0.0.0.0 порт открыт в интернет, даже если ufw его «закрыл». Всё прочее — `127.0.0.1` или не публикуется вовсе.

## Review Focus (топ-5 способов ошибиться)

1. **Утечка внутреннего наружу через Caddy.** `/metrics`, `/admin/reload-model`, логин и API Grafana (`/grafana/login`, `/grafana/api/datasources`, `/grafana/api/ds/query`) не должны отвечать снаружи. Проверка curl'ом, в том числе с `..`, `%2e%2e`, двойным слэшем и другим регистром.
2. **Слияние compose-файлов.** `ports` в override *добавляются* к базовым, а не заменяют их: без `!reset` прод опубликовал бы порт API и Postgres. Проверка — `docker compose -f docker-compose.yml -f docker-compose.prod.yml config` (нет `published` кроме Caddy и 127.0.0.1-админок, нет `build`).
3. **Откат действительно откатывает.** При неудачном `/ready` (или неудачном `pull`) `.env` и checkout возвращаются к прошлому тегу/коммиту, `up -d` выполняется снова, скрипт выходит не с нулём. Первый деплой (прошлого тега нет) — честная ошибка без «отката в никуда». Тест с поддельными `docker`/`curl`/`git`.
4. **CD не красит `main` без сервера** и не печатает секреты: отсутствие `DEPLOY_*` → успех с summary; `known_hosts` закреплён (никакого `StrictHostKeyChecking=no`); вход `tag` из `workflow_dispatch` проверяется регуляркой до попадания в shell.
5. **Реальные IP клиентов.** Caddy перезаписывает `X-Forwarded-For` адресом клиента (входящий XFF от клиента не доверяется), uvicorn доверяет XFF только из подсети `edge`. Иначе все посетители делят один бакет rate limit (s5-fix-report, п. 3) или любой может подделать IP.

---

## Файловая структура (что добавится)

```
docker-compose.prod.yml                    # override для сервера
infra/caddy/Caddyfile                      # единственная точка входа
infra/scripts/deploy.sh                    # деплой с откатом (на сервере)
infra/scripts/bootstrap-server.sh          # подготовка чистого Ubuntu (идемпотентно)
infra/scripts/backup.sh                    # pg_dump + артефакты MLflow, хранить N
.github/workflows/ci.yml                   # + job infra, + job publish (GHCR)
.github/workflows/deploy.yml               # CD: workflow_run CI / workflow_dispatch(tag)
.env.example                               # + SITE_ADDRESS, IMAGE_PREFIX, IMAGE_TAG, EDGE_SUBNET
.gitignore                                 # + backups/
tests/unit/shell_fakes.py                  # поддельные docker/curl/git для тестов скриптов
tests/unit/test_deploy_script.py, test_backup_script.py, test_bootstrap_env.py, test_prod_config.py
```

### Task 1: План (этот файл) — коммит

### Task 2: `docker-compose.prod.yml` + `infra/caddy/Caddyfile`
- [ ] Override: образы GHCR + `build: !reset null` для всех девяти сервисов на собственных образах; `ports: !reset []` у postgres/redis/api; Caddy (80, 443/tcp, 443/udp, тома `caddy-data`/`caddy-config`); сеть `edge` (`EDGE_SUBNET`), API/Grafana/Caddy в ней (и в `default`); лимиты памяти; `logging` для всех; Grafana sub-path, healthcheck по `/grafana/api/health`, `grafana-publish` на `http://grafana:3000/grafana`; `COOKIE_SECURE=true`; `FORWARDED_ALLOW_IPS=${EDGE_SUBNET}`.
- [ ] Caddyfile: блокировки, белый список путей Grafana, заголовки, сжатие.
- [ ] `caddy validate`, `docker compose ... config` (проверить: нет `build`, публикуемые порты только у Caddy и 127.0.0.1-админок).
- [ ] Тест `test_prod_config.py` (без Docker): Caddyfile блокирует `/metrics` и `/admin`, у override нет лишних `ports`, все сервисы с ротацией логов и лимитом памяти.

### Task 3: `deploy.sh`, `backup.sh`, `bootstrap-server.sh` (TDD)
- [ ] `tests/unit/shell_fakes.py`: поиск bash (на Windows — git-bash, не `System32\bash.exe` от WSL), поддельные `docker`/`curl`/`git`, журнал вызовов.
- [ ] Тесты deploy: успех (тег записан, pull/up вызваны, checkout нового коммита); `/ready` не поднялся → откат тега и коммита, второй `up -d`, код ≠ 0; `pull` упал → откат; первый деплой без прошлого тега → ошибка без отката; кривой тег → отказ до каких-либо действий; чистка старых образов.
- [ ] Тесты backup: архив содержит `app.dump`, `mlflow.dump`, `mlartifacts.tar.gz`; хранится ровно N последних; ошибка pg_dump → нет недописанного архива.
- [ ] Тест bootstrap: функция генерации `.env` из `.env.example` даёт секреты, которые проходят `recsys.secrets_check` (включая Fernet-ключ), и заполняет `SITE_ADDRESS`/`COOKIE_SECURE`; повторный запуск не перезаписывает существующий `.env`.
- [ ] shellcheck всех трёх.

### Task 4: CI: job `infra` и job `publish`
- [ ] `infra`: compose config (dev и prod), `caddy validate`, shellcheck.
- [ ] `publish`: `needs: [test, integration, docker, airflow, infra]`, `if: push && main`, matrix `api`/`airflow`/`mlflow`, теги `sha-<7>` и `latest`, OCI-метка `org.opencontainers.image.source` (связывает пакет с репозиторием), кэш gha по образу.

### Task 5: `deploy.yml`
- [ ] Триггеры, проверка секретов, summary, SSH, логин в GHCR на сервере, `deploy.sh`, `logout` в `if: always()`, `concurrency`.

### Task 6: Гейты и локальный e2e
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run pytest -q -m "not integration"`, `docker compose config -q` (dev и prod с `.env.example`), `caddy validate`, shellcheck.
- [ ] Локальные образы перетегированы в имена GHCR (`IMAGE_TAG=local-e2e`), `SITE_ADDRESS=localhost`, `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`. curl: `/` 200, `/ready` 200, `/metrics` и `/admin/reload-model` заблокированы, `/grafana/public-dashboards/<token>` 200 (и панели отвечают), `/grafana/login`, `/grafana/api/datasources` заблокированы; обходы путей; реальный IP клиента; `docker stats` против лимитов; `backup.sh` по живому стеку. Логи — `.superpowers/sdd/overnight/s7-e2e/`. В конце `down` (тома остаются), dev-стек проверен `config`.

### Task 7: Отчёт `.superpowers/sdd/overnight/stage-7-report.md`

---

## Decisions

1. **Override, а не отдельный прод-compose.** `docker-compose.prod.yml` содержит только отличия; сервисы, переменные и healthcheck'и живут в одном месте. Замены списков — через теги Compose `!reset` / `!override` (Compose ≥ 2.24): обычное слияние *дописывает* `ports` из override к базовым, и без `!reset` порт API остался бы опубликованным. *Цена ошибки:* override зависит от версии Compose на сервере; `bootstrap-server.sh` ставит актуальный Docker с плагином compose из официального репозитория, CI проверяет `config`.
2. **Админские UI на `127.0.0.1` остаются и в проде** (MLflow 5000, Airflow 8080, Grafana 3000, Prometheus 9090): spec 6.2 требует доступ к ним через SSH-туннель (`ssh -L 8080:localhost:8080`), а туннелю нужна точка на loopback сервера. Postgres, Redis и API в проде не публикуются совсем: API — только через Caddy, базы — через `docker compose exec`. *Цена:* четыре loopback-порта; снаружи они недоступны (127.0.0.1 не маршрутизируется), но любой процесс на сервере их видит — сервер однопользовательский.
3. **Caddy `2.11.9-alpine`, хотя релизу 6 дней** (правило прошлых этапов — «старше двух недель»). 2.11.8 — пакет security-фиксов, 2.11.9 чинит регрессию 2.11.8 в FastCGI/PHP, который мы не используем. Безопасность прокси, смотрящего в интернет, весит больше «выдержки». *Цена:* если в 2.11.9 найдётся регрессия в reverse_proxy, откат — смена одного тега на 2.11.7.
4. **Grafana под sub-path без срезания префикса** (`GF_SERVER_SERVE_FROM_SUB_PATH=true`, Caddy проксирует `/grafana/...` как есть). Иначе через SSH-туннель (`localhost:3000`) админка грузила бы статику по `/grafana/public/...` и ломалась. Следствие: внутри сети Grafana отвечает под `/grafana/`, поэтому в override переопределены healthcheck и `GRAFANA_URL` у `grafana-publish`. Админка через туннель: `http://localhost:3000/grafana/`.
5. **Белый список путей Grafana:** `/grafana/public-dashboards/*`, `/grafana/api/public/*`, `/grafana/public/*` (статика). Всё остальное под `/grafana` → 404. Логин, `/api/ds/query`, `/api/datasources` снаружи не видны: даже при утечке пароля админа войти можно только через туннель. *Цена:* если новая версия Grafana начнёт грузить для публичного дашборда что-то вне этих префиксов, страница сломается — это ловит e2e (браузер + список запросов).
6. **Блок-лист для API, а не белый список** (`/metrics*`, `/admin*` → 404). Маршрутов у API мало и они публичны по смыслу (страница, `/movies/*`, `/ratings`, `/recommendations`, `/health`, `/ready`, `/docs`); белый список пришлось бы править при каждом новом эндпоинте. 404, а не 403: снаружи не видно, что путь существует. Airflow вызывает `/admin/reload-model` внутри сети compose (`http://api:8000`), Prometheus — `/metrics`, Caddy им не нужен. *Цена:* новый внутренний эндпоинт нужно не забыть добавить в блок-лист; тест `test_prod_config.py` фиксирует два текущих.
7. **Реальный IP клиента: сеть `edge` с фиксированной подсетью (`EDGE_SUBNET`, по умолчанию `10.203.17.0/24`), `FORWARDED_ALLOW_IPS` = эта подсеть.** В `edge` только Caddy, API и Grafana. Caddy по умолчанию не доверяет входящему `X-Forwarded-For` и ставит адрес клиента сам, uvicorn доверяет XFF только от адресов `edge`. Фиксированный IP контейнера Caddy хрупок (адрес может занять контейнер, стартовавший раньше), подсеть — нет. *Цена:* Grafana тоже в `edge` и теоретически могла бы подделать XFF для API; это наш контейнер. Подсеть может пересечься с сетью провайдера VPS — тогда меняется одна переменная.
8. **В GHCR пушится и `mlflow`**, хотя spec 6.4 называет только `api` и `airflow`: сервер не должен ничего собирать (8 ГБ RAM, сборка мешала бы работающему стеку), а все три образа несут один тег, и откат возвращает их согласованно. *Цена:* лишняя сборка в CI; слои почти всегда из кэша gha.
9. **Публикация — job в `ci.yml` с `needs:`, а не отдельный workflow через `workflow_run`.** Одна цепочка «тесты → образы» в одном графе, нельзя случайно опубликовать образ с красными тестами. Деплой — отдельный `deploy.yml` по `workflow_run` от CI: его можно перезапускать и вызывать руками (откат), не пересобирая образы. *Цена:* деплой стартует после завершения всего CI, включая `publish`, — это и нужно.
10. **Тег `sha-<7 символов>` + `latest`.** Короткий sha — то, что видно в `git log --oneline`; `latest` удобен для ручного `docker pull`, но деплой всегда идёт по `sha-*` (воспроизводимо, откат по тегу).
11. **Логин в GHCR на сервере — короткоживущим `GITHUB_TOKEN` деплой-job'а** (`packages: read`, через stdin ssh, `docker logout` в конце), а не вечным PAT на сервере. Пакеты GHCR, созданные из Actions, по умолчанию приватные; так деплой работает и с приватными пакетами. *Цена:* ручной `docker compose pull` на сервере без логина не сработает, пока пакеты приватные (можно сделать их публичными в настройках пакета — это решение пользователя).
12. **`deploy.sh` синхронизирует git-checkout сервера с выкатываемым коммитом** (`git fetch` + `git checkout --detach <sha>`), а не только тег образа. Compose-файлы и конфиги `infra/` (Caddyfile, provisioning Grafana, prometheus.yml) монтируются с диска сервера и должны совпадать с кодом образа; откат возвращает и их. Тело скрипта обёрнуто в функцию `main`: bash читает скрипт по мере выполнения, и `git checkout`, переписавший сам `deploy.sh`, иначе мог бы подсунуть ему середину новой версии. *Цена:* локальные правки в checkout сервера ломают деплой (`git checkout` откажется) — это правильно: сервер не место для правок.
13. **Smoke — `GET https://$SITE_ADDRESS/ready` через Caddy** (весь путь: DNS, сертификат, маршрутизация), таймаут 300 с (`READY_TIMEOUT`): API после рестарта загружает модель из MLflow, `/ready` отдаёт 503 до конца загрузки. Не `docker compose up --wait`: он возвращает 1 из-за завершившихся one-shot контейнеров (stage-5 plan, Decision 24). *Цена:* проблема с сертификатом (а не с кодом) тоже вызовет откат; логи деплоя покажут причину.
14. **После успешного деплоя удаляются образы проекта, кроме нового, предыдущего и `latest`.** Каждая версия с новыми зависимостями — до ~1 ГБ новых слоёв; на диске 80 ГБ без чистки место кончится за месяцы. Предыдущий тег оставлен: на нём откат не требует сети.
15. **Секреты в `.env` генерирует `bootstrap-server.sh` через `openssl rand`**: URL-safe base64 без `=` (пароли входят в URL БД) и отдельно Fernet-ключ (URL-safe base64 32 байт **с** `=`). Тест прогоняет результат через `recsys.secrets_check`. Существующий `.env` не перезаписывается.
16. **SSH: `PasswordAuthentication no` в `/etc/ssh/sshd_config.d/00-hardening.conf`** (первым по алфавиту: в sshd побеждает первое значение, а cloud-init кладёт `50-cloud-init.conf` с `yes`). Отключается только если у root или deploy-пользователя уже есть `authorized_keys`, и после `sshd -t` — иначе скрипт запер бы владельца снаружи.
17. **Swap 2 ГБ на сервере.** Пик стека ~3,6 ГБ + обучение; swap — страховка от OOM-killer'а при всплеске, а не рабочая память.
18. **Бэкап: `pg_dump -Fc` баз `app` и `mlflow` + tar тома `mlartifacts` в один датированный архив, хранить 7 последних** (spec 6.2 требовал только `app`; без `mlflow` и артефактов после потери диска нечем откатить champion). Дампы через `docker compose exec` (без опубликованного порта Postgres), архив пишется во временный файл и переименовывается — недописанный архив не выглядит как бэкап. Права 600. Ночной запуск — cron от deploy-пользователя. Хранение на том же сервере — компромисс spec 6.2 (вынос наружу — v2).

### Раздел «Deploy» (черновик для README, этап 8)

> **Как это выкатывается.** Push в `main` → CI (ruff, pytest, integration, сборки) → job `publish` пушит `ghcr.io/<owner>/<repo>-{api,airflow,mlflow}:sha-<7>` → `deploy.yml` заходит на сервер по SSH и запускает `infra/scripts/deploy.sh sha-<7> <sha>`: checkout коммита, новый `IMAGE_TAG` в `.env`, `docker compose pull && up -d`, ожидание `GET /ready` через Caddy. Не дождались — прошлый тег и коммит возвращаются, стек поднимается на них, workflow красный.
>
> **Ручной откат:** Actions → Deploy → Run workflow → `tag = sha-<7>` прошлой версии.
>
> **Модели выкатываются отдельно** (spec 6.4): новый образ не меняет champion в MLflow, новый champion не требует деплоя.
>
> **Первый запуск сервера:** VPS (Ubuntu 24.04, 4 vCPU / 8 ГБ) → `sudo bash bootstrap-server.sh` (пользователь `deploy`, ufw 22/80/443, SSH только по ключу, Docker, клон в `/opt/movielens-recsys`, `.env` с секретами, cron бэкапа) → секреты GitHub `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`, `DEPLOY_KNOWN_HOSTS` → push в `main` (или Run workflow). Публичный адрес — `https://<ip>.sslip.io/`, дашборд — ссылка из `docker compose logs grafana-publish`.
>
> **Админки** — только через туннель: `ssh -L 8080:localhost:8080 -L 5000:localhost:5000 -L 3000:localhost:3000 deploy@<ip>` → Airflow `localhost:8080`, MLflow `localhost:5000`, Grafana `localhost:3000/grafana/`.
