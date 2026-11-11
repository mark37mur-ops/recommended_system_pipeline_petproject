# Этапы 0–1: окружение и базовая линия. План реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Рабочее окружение (Docker, uv, GitHub) и первый измеримый результат: MovieLens 1M загружен и провалидирован, разбит по времени без утечки, popularity-бейзлайн посчитан по NDCG@10 / Recall@10 / Coverage. CI проверяет каждый push.

**Architecture:** Python-пакет `recsys` в `src/`: `data/` (скачивание, парсинг, валидация, split), `metrics.py`, `models/popularity.py` и точка входа `pipelines/baseline.py`, которая склеивает всё в один прогон. Чистые функции на pandas без баз данных: Postgres и MLflow появятся на этапах 2–3.

**Tech Stack:** Python 3.12, uv, pandas, numpy, pyarrow, pytest, ruff, GitHub Actions; Docker Desktop (WSL2) ставится, но в коде этого этапа не используется.

**Spec:** `docs/superpowers/specs/2026-11-11-movielens-recsys-design.md` (секции 2.1, 2.5, 6.3–6.6)

## Global Constraints

- Python `>=3.12,<3.13`, layout `src/recsys/`, зависимости только через `uv add` (точные версии фиксирует `uv.lock`, руками версии не пишем).
- Код, идентификаторы и коммиты на английском; объяснения пользователю на русском.
- `ruff` — линтер и форматтер, `line-length = 100`, target `py312`.
- Данные лежат в `data/` (в `.gitignore`), в git никогда не коммитятся.
- Всё на диске `D:` (`D:\movielens-recsys`, данные Docker в `D:\DockerData`). На `C:` свободно ~12 ГБ.
- Split: per-user temporal, `n_ranker=5`, `n_test=5`, `min_ratings=15` (spec 2.1).
- Метрики: K=10, главная NDCG@10; relevant = все фильмы пользователя в тесте (spec 2.5).
- Детерминизм: tie-break в split = детерминированный хэш от (`user_id`, `movie_id`) (не `movie_id`: он коррелирует со временем добавления фильма); tie-break в popularity = `movie_id` по возрастанию (без изменений).
- Шаги с пометкой **[ты]** выполняет пользователь: установка программ, аккаунты, публикация на GitHub. `git push` делается только по явному «да».

## Review Focus

1. **Не-ASCII в названиях фильмов.** Файлы ML-1M в кодировке latin-1, есть названия вроде `Misérables, Les (1995)`. Загрузка не должна падать или ломать символы. Тест в Task 3.
2. **Скобки внутри названия.** `City of Lost Children, The (Cité des enfants perdus, La) (1995)`: год берётся только из последней `(YYYY)` в конце, название без неё. Тест в Task 3.
3. **Одинаковые timestamp у одного пользователя.** В ML-1M это очень частая ситуация. Split обязан быть детерминированным (tie-break по хэшу от `user_id`, `movie_id`) и не менять результат между запусками. Тест в Task 5.
4. **Короткий или дублированный список рекомендаций.** Меньше k фильмов или повтор одного id. NDCG ≤ 1, повтор не засчитывается дважды. Тест в Task 6.
5. **Повторный запуск скачивания.** Если архив уже есть и MD5 совпадает, повторно не качаем. Если файл битый, качаем заново. Тест в Task 3.

---

## Файловая структура (что появится после плана)

```
movielens-recsys/
├── .github/workflows/ci.yml
├── .gitignore
├── pyproject.toml / uv.lock / .python-version
├── src/recsys/
│   ├── __init__.py                 # __version__
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py             # скачать + проверить MD5 + распаковать
│   │   ├── movielens.py            # парсинг ratings/movies/users.dat → DataFrame
│   │   ├── validate.py             # проверки качества данных
│   │   └── split.py                # per-user temporal split
│   ├── metrics.py                  # ndcg_at_k, recall_at_k, coverage, evaluate
│   ├── models/
│   │   ├── __init__.py
│   │   └── popularity.py           # PopularityRecommender
│   └── pipelines/
│       ├── __init__.py
│       └── baseline.py             # CLI: данные → split → popularity → метрики
├── notebooks/01_eda.ipynb
└── tests/unit/
    ├── test_smoke.py
    ├── test_download.py
    ├── test_movielens.py
    ├── test_validate.py
    ├── test_split.py
    ├── test_metrics.py
    ├── test_popularity.py
    └── test_baseline.py
```

---

## Этап 0. Окружение

### Task 1: Docker Desktop с данными на D:

**Files:** создаётся `C:\Users\Krasn\.wslconfig` (вне репозитория)

- [ ] **Step 1 [ты]: Установить Docker Desktop для Windows** с docker.com (бесплатно для личного использования). В установщике оставить «Use WSL 2».
- [ ] **Step 2 [ты]: Перенести диск Docker на D:.** Docker Desktop → Settings → Resources → Advanced → *Disk image location* = `D:\DockerData` → Apply & Restart.
- [ ] **Step 3: Ограничить память WSL.** Создать `C:\Users\Krasn\.wslconfig`:
  ```ini
  [wsl2]
  memory=6GB
  ```
  Затем выполнить `wsl --shutdown` и перезапустить Docker Desktop.
- [ ] **Step 4: Проверить.**
  Run: `docker run --rm hello-world`
  Expected: в выводе есть `Hello from Docker!`
  Run: `docker info --format "{{.MemTotal}}"`
  Expected: число около 6 ГБ (≈ 6.2e9 байт или меньше)

### Task 2: Скелет проекта, uv, ruff, pytest, GitHub

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `src/recsys/__init__.py`, `tests/unit/test_smoke.py`

**Interfaces:**
- Produces: импортируемый пакет `recsys` с `recsys.__version__ == "0.1.0"`; команды `uv run pytest` и `uv run ruff check .` работают из корня.

- [ ] **Step 1 [ты]: Установить uv:** `winget install --id=astral-sh.uv -e`, затем перезапустить терминал.
  Run: `uv --version`
  Expected: `uv 0.x.y`
- [ ] **Step 2: Инициализировать проект:** `uv init --package --no-readme --name recsys --python 3.12 .` (README пишем только на этапе 8; создаёт `pyproject.toml` и `src/recsys/__init__.py`). В `pyproject.toml` задать `requires-python = ">=3.12,<3.13"` и `version = "0.1.0"`, а в `__init__.py` оставить только `__version__ = "0.1.0"`.
- [ ] **Step 3: Зависимости:** `uv add pandas numpy pyarrow` и `uv add --dev pytest ruff`. Добавить в `pyproject.toml`:
  ```toml
  [tool.ruff]
  line-length = 100
  target-version = "py312"

  [tool.ruff.lint]
  select = ["E", "F", "I", "B", "UP"]

  [tool.pytest.ini_options]
  testpaths = ["tests"]
  ```
- [ ] **Step 4: `.gitignore`:** `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `data/`, `.ipynb_checkpoints/`, `.env`, `mlruns/`.
- [ ] **Step 5: Smoke-тест** `tests/unit/test_smoke.py`:
  ```python
  import recsys

  def test_package_importable():
      assert recsys.__version__ == "0.1.0"
  ```
  Run: `uv run pytest -v`
  Expected: `1 passed`
- [ ] **Step 6: Ранняя проверка риска этапа 2.** Убедиться, что ML-библиотеки ставятся на Windows. В ветку ничего не добавляем:
  Run: `uv run --with implicit --with lightgbm python -c "import implicit, lightgbm; print(implicit.__version__, lightgbm.__version__)"`
  Expected: две версии без ошибок. Если `implicit` не ставится (нет wheel под Windows), записать это в план этапа 2: обучение будет идти в Docker или WSL.
- [ ] **Step 7: Линтер чистый.**
  Run: `uv run ruff check . && uv run ruff format --check .`
  Expected: `All checks passed!` и `... files already formatted`
- [ ] **Step 8: Commit.**
  ```bash
  git add pyproject.toml uv.lock .python-version .gitignore src tests
  git commit -m "chore: project skeleton with uv, ruff, pytest"
  ```
- [ ] **Step 9 [ты]: Создать пустой репозиторий на GitHub** `movielens-recsys` без README и без .gitignore. Для портфолио он должен быть публичным, но можно начать с приватного. Затем, после явного «да» на push:
  ```bash
  git remote add origin https://github.com/<login>/movielens-recsys.git
  git push -u origin main
  ```
  Expected: на странице репозитория видны файлы.

---

## Этап 1. Данные и базовая линия

### Task 3: Скачивание и парсинг MovieLens 1M

**Files:**
- Create: `src/recsys/data/__init__.py`, `src/recsys/data/download.py`, `src/recsys/data/movielens.py`
- Test: `tests/unit/test_download.py`, `tests/unit/test_movielens.py`

**Interfaces:**
- Produces:
  - `ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"`, `ML1M_MD5: str`
  - `download_ml1m(dest_dir: Path, url: str = ML1M_URL, md5: str = ML1M_MD5) -> Path` возвращает путь к распакованной папке `dest_dir / "ml-1m"`
  - `file_md5(path: Path) -> str`
  - `load_ratings(raw_dir: Path) -> pd.DataFrame` с колонками `user_id:int32, movie_id:int32, rating:int8, timestamp:int64`
  - `load_movies(raw_dir: Path) -> pd.DataFrame` с колонками `movie_id:int32, title:str, year:Int16 (nullable), genres:list[str]`
  - `load_users(raw_dir: Path) -> pd.DataFrame` с колонками `user_id:int32, gender:str, age:int8, occupation:int8, zip:str`

Формат файлов ML-1M: разделитель `::`, без заголовка, кодировка latin-1.
`ratings.dat`: `UserID::MovieID::Rating::Timestamp`; `movies.dat`: `MovieID::Title::Genres` (жанры через `|`); `users.dat`: `UserID::Gender::Age::Occupation::Zip-code`.

- [ ] **Step 1: Тесты парсинга** (`test_movielens.py`). Фикстура пишет мини-файлы в `tmp_path` **в кодировке latin-1**:
  ```python
  MOVIES = (
      "1::Toy Story (1995)::Animation|Children's|Comedy\n"
      "2::City of Lost Children, The (Cité des enfants perdus, La) (1995)::Adventure|Sci-Fi\n"
      "3::Misérables, Les (1995)::Drama\n"
  )
  RATINGS = "1::1::5::978300760\n1::2::3::978302109\n2::3::4::978301968\n"
  USERS = "1::F::1::10::48067\n2::M::56::16::70072\n"

  def test_load_movies_parses_year_title_genres(raw_dir):
      m = load_movies(raw_dir).set_index("movie_id")
      assert m.loc[1, "title"] == "Toy Story"
      assert m.loc[1, "year"] == 1995
      assert m.loc[1, "genres"] == ["Animation", "Children's", "Comedy"]
      assert m.loc[2, "title"] == "City of Lost Children, The (Cité des enfants perdus, La)"
      assert m.loc[3, "title"] == "Misérables, Les"          # latin-1 читается корректно

  def test_load_ratings_columns_and_dtypes(raw_dir):
      r = load_ratings(raw_dir)
      assert list(r.columns) == ["user_id", "movie_id", "rating", "timestamp"]
      assert len(r) == 3 and r["rating"].dtype == "int8"

  def test_load_users(raw_dir):
      u = load_users(raw_dir).set_index("user_id")
      assert u.loc[2, "age"] == 56 and u.loc[1, "zip"] == "48067"
  ```
- [ ] **Step 2: Запустить и увидеть падение.** Run: `uv run pytest tests/unit/test_movielens.py -v`. Expected: FAIL `ImportError`.
- [ ] **Step 3: Реализовать `movielens.py`.** `pd.read_csv(sep="::", engine="python", encoding="latin-1", header=None, names=[...])`. Год вытаскивается регуляркой `r"\s*\((\d{4})\)\s*$"`: группа идёт в `year`, а сам матч вырезается из `title`. Если года нет, `year = <NA>`, title не меняется.
- [ ] **Step 4: Тесты проходят.** Run: `uv run pytest tests/unit/test_movielens.py -v`. Expected: `3 passed`.
- [ ] **Step 5: Тесты скачивания** (`test_download.py`). Без сети: `url` указывает на локальный zip через `Path.as_uri()` (`file:///...`), в модуле `from urllib.request import urlopen`, чтобы тест мог подменить `recsys.data.download.urlopen`.
  ```python
  def test_download_extracts_and_verifies(tmp_path, fake_zip):   # fake_zip: (uri, md5)
      out = download_ml1m(tmp_path, url=fake_zip.uri, md5=fake_zip.md5)
      assert (out / "ratings.dat").exists()

  def test_download_skips_when_archive_valid(tmp_path, fake_zip, monkeypatch):
      download_ml1m(tmp_path, url=fake_zip.uri, md5=fake_zip.md5)
      monkeypatch.setattr("recsys.data.download.urlopen", _fail)   # второй вызов сеть не трогает
      download_ml1m(tmp_path, url=fake_zip.uri, md5=fake_zip.md5)

  def test_download_redownloads_corrupted_archive(tmp_path, fake_zip):
      (tmp_path / "ml-1m.zip").write_bytes(b"garbage")
      out = download_ml1m(tmp_path, url=fake_zip.uri, md5=fake_zip.md5)
      assert (out / "ratings.dat").exists()

  def test_download_raises_on_md5_mismatch(tmp_path, fake_zip):
      with pytest.raises(ValueError, match="MD5"):
          download_ml1m(tmp_path, url=fake_zip.uri, md5="0" * 32)
  ```
- [ ] **Step 6: Запустить и увидеть падение.** Run: `uv run pytest tests/unit/test_download.py -v`. Expected: FAIL.
- [ ] **Step 7: Реализовать `download.py`.** Шаги: если архив есть и MD5 совпадает, пропустить скачивание; иначе скачать во временный файл `.part`, проверить MD5 (при несовпадении `ValueError("MD5 mismatch ...")`, `.part` удаляется), переименовать и распаковать `zipfile`. Значение `ML1M_MD5` взять из официального `https://files.grouplens.org/datasets/movielens/ml-1m.zip.md5` (скачивание подтверждает пользователь) и записать константой.
- [ ] **Step 8: Тесты проходят.** Run: `uv run pytest tests/unit -v`. Expected: все passed.
- [ ] **Step 9: Скачать настоящие данные** (≈6 МБ, с подтверждением пользователя):
  Run: `uv run python -c "from pathlib import Path; from recsys.data.download import download_ml1m; from recsys.data.movielens import load_ratings, load_movies, load_users as lu; d=download_ml1m(Path('data/raw')); print(len(load_ratings(d)), len(load_movies(d)), len(lu(d)))"`
  Expected: `1000209 3883 6040`
- [ ] **Step 10: Commit.** `git add src/recsys/data tests/unit && git commit -m "feat(data): download and parse MovieLens 1M"`

### Task 4: Валидация данных

**Files:**
- Create: `src/recsys/data/validate.py`
- Test: `tests/unit/test_validate.py`

**Interfaces:**
- Consumes: DataFrame формата `load_ratings` (Task 3).
- Produces: `class DataValidationError(Exception)` с атрибутом `problems: list[str]`; `validate_ratings(ratings: pd.DataFrame, previous_count: int | None = None, max_drop: float = 0.05) -> None`, который при нарушениях бросает исключение со **всеми** найденными проблемами сразу.

- [ ] **Step 1: Тесты.**
  ```python
  def test_valid_data_passes(): validate_ratings(df_ok)            # без исключения
  def test_nulls_detected():     _problems(df_with_nan)   contains "null"
  def test_rating_out_of_range(): _problems(df_with_rating_6) contains "rating"
  def test_duplicates_detected(): _problems(df_dup_pair)  contains "duplicate"
  def test_count_drop_detected():
      _problems(df_100_rows, previous_count=200) contains "dropped"      # падение на 50% > 5%
  def test_small_drop_allowed():  validate_ratings(df_100_rows, previous_count=104)  # 3.8% < 5%
  def test_all_problems_reported_together():
      err = _problems(df_with_nan_and_dup); assert len(err) >= 2
  ```
  (`_problems` — хелпер в тесте: ловит `DataValidationError` и возвращает `.problems`, проверка через подстроку.)
- [ ] **Step 2: Запустить и увидеть падение.** Run: `uv run pytest tests/unit/test_validate.py -v`. Expected: FAIL.
- [ ] **Step 3: Реализовать `validate_ratings`.** Проверки: NULL в любой колонке; `rating ∉ 1..5`; дубли `(user_id, movie_id)`; `len < previous_count * (1 - max_drop)`.
- [ ] **Step 4: Тесты проходят, плюс проверка на реальных данных.** Run: `uv run pytest tests/unit/test_validate.py -v`. Expected: `7 passed`.
  Run: `uv run python -c "from pathlib import Path; from recsys.data.movielens import load_ratings; from recsys.data.validate import validate_ratings; validate_ratings(load_ratings(Path('data/raw/ml-1m'))); print('ok')"`. Expected: `ok`.
- [ ] **Step 5: Commit.** `git commit -m "feat(data): ratings validation"`

### Task 5: Per-user temporal split

**Files:**
- Create: `src/recsys/data/split.py`
- Test: `tests/unit/test_split.py`

**Interfaces:**
- Consumes: DataFrame формата `load_ratings`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class Split:
      als_train: pd.DataFrame
      ranker_train: pd.DataFrame
      test: pd.DataFrame

  def temporal_split(ratings: pd.DataFrame, n_ranker: int = 5, n_test: int = 5,
                     min_ratings: int = 15) -> Split
  ```
  Все три части имеют колонки как у входа, индекс сброшен.

- [ ] **Step 1: Тесты** (синтетика: пользователь A с 20 оценками, B с 10, C с 15 с одинаковыми timestamp).
  ```python
  def test_sizes_per_user():
      s = temporal_split(df)
      assert _count(s.test, "A") == 5 and _count(s.ranker_train, "A") == 5
      assert _count(s.als_train, "A") == 10

  def test_short_history_goes_to_als_train_only():
      s = temporal_split(df)
      assert _count(s.als_train, "B") == 10
      assert _count(s.test, "B") == 0 and _count(s.ranker_train, "B") == 0

  def test_exactly_min_ratings_is_split():          # граница: 15 → 5/5/5
      assert _count(temporal_split(df).test, "C") == 5

  def test_no_future_leakage():
      s = temporal_split(df)
      for u in users_with_split:
          assert s.als_train[u].timestamp.max() <= s.ranker_train[u].timestamp.min()
          assert s.ranker_train[u].timestamp.max() <= s.test[u].timestamp.min()

  def test_parts_are_disjoint_and_complete():
      s = temporal_split(df)
      parts = pd.concat([s.als_train, s.ranker_train, s.test])
      assert len(parts) == len(df)
      assert not parts.duplicated(["user_id", "movie_id"]).any()

  def test_ties_broken_by_hash_deterministically():
      # у C все timestamp равны: порядок задаёт хэш (user_id, movie_id), не movie_id;
      # результат не зависит от порядка строк на входе, три части не пересекаются
      s = temporal_split(df_c_ties)
      assert temporal_split(df_c_ties.sample(frac=1, random_state=1)).test.equals(s.test)
  ```
- [ ] **Step 2: Запустить и увидеть падение.** Run: `uv run pytest tests/unit/test_split.py -v`. Expected: FAIL.
- [ ] **Step 3: Реализовать `temporal_split`.** Сортировка `sort_values(["user_id", "timestamp", "movie_id"])`, затем позиция с конца `groupby("user_id").cumcount(ascending=False)` и размер группы через `transform("size")`. Векторно, без цикла по пользователям: на 1М строк это должно занимать секунды.
- [ ] **Step 4: Тесты проходят, плюс проверка на реальных данных.** Run: `uv run pytest tests/unit/test_split.py -v`. Expected: `6 passed`.
  Run: `uv run python -c "from pathlib import Path; from recsys.data.movielens import load_ratings; from recsys.data.split import temporal_split; s=temporal_split(load_ratings(Path('data/raw/ml-1m'))); print(len(s.als_train), len(s.ranker_train), len(s.test))"`
  Expected: `ranker_train = test = 30200` (6040 × 5, у всех пользователей ML-1M ≥ 20 оценок), `als_train = 939809`.
- [ ] **Step 5: Commit.** `git commit -m "feat(data): per-user temporal split"`

### Task 6: Метрики (TDD с ручным расчётом)

**Files:**
- Create: `src/recsys/metrics.py`
- Test: `tests/unit/test_metrics.py`

**Interfaces:**
- Produces:
  ```python
  def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int = 10) -> float
  def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int = 10) -> float
  def coverage(recommendations: Mapping[int, Sequence[int]], n_items: int) -> float
  def evaluate(recommendations: Mapping[int, Sequence[int]], test: pd.DataFrame,
               n_items: int, k: int = 10) -> dict[str, float]
      # -> {"ndcg_at_10": ..., "recall_at_10": ..., "coverage": ...}; ключи с фактическим k
  ```
  Релевантность бинарная: релевантны все `movie_id` пользователя в `test`. Повторы в `recommended` засчитываются один раз (первое вхождение). Пустой `relevant` даёт `0.0`. `evaluate` усредняет по пользователям из `test`; если пользователя нет в `recommendations`, его метрики равны 0.

- [ ] **Step 1: Тесты с ручным расчётом.**
  ```python
  def test_ndcg_hand_computed():
      # DCG = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG = 1 + 1/log2(3) = 1.6309298
      assert ndcg_at_k([1, 2, 3], {1, 3}, k=3) == pytest.approx(0.9197208, abs=1e-6)

  def test_ndcg_perfect_and_zero():
      assert ndcg_at_k([1, 2], {1, 2}, k=2) == pytest.approx(1.0)
      assert ndcg_at_k([5, 6], {1, 2}, k=2) == 0.0

  def test_ndcg_respects_k():
      assert ndcg_at_k([9, 9, 1], {1}, k=2) == 0.0

  def test_duplicates_not_double_counted():
      assert ndcg_at_k([1, 1, 1], {1}, k=3) == pytest.approx(1.0)
      assert recall_at_k([1, 1], {1, 2}, k=2) == pytest.approx(0.5)

  def test_short_list_ok():
      assert ndcg_at_k([1], {1, 2}, k=10) == pytest.approx(1 / (1 + 1 / math.log2(3)))

  def test_empty_relevant_is_zero():
      assert ndcg_at_k([1], set(), k=10) == 0.0 and recall_at_k([1], set(), k=10) == 0.0

  def test_recall():
      assert recall_at_k([1, 2, 3], {1, 3, 7, 8}, k=3) == pytest.approx(0.5)

  def test_coverage():
      assert coverage({1: [1, 2], 2: [2, 3]}, n_items=10) == pytest.approx(0.3)

  def test_evaluate_averages_and_missing_user_is_zero():
      test = pd.DataFrame({"user_id": [1, 2], "movie_id": [10, 20]})
      res = evaluate({1: [10]}, test, n_items=100, k=10)
      assert res["ndcg_at_10"] == pytest.approx(0.5)      # (1.0 + 0.0) / 2
      assert res["recall_at_10"] == pytest.approx(0.5)
      assert res["coverage"] == pytest.approx(0.01)
  ```
- [ ] **Step 2: Запустить и увидеть падение.** Run: `uv run pytest tests/unit/test_metrics.py -v`. Expected: FAIL.
- [ ] **Step 3: Реализовать `metrics.py`.** `dedup = list(dict.fromkeys(recommended))[:k]`; DCG = Σ 1/log2(i+2) по попаданиям; IDCG по `min(len(relevant), k)` идеальным позициям.
- [ ] **Step 4: Тесты проходят.** Run: `uv run pytest tests/unit/test_metrics.py -v`. Expected: `9 passed`.
- [ ] **Step 5: Commit.** `git commit -m "feat: ranking metrics ndcg/recall/coverage"`

### Task 7: Popularity-бейзлайн

**Files:**
- Create: `src/recsys/models/__init__.py`, `src/recsys/models/popularity.py`
- Test: `tests/unit/test_popularity.py`

**Interfaces:**
- Consumes: DataFrame формата `load_ratings`.
- Produces:
  ```python
  class PopularityRecommender:
      def fit(self, ratings: pd.DataFrame) -> "PopularityRecommender"
      def recommend(self, k: int = 10, exclude: set[int] | None = None) -> list[int]
      def recommend_batch(self, user_ids: Iterable[int], seen: Mapping[int, set[int]],
                          k: int = 10) -> dict[int, list[int]]
  ```
  Популярность = число оценок фильма. При равенстве меньший `movie_id` идёт первым. `recommend` до `fit` бросает `RuntimeError`.

- [ ] **Step 1: Тесты.**
  ```python
  def test_orders_by_count_then_movie_id():
      # фильм 2: 3 оценки, фильмы 1 и 3: по 1
      assert PopularityRecommender().fit(df).recommend(k=3) == [2, 1, 3]

  def test_excludes_seen():
      assert PopularityRecommender().fit(df).recommend(k=2, exclude={2}) == [1, 3]

  def test_k_larger_than_catalog_returns_all_unseen():
      assert len(PopularityRecommender().fit(df).recommend(k=100, exclude={1})) == 2

  def test_batch_uses_per_user_seen():
      recs = PopularityRecommender().fit(df).recommend_batch([7, 8], {7: {2}, 8: set()}, k=1)
      assert recs == {7: [1], 8: [2]}

  def test_recommend_before_fit_raises():
      with pytest.raises(RuntimeError):
          PopularityRecommender().recommend()
  ```
- [ ] **Step 2: Запустить и увидеть падение.** Run: `uv run pytest tests/unit/test_popularity.py -v`. Expected: FAIL.
- [ ] **Step 3: Реализовать.** В `fit` сохраняется отсортированный массив `movie_id`. `recommend` идёт по нему и пропускает `exclude`, пока не наберёт k.
- [ ] **Step 4: Тесты проходят.** Run: `uv run pytest tests/unit/test_popularity.py -v`. Expected: `5 passed`.
- [ ] **Step 5: Commit.** `git commit -m "feat(models): popularity baseline"`

### Task 8: Пайплайн бейзлайна + EDA-ноутбук

**Files:**
- Create: `src/recsys/pipelines/__init__.py`, `src/recsys/pipelines/baseline.py`, `notebooks/01_eda.ipynb`
- Test: `tests/unit/test_baseline.py`

**Interfaces:**
- Consumes: `load_ratings`, `load_movies` (Task 3), `validate_ratings` (Task 4), `temporal_split` (Task 5), `evaluate` (Task 6), `PopularityRecommender` (Task 7).
- Produces: `run_baseline(raw_dir: Path, k: int = 10) -> dict[str, float]` и CLI `uv run python -m recsys.pipelines.baseline --data-dir data/raw/ml-1m [--k 10]`, который печатает метрики по строке: `ndcg_at_10: 0.xxxx`.
  Логика: загрузить → провалидировать → split → `fit` на `als_train ∪ ranker_train` (всё до теста) → `seen` = те же данные по пользователю → `recommend_batch` для пользователей из `test` → `evaluate(n_items = число фильмов в movies)`.

- [ ] **Step 1: Тест на синтетике.** В `tmp_path` пишутся `ratings.dat` и `movies.dat` в формате ML-1M (30 пользователей × 20 оценок, фильм 1 оценили все). Проверки: ключи `{"ndcg_at_10", "recall_at_10", "coverage"}`, все значения в `[0, 1]`, два запуска дают одинаковый результат.
- [ ] **Step 2: Запустить и увидеть падение.** Run: `uv run pytest tests/unit/test_baseline.py -v`. Expected: FAIL.
- [ ] **Step 3: Реализовать `run_baseline` и `main()` на `argparse`** (`if __name__ == "__main__": main()`).
- [ ] **Step 4: Тест проходит, затем прогон на реальных данных.** Run: `uv run pytest -v`. Expected: все passed.
  Run: `uv run python -m recsys.pipelines.baseline --data-dir data/raw/ml-1m`
  Expected: три строки с метриками, `ndcg_at_10 > 0`, `coverage < 0.1` (популярное всем советует почти одно и то же, ожидаем единицы процентов). Числа записать: это планка, которую ALS обязан побить на этапе 2.
- [ ] **Step 5: EDA-ноутбук.** `uv add --group notebooks jupyter matplotlib`. В `notebooks/01_eda.ipynb` пять вопросов, у каждого ячейка с графиком и вывод одной строкой:
  1. распределение оценок 1–5;
  2. число оценок на пользователя (лог-шкала: длинный хвост);
  3. число оценок на фильм (лог-шкала: сколько фильмов почти никто не оценил);
  4. период данных: даты первой и последней оценки, гистограмма по месяцам;
  5. сколько оценок у пользователя с одинаковым timestamp (почему нужен tie-break в split).

  Run: `uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb`. Expected: выполняется без ошибок.
- [ ] **Step 6: Commit.** `git commit -m "feat: baseline pipeline and EDA notebook"`

### Task 9: CI в GitHub Actions

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: команды из Task 2 (`uv sync`, `ruff`, `pytest`).

- [ ] **Step 1: Написать workflow.** Триггеры `push` и `pull_request`, `ubuntu-latest`. Шаги: `actions/checkout` → `astral-sh/setup-uv` (с `enable-cache: true`) → `uv sync --frozen` → `uv run ruff check .` → `uv run ruff format --check .` → `uv run pytest -v`. Мажорные версии actions взять актуальные со страниц репозиториев на момент реализации и записать явно (`@vN`).
- [ ] **Step 2: Проверить локально те же команды.** Run: `uv sync --frozen && uv run ruff check . && uv run ruff format --check . && uv run pytest`. Expected: всё зелёное.
- [ ] **Step 3: Commit и push** (push после явного «да»): `git commit -m "ci: lint and tests on every push"` → `git push`.
- [ ] **Step 4: Проверить в GitHub.** Run: `gh run watch` (если установлен `gh`) или вкладка Actions. Expected: зелёная галочка на последнем коммите.
- [ ] **Step 5: Сломать и починить (учебный шаг).** Временно испортить ассерт в `test_smoke.py` в отдельной ветке, сделать push и убедиться, что CI красный. Ветку потом удалить. Так видно, что CI действительно ловит ошибки.

---

## Критерии готовности этапов 0–1

- [ ] `docker run --rm hello-world` работает, данные Docker на `D:\DockerData`
- [ ] `uv run pytest` зелёный (≥ 35 тестов), `ruff` чистый
- [ ] `python -m recsys.pipelines.baseline` печатает NDCG@10 / Recall@10 / Coverage на ML-1M, числа записаны
- [ ] EDA-ноутбук выполняется целиком
- [ ] На GitHub зелёная галочка CI
