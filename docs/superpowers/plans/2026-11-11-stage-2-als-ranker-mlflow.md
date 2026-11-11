# Этап 2: ALS + ранжировщик, эксперименты в MLflow. План реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Двухэтапная модель (кандидаты ALS + LightGBM `lambdarank`), честно сравнённая с popularity на одном и том же тестовом сплите по NDCG@10 / Recall@10 / Coverage. Все запуски (включая маленький sweep гиперпараметров ALS) пишутся в MLflow, лучшая модель регистрируется как `movielens-recommender` с alias `champion`. Интерфейсы fold-in и загрузки чемпиона готовы для API этапа 3.

**Architecture:** Пакет `recsys` получает: `evaluation.py` (единый протокол оценки для всех моделей), `models/als.py` (генератор кандидатов + fold-in), `features.py` (чистые функции признаков, общие для обучения и сервиса), `models/ranker.py` (обёртка `LGBMRanker`), `models/two_stage.py` (сборка ALS + признаки + ранжировщик, сохранение в папку-бандл), `models/pyfunc.py` (MLflow pyfunc поверх бандла), `tracking.py` (MLflow: URI, эксперимент, теги, регистрация, alias, загрузка чемпиона), `pipelines/train.py` (CLI обучения). Ядро обучения (`train_and_evaluate`) не знает про MLflow, MLflow-слой тонкий и тестируется на временной sqlite-базе.

**Tech Stack:** Python 3.12, uv, pandas, numpy, scipy (CSR-матрицы), `implicit` 0.7.3 (CPU ALS), `lightgbm` 4.7.0 (`LGBMRanker`), `threadpoolctl` (BLAS в один поток для воспроизводимости), `mlflow` 3.18 (tracking + Model Registry, локальный backend `sqlite:///mlruns/mlflow.db`), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-11-movielens-recsys-design.md` (секции 2.1–2.5, 3.2, 4.3, 6.3, 6.5, 6.6)

## Global Constraints

- Всё из плана этапов 0–1 остаётся в силе: `src/`-layout, зависимости только через `uv add`, ruff (line-length 100, py312, правила E,F,I,B,UP, ноутбуки тоже проверяются), код и коммиты на английском.
- Данные, `mlruns/` и любые большие файлы живут на `D:` внутри репозитория (`data/` и `mlruns/` в `.gitignore`). На `C:` ничего не пишем: `UV_CACHE_DIR=D:/uv-cache`.
- CI (Linux) запускает `uv sync --frozen`, ruff check, ruff format --check, pytest с `UV_FROZEN=1`. Тесты не ходят в сеть, не читают реальные данные, не требуют MLflow-сервера и укладываются в секунды (весь набор заметно меньше минуты).
- Сплит не меняется: per-user temporal, `n_ranker=5`, `n_test=5`, `min_ratings=15`. Метрики: K=10, главная NDCG@10, relevant = все фильмы пользователя в части-мишени. `n_items` для coverage = число фильмов в `movies.dat` (как у бейзлайна).
- Детерминизм: ALS — `random_state` + BLAS в один поток (`threadpool_limits(1, "blas")`); LightGBM — `seed`, `deterministic=True`, `force_row_wise=True`; разбиение пользователей ранжировщика на train/valid и маскирование демографии — детерминированные по `seed`. На синтетике два прогона дают одинаковые числа (тест); на реальных данных допускаются отличия в последних знаках.
- Тюнинг гиперпараметров **никогда** не смотрит на `test`: sweep ALS оценивается на валидационном протоколе (ALS на `als_train`, мишень `ranker_train`).
- Коммиты заканчиваются пустой строкой и `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `git push` не делаем. README не создаём (этап 8).

## Protocol (как обучаем и оцениваем)

```
split: [ als_train | ranker_train (5) | test (5) ]

Стадия A (данные для ранжировщика):
  ALS_A.fit(als_train) → 100 кандидатов на пользователя из ranker_train (seen = als_train)
  признаки из als_train (популярность, средние, жанровые профили), демография из users.dat
  метка = 1, если кандидат есть в ranker_train пользователя, иначе 0
  LGBMRanker(lambdarank) учится на этих группах (early stopping на 10% пользователей)

Стадия B (оценка на test):
  history = als_train ∪ ranker_train
  ALS_B.fit(history) → 100 кандидатов на пользователя из test (seen = history)
  признаки из history
  popularity.fit(history) | ALS-only = топ-10 кандидатов ALS_B | ALS+ranker = переранжированные кандидаты
  все три модели → один и тот же evaluate(…, test, n_items)
```

Ранжировщик никогда не видит `test`; ALS, по скору которого ранжировщик учится, не видит меток ранжировщика. Регистрируется ровно оценённая модель (ALS_B + ранжировщик + таблица признаков).

## Decisions

1. **Локальный MLflow backend вместо Postgres + MLflow в compose.** Spec 6.6 упоминает compose на этапе 2, но Docker локально недоступен. Tracking URI по умолчанию `sqlite:///mlruns/mlflow.db`, артефакты в `mlruns/artifacts`; всё переопределяется `MLFLOW_TRACKING_URI` (или `--tracking-uri`), так что этапы 3–4 просто указывают на сервер MLflow. Compose появится на этапе 3.
2. **Регистрируется оценённая модель, без дообучения на test.** «Что измерили, то и выкатываем»: метрики версии в реестре относятся ровно к её весам. Последние 5 оценок пользователей ML-1M в факторы не попадают, но в сервисе пользователь всё равно идёт через fold-in по своим оценкам из Postgres, так что это не теряется. Этап 4 может пересмотреть.
3. **Бинарные метки ранжировщика.** Метрика бинарная (релевантен любой фильм из мишени, независимо от оценки), поэтому метка = «оценил ли пользователь кандидата в `ranker_train`». Градуированные метки оптимизировали бы другую цель.
4. **Confidence-веса.** `implicit` трактует значение в матрице как confidence (`C = alpha · w(r)`, ненулевые клетки — позитивы). Схемы `w(r)`: `linear` (`w = r`, 5 — сильный сигнал, 1 — слабый, по spec 2.2) и `binary` (`w = 1`) как контрольная точка sweep. По умолчанию берётся лучшая из **соответствующих spec** схем; если `binary` окажется лучше, это записывается в отчёт, но не выбирается молча.
5. **Маскирование демографии при обучении ранжировщика (`demographics_dropout=0.1`).** У веб-пользователей возраст/пол/профессия = NaN (spec 2.3), а у всех пользователей ML-1M они есть. Если ранжировщик ни разу не видел NaN, LightGBM на сервинге отправит NaN по «нулевой» ветке (пол 0 = F, профессия 0 = other). Поэтому у детерминированной доли 10% пользователей в обучающей выборке демография заменяется на NaN.
6. **Жанровая близость пользователь × фильм.** «Доля оценок пользователя в жанрах фильма» считается так: `share[u, g]` = доля оценённых пользователем фильмов с жанром `g`; признаки пары — `genre_affinity_mean` (среднее `share[u, g]` по жанрам фильма) и `genre_affinity_max` (максимум). Векторизуется матричным умножением, на 600k строк — доли секунды.
7. **Группы без позитивов выбрасываются** из обучения ранжировщика: в lambdarank у них нулевой градиент, они только тратят время.
8. **Сравнение с текущим чемпионом на этапе 2.** Spec 4.2 требует сравнивать на одном сплите. На этапе 2 данные фиксированы, поэтому метрики чемпиона берутся из тегов его версии, **если** его `data_snapshot` совпадает с текущим (тот же снапшот → тот же детерминированный сплит). При другом снапшоте alias не переносится, причина пишется в тег; честная переоценка чемпиона на новом сплите — задача `evaluate` этапа 4. Правило `decide_promotion` реализовано по spec 4.2: чемпиона нет, или `NDCG@10(challenger) ≥ NDCG@10(champion) − 0.005` и coverage упал не больше чем на 10% (относительно).
9. **Выбор варианта для сервиса (spec 2.5).** Если `ndcg_at_10(ALS+ranker) ≤ ndcg_at_10(ALS-only)`, регистрируется бандл без ранжировщика (`variant=als_only`). Если ALS-only не обгоняет popularity, это печатается как предупреждение и пишется тегом `beats_popularity=false`.
10. **Pyfunc без `code_paths`.** Модель ссылается на установленный пакет `recsys` (в образе API он есть). Копия кода внутри модели могла бы затенить установленный `recsys` и дать train/serve skew в `features.py`. `pip_requirements` задаются явно, чтобы MLflow не запускал медленный вывод зависимостей.
11. **Вход pyfunc — оценки.** `predict(DataFrame[user_id, movie_id, rating], params={"k": 10})` делает fold-in по переданным оценкам и возвращает `DataFrame[user_id, rank, movie_id, score]`. Это единый путь для любого пользователя (API всегда может достать оценки из Postgres). Для известных пользователей есть отдельный метод с сохранёнными факторами (им пользуется оценка).

## Review Focus

1. **Утечка через ALS для ранжировщика.** ALS, дающий кандидатов и скор для обучения ранжировщика, обязан быть обучен только на `als_train`; признаки стадии A тоже считаются только по `als_train`. Тест в Task 7 проверяет число оценок, на которых обучены ALS_A и ALS_B, и неизменность метрик при порче `test` у sweep.
2. **Разные протоколы у моделей.** Popularity, ALS-only и ALS+ranker должны видеть одну и ту же историю, одних и тех же тестовых пользователей и одинаковую фильтрацию seen. Единый `EvalContext` + проверка «нет seen в топ-k» в `evaluate_recommendations` (Task 2).
3. **Train/serve skew признаков.** Один и тот же пользователь через батч-путь (обучение/оценка) и через fold-in путь (сервис) должен получить идентичные строки признаков при тех же кандидатах; веб-пользователь без демографии не ломает ранжировщик (NaN). Тесты в Task 4 и Task 6.
4. **Края `implicit`.** Когда непросмотренных фильмов меньше `N`, `recommend` возвращает «пустые» позиции (скор `-inf`/мусорные id) — их надо отбросить; id внутри `implicit` — номера строк/столбцов, не `movie_id`; новый пользователь с фильмами, которых нет в модели. Тесты в Task 3.
5. **MLflow-регистрация и alias.** Второй, более слабый запуск не должен забрать alias `champion`; загрузка `models:/movielens-recommender@champion` должна вернуть рабочий рекомендатор; тесты не должны трогать `./mlruns` (временная sqlite-база). Тесты в Task 8.

---

## Файловая структура (что добавится)

```
src/recsys/
├── evaluation.py            # EvalContext, build_history, единый протокол, таблица сравнения
├── features.py              # GENRES, FEATURE_COLUMNS, movie/user/pair features (train == serve)
├── tracking.py              # MLflow: URI, эксперимент, теги, регистрация, alias, load_champion
├── models/
│   ├── als.py               # ALSConfig, confidence_weights, ALSCandidateGenerator (fit, candidates, fold-in, save/load)
│   ├── ranker.py            # RankerConfig, label_candidates, build_training_frame, LambdaRanker
│   ├── two_stage.py         # TwoStageRecommender (ALS + признаки + ranker), save/load бандла
│   └── pyfunc.py            # RecommenderPyfunc (mlflow.pyfunc.PythonModel)
└── pipelines/
    ├── baseline.py          # (изменён) использует evaluation.py, проверка k >= 1
    └── train.py             # train_and_evaluate, validate_als, sweep, CLI
tests/unit/
├── synthetic.py             # общий генератор синтетики с «вкусами» (кластеры жанров)
├── test_evaluation.py  test_als.py  test_features.py  test_ranker.py
├── test_two_stage.py  test_train_pipeline.py  test_tracking.py
└── (изменены) test_validate.py  test_metrics.py  test_baseline.py
```

---

### Task 1: Зависимости и долги этапа 1 (валидация оценок, `k >= 1`)

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (`uv add implicit lightgbm mlflow scipy threadpoolctl`), `src/recsys/data/validate.py`, `src/recsys/metrics.py`, `src/recsys/pipelines/baseline.py`
- Test: `tests/unit/test_validate.py`, `tests/unit/test_metrics.py`, `tests/unit/test_baseline.py`

**Interfaces:**
- `validate_ratings` дополнительно сообщает `"N rows have non-integer rating"` (дробные значения среди не-NaN) и `"column 'rating' is not numeric"` (тогда проверка диапазона пропускается, а не падает).
- `ndcg_at_k`, `recall_at_k`, `evaluate` бросают `ValueError("k must be >= 1, got 0")` при `k < 1`.
- `recsys.pipelines.baseline.positive_int(value: str) -> int` — тип для argparse; CLI с `--k 0` завершается `SystemExit(2)`.

- [ ] **Step 1: Тесты.**
  ```python
  def test_non_integer_rating_rejected():           # rating 3.5 → problem contains "non-integer"
  def test_integer_valued_float_ratings_pass():     # float64 [1.0..5.0] без NaN → нет исключения
  def test_non_numeric_rating_reported():           # rating "5" (object) → "not numeric", не TypeError
  @pytest.mark.parametrize("fn", [ndcg_at_k, recall_at_k])
  def test_k_below_one_raises(fn): with pytest.raises(ValueError, match="k must be >= 1"): fn([1], {1}, k=0)
  def test_evaluate_k_below_one_raises():
  def test_cli_rejects_non_positive_k(monkeypatch): sys.argv = [..., "--k", "0"] → SystemExit code 2
  ```
- [ ] **Step 2: Увидеть падение** (`uv run pytest tests/unit -q`): новые тесты FAIL (нет исключения / TypeError).
- [ ] **Step 3: Реализовать**, тесты зелёные, остальные 45 не сломаны.
- [ ] **Step 4: Commit** `feat: reject non-integer ratings and k < 1; add ALS/ranker/MLflow deps`.

### Task 2: Единый протокол оценки

**Files:**
- Create: `src/recsys/evaluation.py`; Modify: `src/recsys/pipelines/baseline.py`
- Test: `tests/unit/test_evaluation.py`

**Interfaces:**
```python
class ProtocolViolation(Exception): ...

@dataclass(frozen=True)
class EvalContext:
    history: pd.DataFrame          # что модели разрешено видеть
    target: pd.DataFrame           # мишень (test или ranker_train)
    seen: dict[int, set[int]]      # по history
    users: list[int]               # отсортированные пользователи target
    n_items: int
    k: int

def build_history(split: Split) -> tuple[pd.DataFrame, dict[int, set[int]]]   # перенесён из baseline
def make_context(history, target, n_items, k=10) -> EvalContext
def test_context(split, n_items, k=10) -> EvalContext        # history = als_train ∪ ranker_train, target = test
def validation_context(split, n_items, k=10) -> EvalContext  # history = als_train, target = ranker_train
RecommendFn = Callable[[EvalContext], Mapping[int, Sequence[int]]]
def evaluate_recommendations(recs, ctx) -> dict[str, float]  # ProtocolViolation, если в топ-k есть seen
def compare_models(models: Mapping[str, RecommendFn], ctx) -> dict[str, dict[str, float]]
def format_comparison(results, k=10) -> str                  # таблица: model | ndcg_at_10 | recall_at_10 | coverage
```
`baseline.build_history` остаётся доступным (реэкспорт), `recommend_for_test_users`/`run_baseline` идут через `test_context` — golden-значения бейзлайна не меняются.

- [ ] **Step 1: Тесты** (`test_evaluation.py`):
  `test_test_context_history_is_als_plus_ranker_and_target_is_test`, `test_validation_context_uses_als_train_and_ranker_train`, `test_context_users_sorted_and_seen_from_history`, `test_evaluate_recommendations_rejects_seen_item` (`ProtocolViolation`), `test_seen_beyond_k_is_not_a_violation`, `test_compare_models_passes_same_context_to_every_model` (шпион записывает `id(ctx)`), `test_format_comparison_lists_models_in_order`.
- [ ] **Step 2: FAIL** (`ImportError`). **Step 3: Реализовать.** **Step 4:** весь набор зелёный, включая golden-тест бейзлайна.
- [ ] **Step 5: Commit** `feat: shared evaluation protocol for all models`.

### Task 3: ALS-генератор кандидатов и fold-in

**Files:**
- Create: `src/recsys/models/als.py`, `tests/unit/synthetic.py`
- Test: `tests/unit/test_als.py`

**Interfaces:**
```python
RATING_WEIGHTS = ("linear", "binary")
def confidence_weights(ratings: ArrayLike, scheme: str = "linear") -> np.ndarray   # float32

@dataclass(frozen=True)
class ALSConfig:
    factors: int = 64; regularization: float = 0.05; alpha: float = 1.0; iterations: int = 15
    rating_weight: str = "linear"; random_state: int = 42; num_threads: int = 0

CANDIDATE_COLUMNS = ["user_id", "movie_id", "als_score", "als_rank"]   # als_rank с 1

class ALSCandidateGenerator:
    def __init__(self, config: ALSConfig = ALSConfig()) -> None
    def fit(self, ratings: pd.DataFrame) -> Self              # user_id, movie_id, rating
    n_train_ratings: int; user_ids: np.ndarray; movie_ids: np.ndarray
    def has_user(self, user_id: int) -> bool
    def candidates(self, user_ids: Sequence[int], n: int = 100) -> pd.DataFrame        # сохранённые факторы, seen = обучающие оценки
    def candidates_for_ratings(self, ratings: pd.DataFrame, n: int = 100) -> pd.DataFrame  # fold-in (recalculate_user=True), seen = переданные оценки
    def save(self, path: Path) -> None; @classmethod def load(cls, path: Path) -> Self
```
Фильмы без факторов (нет в обучении) игнорируются при fold-in, но всё равно считаются seen (их и так нет в выдаче). Позиции с нечисловым скором отбрасываются, поэтому на маленьком каталоге кандидатов может быть меньше `n`. Неизвестный пользователь в `candidates` → `KeyError`.

- [ ] **Step 1: Тесты** (синтетика: 3 «вкуса» по жанрам, ~60 пользователей × ~25 оценок, детерминированный генератор в `tests/unit/synthetic.py`):
  `test_confidence_weights_linear_and_binary`, `test_unknown_weight_scheme_raises`, `test_candidates_schema_ranks_and_sorted_scores`, `test_candidates_exclude_training_items`, `test_candidates_capped_by_unseen_catalog` (n больше каталога → ровно |unseen| строк, все id валидны), `test_unknown_user_raises_key_error`, `test_fold_in_excludes_rated_and_ignores_unknown_movies`, `test_fold_in_close_to_fitted_factors_for_known_user` (пересечение топ-10 ≥ 0.7 в среднем), `test_fit_is_deterministic`, `test_save_load_roundtrip_same_candidates`, `test_als_prefers_users_taste` (у пользователя вкуса A в топ-10 больше всего фильмов жанра A).
- [ ] **Step 2: FAIL. Step 3: Реализовать. Step 4: зелёный.**
- [ ] **Step 5: Commit** `feat(models): ALS candidate generator with fold-in`.

### Task 4: Признаки (`features.py`)

**Files:**
- Create: `src/recsys/features.py`
- Test: `tests/unit/test_features.py`

**Interfaces:**
```python
GENRES: tuple[str, ...]                  # 18 жанров ML-1M в фиксированном порядке
GENRE_COLUMNS = [f"genre_{slug}" ...]
FEATURE_COLUMNS: list[str]               # als_score, als_rank, movie_n_ratings, movie_mean_rating, movie_year,
                                         # *GENRE_COLUMNS, user_n_ratings, user_mean_rating, user_age, user_gender,
                                         # user_occupation, genre_affinity_mean, genre_affinity_max
CATEGORICAL_FEATURES = ["user_occupation"]
def movie_features(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame   # index movie_id: n, mean, year, genre_*
def user_features(ratings: pd.DataFrame, movie_genres: pd.DataFrame,
                  users: pd.DataFrame | None = None) -> pd.DataFrame              # index user_id: n, mean, age, gender, occupation, share_*
def drop_demographics(user_feats: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame
def build_feature_frame(candidates: pd.DataFrame, movie_feats: pd.DataFrame,
                        user_feats: pd.DataFrame) -> pd.DataFrame                  # user_id, movie_id + FEATURE_COLUMNS, порядок строк как у candidates
```
Все функции чистые (вход — DataFrame, без глобального состояния); один и тот же код вызывается при обучении, оценке и в сервисе (fold-in путь передаёт оценки веб-пользователя и `users=None`).

- [ ] **Step 1: Тесты:** `test_movie_features_counts_means_year_genres` (ручной расчёт), `test_unrated_movie_zero_count_nan_mean`, `test_user_features_counts_means_demographics` (`M → 1.0`, `F → 0.0`), `test_user_without_profile_has_nan_demographics`, `test_genre_affinity_hand_computed` (пользователь: 2 Comedy, 1 Drama|Comedy, 1 Horror → share Comedy 0.75, Drama 0.25; кандидат Drama|Comedy → mean 0.5, max 0.75), `test_movie_without_known_genres_has_nan_affinity`, `test_single_user_features_match_batch` (train/serve parity), `test_feature_frame_columns_and_row_alignment`, `test_drop_demographics_deterministic_fraction`.
- [ ] **Step 2: FAIL. Step 3: Реализовать. Step 4: зелёный.**
- [ ] **Step 5: Commit** `feat: feature module shared by training and serving`.

### Task 5: Ранжировщик LightGBM

**Files:**
- Create: `src/recsys/models/ranker.py`
- Test: `tests/unit/test_ranker.py`

**Interfaces:**
```python
@dataclass(frozen=True)
class RankerConfig:
    n_estimators: int = 300; learning_rate: float = 0.05; num_leaves: int = 31
    min_child_samples: int = 50; colsample_bytree: float = 0.8; early_stopping_rounds: int = 30
    valid_fraction: float = 0.1; demographics_dropout: float = 0.1; random_state: int = 42; n_jobs: int = 0

def label_candidates(candidates: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray     # int8: 1, если пара в targets
def build_training_frame(candidates, targets, movie_feats, user_feats) -> pd.DataFrame  # фичи + label, группы подряд, без групп без позитивов
class LambdaRanker:
    def __init__(self, config: RankerConfig = RankerConfig()) -> None
    def fit(self, frame: pd.DataFrame) -> Self      # объектив lambdarank, eval_at=[10], early stopping на отложенных пользователях
    def predict(self, features: pd.DataFrame) -> np.ndarray
    best_iteration: int; def feature_importance(self) -> dict[str, float]   # gain
    def save(self, path: Path) -> None; @classmethod def load(cls, path: Path) -> Self
```

- [ ] **Step 1: Тесты:** `test_labels_mark_target_pairs_only`, `test_training_frame_drops_groups_without_positives`, `test_training_frame_groups_are_contiguous`, `test_ranker_learns_planted_signal` (метка зависит от одного признака → он получает наибольший gain, NDCG ранжировщика > случайного), `test_ranker_is_deterministic`, `test_ranker_save_load_same_predictions`, `test_ranker_handles_nan_demographics`.
- [ ] **Step 2: FAIL. Step 3: Реализовать. Step 4: зелёный.**
- [ ] **Step 5: Commit** `feat(models): LightGBM lambdarank re-ranker`.

### Task 6: Двухэтапный рекомендатор и бандл

**Files:**
- Create: `src/recsys/models/two_stage.py`
- Test: `tests/unit/test_two_stage.py`

**Interfaces:**
```python
class TwoStageRecommender:
    def __init__(self, als: ALSCandidateGenerator, movie_feats: pd.DataFrame, user_feats: pd.DataFrame,
                 ranker: LambdaRanker | None = None, n_candidates: int = 100) -> None
    variant: str                                     # "als_ranker" | "als_only"
    def without_ranker(self) -> "TwoStageRecommender"
    def recommend_known_users(self, user_ids: Sequence[int], k: int = 10) -> dict[int, list[int]]
    def recommend_for_ratings(self, ratings: pd.DataFrame, k: int = 10,
                              profiles: pd.DataFrame | None = None) -> pd.DataFrame   # user_id, rank, movie_id, score
    def recommend_new_user(self, ratings: Sequence[tuple[int, int]], k: int = 10,
                           profile: Mapping[str, object] | None = None) -> list[int]
    def save(self, path: Path) -> None; @classmethod def load(cls, path: Path) -> Self
```
Порядок: скор ранжировщика по убыванию, при равенстве — `als_rank` по возрастанию. Без ранжировщика — порядок ALS. `profile` = `{"gender": "F", "age": 25, "occupation": 4}`, отсутствующие ключи → NaN.

- [ ] **Step 1: Тесты:** `test_known_users_get_k_unseen_items`, `test_als_only_equals_top_k_als_candidates`, `test_ranker_changes_order_within_candidates`, `test_new_user_fold_in_excludes_rated`, `test_new_user_without_profile_works`, `test_recommend_for_ratings_output_schema`, `test_fold_in_and_batch_paths_build_identical_features`, `test_save_load_roundtrip_same_recommendations` (с ранжировщиком и без).
- [ ] **Step 2: FAIL. Step 3: Реализовать. Step 4: зелёный.**
- [ ] **Step 5: Commit** `feat(models): two-stage recommender bundle`.

### Task 7: Ядро обучения (без MLflow)

**Files:**
- Create: `src/recsys/pipelines/train.py`
- Test: `tests/unit/test_train_pipeline.py`

**Interfaces:**
```python
@dataclass(frozen=True)
class TrainConfig:
    als: ALSConfig = ALSConfig(); ranker: RankerConfig = RankerConfig(); n_candidates: int = 100; k: int = 10

@dataclass
class TrainResult:
    metrics: dict[str, dict[str, float]]      # popularity / als_only / als_ranker → evaluate(...)
    variant: str                              # выбранный для сервиса
    recommender: TwoStageRecommender          # уже без ранжировщика, если variant == "als_only"
    diagnostics: dict[str, float]             # n_ratings ALS_A и ALS_B, размеры выборки ранжировщика, recall@n кандидатов, best_iteration
    feature_importance: dict[str, float]

MODEL_NAMES = ("popularity", "als_only", "als_ranker")
def select_variant(metrics, k=10) -> str
def validate_als(split, n_items, als_config, k=10) -> dict[str, float]        # ALS на als_train → ranker_train
def train_and_evaluate(ratings, movies, users, config=TrainConfig()) -> TrainResult
```
- [ ] **Step 1: Тесты** (синтетика из Task 3, маленькие конфиги, `num_threads=1`, `n_jobs=1`):
  `test_train_and_evaluate_reports_three_models` (ключи и диапазон [0,1]), `test_als_beats_popularity_on_structured_synthetic`, `test_ranker_trained_on_als_train_only_and_eval_on_history` (через `diagnostics`), `test_registered_recommender_never_recommends_history`, `test_pipeline_is_deterministic`, `test_select_variant_prefers_als_only_when_ranker_not_better` (включая равенство), `test_validate_als_ignores_test_part` (порча `split.test` не меняет результат).
- [ ] **Step 2: FAIL. Step 3: Реализовать. Step 4: зелёный.**
- [ ] **Step 5: Commit** `feat: two-stage training pipeline with leakage-free protocol`.

### Task 8: MLflow: трекинг, pyfunc, реестр, alias, CLI

**Files:**
- Create: `src/recsys/tracking.py`, `src/recsys/models/pyfunc.py`; Modify: `src/recsys/pipelines/train.py` (CLI `main`)
- Test: `tests/unit/test_tracking.py`

**Interfaces:**
```python
DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"; EXPERIMENT_NAME = "movielens-recsys"
MODEL_NAME = "movielens-recommender"; CHAMPION_ALIAS = "champion"
def resolve_tracking_uri(uri: str | None = None) -> str            # аргумент > $MLFLOW_TRACKING_URI > default
def configure_mlflow(uri: str | None = None, experiment: str = EXPERIMENT_NAME) -> str   # experiment_id
def git_sha() -> str; def data_snapshot_hash(ratings: pd.DataFrame) -> str   # не зависит от порядка строк
def decide_promotion(challenger: Mapping[str, float], champion: Mapping[str, float] | None,
                     ndcg_tolerance: float = 0.005, max_coverage_drop: float = 0.10) -> tuple[bool, str]
def log_train_run(result: TrainResult, config: TrainConfig, tags: Mapping[str, str], run_name: str) -> str   # run_id
def register_and_promote(run_id: str, model_uri: str, result: TrainResult, snapshot: str) -> RegistrationOutcome
def load_champion(tracking_uri: str | None = None) -> TwoStageRecommender
class RecommenderPyfunc(mlflow.pyfunc.PythonModel)   # predict(ratings DataFrame, params={"k"}) → DataFrame
```
CLI: `uv run python -m recsys.pipelines.train --data-dir data/raw/ml-1m [--sweep] [--tracking-uri URI] [--no-register] [--factors ... --regularization ... --alpha ... --iterations ... --rating-weight ...]`. Печатает таблицу сравнения, run id, версию модели, решение по alias.

В MLflow (spec 4.3): params (ALS, LightGBM, сиды, n_candidates, k); metrics `{model}_{metric}` для трёх моделей + `ndcg_at_10/recall_at_10/coverage` выбранного варианта + диагностика; tags `git_sha`, `data_snapshot`, `n_users`, `n_ratings`, `web_share`, `variant`, `run_type`; artifacts: `comparison.md`, `comparison.json`, `movie_features.parquet`, `feature_importance.json`, pyfunc `model`. Sweep-запуски — дочерние runs с `run_type=sweep` и метриками `val_*`.

- [ ] **Step 1: Тесты** (временная sqlite-база в `tmp_path`, без сервера): `test_resolve_tracking_uri_precedence`, `test_data_snapshot_hash_order_invariant_and_content_sensitive`, `test_decide_promotion_rules` (parametrize: нет чемпиона; лучше; в пределах допуска; хуже допуска; падение coverage > 10%), `test_log_register_promote_and_load_champion` (первая версия → `champion`, загруженный чемпион рекомендует k фильмов известному и новому пользователю), `test_weaker_second_version_does_not_take_alias`, `test_pyfunc_predict_schema`, `test_cli_end_to_end_on_tiny_raw_dir`.
- [ ] **Step 2: FAIL. Step 3: Реализовать. Step 4: зелёный, время набора проверить (`--durations=10`).**
- [ ] **Step 5: Commit** `feat: MLflow tracking, pyfunc model and registry with champion alias`.

### Task 9: Реальные данные: sweep, дефолты, финальный прогон

- [ ] **Step 1:** `uv run python -m recsys.pipelines.train --data-dir data/raw/ml-1m --sweep` — sweep по сетке ALS (factors × regularization × alpha × rating_weight, ~10 конфигов), оценка на валидационном протоколе, лучший конфиг идёт в основной прогон. Ожидается: время — минуты, ALS-only заметно выше popularity (0.0339).
- [ ] **Step 2:** Перенести лучший конфиг в дефолты `ALSConfig` (коммит `chore: ALS defaults from validation sweep`), повторить прогон без `--sweep`, записать точную таблицу и id run/версии.
- [ ] **Step 3:** Финальная проверка: `uv lock --check`, `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`.

## Критерии готовности этапа 2

- [ ] Таблица popularity → ALS-only → ALS+ranker на одном test-сплите напечатана CLI и лежит в MLflow (`comparison.md`)
- [ ] ALS-only обгоняет popularity; про ранжировщик честно записано, обгоняет ли он ALS-only, и какой вариант зарегистрирован
- [ ] `models:/movielens-recommender@champion` загружается через `load_champion()` и рекомендует известному и новому пользователю
- [ ] Набор тестов зелёный и быстрый, ruff чистый, `uv lock --check` проходит
