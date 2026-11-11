"""Data drift report of `retrain_daily` (spec 5.3).

Two parts, both written by the `drift_report` task:

1. **Summary** (stage 4, dependency-free): rating shares, counts and web share, compared with
   the previous run and web vs MovieLens (`summarize_ratings`, `summary_report`).
2. **Evidently** (stage 5): two reports with the same machinery (`drift_section`):
   - *ratings*: reference = MovieLens ratings, current = web ratings of the 7 days before the
     snapshot; features `rating`, `genre`, `release_year`, `user_activity`;
   - *recommendations* (prediction drift): reference = the movies MovieLens users rated,
     current = the movies the model recommended (`path=full` impressions) in the same 7 days;
     features `popularity`, `genre`, `release_year`.
   Each yields an HTML report (MLflow artifact) and a drift share: the fraction of features whose
   distance to the reference exceeds the threshold. `drift_share` is the ratings one.

Drift is a signal to look closer, not an alarm: the 2026 web audience differs from the 2000
MovieLens one by construction. Nothing here fails the pipeline: too little current data gives
"insufficient_data" (share None), an Evidently failure gives "error" (share None).

Evidently is imported lazily (`_import_evidently`): it is in the `drift` dependency group, which
the Airflow image installs and the API image does not.
"""

import html
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SUMMARY_ENGINE = "summary-v1"
DRIFT_ENGINE = "evidently"
RATING_VALUES = (1, 2, 3, 4, 5)

CURRENT_WINDOW = timedelta(days=7)
# Below this many current rows a distance between distributions is mostly noise.
MIN_CURRENT_ROWS = 50
# MovieLens has 1M ratings; a fixed random sample keeps the report fast and a few MB in size.
REFERENCE_SIZE = 50_000
SEED = 0
# Fixed methods (Evidently's own defaults for a reference of >1000 rows). Left to itself,
# Evidently switches to p-value tests on small samples and the meaning of the share would
# depend on data size. Distance >= threshold -> the feature drifted.
NUM_METHOD, CAT_METHOD, THRESHOLD = "wasserstein", "jensenshannon", 0.1

RATING_FEATURES = {
    "rating": "categorical",
    "genre": "categorical",
    "release_year": "numerical",
    "user_activity": "numerical",
}
RECOMMENDATION_FEATURES = {
    "popularity": "numerical",
    "genre": "categorical",
    "release_year": "numerical",
}
NO_GENRE = "(no genre)"


# ----- stage-4 summary -----------------------------------------------------------------------


def _share(ratings: pd.Series) -> dict[str, float] | None:
    if ratings.empty:
        return None
    counts = ratings.value_counts()
    total = float(len(ratings))
    return {str(v): float(counts.get(v, 0)) / total for v in RATING_VALUES}


def summarize_ratings(ratings: pd.DataFrame) -> dict:
    """Small JSON-able description of a ratings snapshot."""
    if "source" in ratings.columns:
        is_web = ratings["source"] == "web"
    else:
        is_web = pd.Series(False, index=ratings.index)
    web, movielens = ratings[is_web], ratings[~is_web]
    n = len(ratings)
    return {
        "n_ratings": int(n),
        "n_web_ratings": int(len(web)),
        "web_share": float(len(web) / n) if n else 0.0,
        "n_users": int(ratings["user_id"].nunique()),
        "n_web_users": int(web["user_id"].nunique()),
        "n_movies": int(ratings["movie_id"].nunique()),
        "rating_share": _share(ratings["rating"]),
        "movielens_rating_share": _share(movielens["rating"]),
        "web_rating_share": _share(web["rating"]),
    }


def total_variation(p: Mapping[str, float] | None, q: Mapping[str, float] | None) -> float | None:
    """0 = same distribution, 1 = disjoint."""
    if p is None or q is None:
        return None
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def summary_report(current: Mapping, previous: Mapping | None) -> dict:
    """Compare this run's snapshot with the previous run's and web with MovieLens."""
    web_vs_ml = total_variation(current["web_rating_share"], current["movielens_rating_share"])
    deltas = None
    if previous is not None:
        prev_n = previous["n_ratings"]
        deltas = {
            "n_ratings_change": (current["n_ratings"] - prev_n) / prev_n if prev_n else None,
            "web_share_change": current["web_share"] - previous["web_share"],
            "rating_share_tvd": total_variation(current["rating_share"], previous["rating_share"]),
        }
    metrics = {
        "drift_n_ratings": float(current["n_ratings"]),
        "drift_web_share": float(current["web_share"]),
    }
    if web_vs_ml is not None:
        metrics["drift_web_vs_movielens_tvd"] = float(web_vs_ml)
    for key, value in (deltas or {}).items():
        if value is not None:
            metrics[f"drift_{key}"] = float(value)
    return {
        "engine": SUMMARY_ENGINE,
        "current": dict(current),
        "previous": None if previous is None else dict(previous),
        "deltas": deltas,
        "web_vs_movielens_tvd": web_vs_ml,
        "metrics": metrics,
    }


# ----- feature frames ------------------------------------------------------------------------


def _movie_table(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Per movie: release year, genre list and popularity (ratings in this snapshot)."""
    table = pd.DataFrame(
        {
            # .to_numpy(): positional, the frame is indexed by movie_id, not by row number
            "release_year": pd.to_numeric(movies["year"], errors="coerce")
            .astype("float64")
            .to_numpy(),
            "genres": [
                list(g) if g is not None and len(g) else [NO_GENRE] for g in movies["genres"]
            ],
        },
        index=movies["movie_id"].astype("int64"),
    )
    table["popularity"] = ratings.groupby("movie_id").size().reindex(table.index, fill_value=0)
    return table


def _pick_genre(genre_lists: Sequence[list[str]], seed: int) -> list[str]:
    """One genre per row, uniformly among the movie's genres: a movie with n genres adds 1/n to
    each of them in expectation, the usual way to describe a genre mix of multi-genre items."""
    rng = np.random.default_rng(seed)
    return [str(g[int(rng.integers(len(g)))]) for g in genre_lists]


def _sample(frame: pd.DataFrame, size: int | None) -> pd.DataFrame:
    if size is not None and len(frame) > size:
        frame = frame.sample(n=size, random_state=SEED)
    return frame


def _is_web(ratings: pd.DataFrame) -> pd.Series:
    if "source" not in ratings:
        return pd.Series(False, index=ratings.index)
    return ratings["source"] == "web"


def rating_frames(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    until: datetime,
    window: timedelta = CURRENT_WINDOW,
    reference_size: int | None = REFERENCE_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(reference, current) rows of RATING_FEATURES: MovieLens ratings (sampled) vs web
    ratings with `until - window < timestamp <= until` (`until` = snapshot time, so a retry
    of the task sees the same window)."""
    table = _movie_table(movies, ratings)
    activity = ratings.groupby("user_id").size()
    end, start = until.timestamp(), (until - window).timestamp()
    web = _is_web(ratings)
    recent = web & (ratings["timestamp"] > start) & (ratings["timestamp"] <= end)

    def features(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
        movie = table.reindex(rows["movie_id"].astype("int64"))
        return pd.DataFrame(
            {
                "rating": rows["rating"].astype(int).astype(str).to_numpy(),
                "genre": _pick_genre(list(movie["genres"]), seed),
                "release_year": movie["release_year"].to_numpy(),
                "user_activity": rows["user_id"].map(activity).astype("int64").to_numpy(),
            }
        )

    reference = features(_sample(ratings[~web], reference_size), SEED)
    current = features(ratings[recent], SEED + 1)
    return reference, current


def recommendation_frames(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    shown_movie_ids: Sequence[int],
    reference_size: int | None = REFERENCE_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(reference, current) rows of RECOMMENDATION_FEATURES: one row per MovieLens rating
    (what that audience watched) vs one row per impression (what the model recommended)."""
    table = _movie_table(movies, ratings)

    def features(movie_ids: Sequence[int], seed: int) -> pd.DataFrame:
        movie = table.reindex(pd.Index(np.asarray(movie_ids, dtype="int64")))
        genres = [g if isinstance(g, list) else [NO_GENRE] for g in movie["genres"]]
        return pd.DataFrame(
            {
                "popularity": movie["popularity"].fillna(0).astype("int64").to_numpy(),
                "genre": _pick_genre(genres, seed),
                "release_year": movie["release_year"].to_numpy(),
            }
        )

    reference_ids = _sample(ratings[~_is_web(ratings)], reference_size)["movie_id"]
    return features(list(reference_ids), SEED + 2), features(list(shown_movie_ids), SEED + 3)


# ----- Evidently -----------------------------------------------------------------------------


def _import_evidently():
    """Evidently, made importable under pandas 3.

    Evidently 0.7.23 imports `pandas.core.dtypes.common.infer_dtype_from_object`, which pandas 3
    removed; only its legacy TestColumnsType uses it, and we never call that. The stand-in does
    what the pandas 2 function did for dtype-like input (stage-5 plan, Decision 2)."""
    import pandas.core.dtypes.common as pandas_common

    if not hasattr(pandas_common, "infer_dtype_from_object"):
        pandas_common.infer_dtype_from_object = lambda dtype: pd.api.types.pandas_dtype(dtype).type

    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    return DataDefinition, Dataset, Report, DataDriftPreset


def _page(title: str, body: str) -> str:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"</head><body style='font-family: sans-serif; max-width: 48em; margin: 2em auto'>"
        f"<h1>{html.escape(title)}</h1>{body}</body></html>"
    )


def _parse_snapshot(result: dict) -> tuple[float | None, dict[str, dict]]:
    """(share of drifted columns, per-column method / threshold / score / drifted)."""
    share, columns = None, {}
    for metric in result.get("metrics", []):
        config, value = metric.get("config", {}), metric.get("value")
        kind = str(config.get("type", ""))
        if kind.endswith(":DriftedColumnsCount"):
            share = float(value["share"])
        elif kind.endswith(":ValueDrift"):
            score, threshold = float(value), float(config["threshold"])
            columns[config["column"]] = {
                "method": config["method"],
                "threshold": threshold,
                "score": score,
                "drifted": bool(score >= threshold),
            }
    return share, columns


def drift_section(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Mapping[str, str],
    html_path: Path,
    title: str = "Drift report",
    min_current: int = MIN_CURRENT_ROWS,
) -> dict:
    """Run one Evidently drift report and write its HTML to `html_path`. Never raises for
    data reasons: returns {"status": ok | insufficient_data | error, "drift_share", ...}."""
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "status": "ok",
        "drift_share": None,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "min_current": int(min_current),
        "columns": {},
        "error": None,
    }
    if len(current) < min_current or len(reference) < min_current:
        out["status"] = "insufficient_data"
        html_path.write_text(
            _page(
                title,
                f"<p><b>Insufficient data.</b> Current: {len(current)} rows, reference: "
                f"{len(reference)} rows; at least {min_current} of each are needed for a "
                "drift verdict. drift_share is not computed (NULL), which means "
                "&laquo;cannot say&raquo;, not &laquo;no drift&raquo;.</p>",
            ),
            encoding="utf-8",
        )
        return out
    try:
        DataDefinition, Dataset, Report, DataDriftPreset = _import_evidently()
        definition = DataDefinition(
            numerical_columns=[c for c, kind in features.items() if kind == "numerical"],
            categorical_columns=[c for c, kind in features.items() if kind == "categorical"],
        )
        columns = list(features)
        preset = DataDriftPreset(
            columns=columns,
            num_method=NUM_METHOD,
            cat_method=CAT_METHOD,
            num_threshold=THRESHOLD,
            cat_threshold=THRESHOLD,
        )
        snapshot = Report([preset]).run(
            current_data=Dataset.from_pandas(current[columns], data_definition=definition),
            reference_data=Dataset.from_pandas(reference[columns], data_definition=definition),
        )
        snapshot.save_html(str(html_path))
        out["drift_share"], out["columns"] = _parse_snapshot(snapshot.dict())
    except Exception as exc:  # noqa: BLE001 - a broken report must not stop retraining
        log.exception("Evidently report %r failed", title)
        out.update(
            status="error", drift_share=None, columns={}, error=f"{type(exc).__name__}: {exc}"
        )
        html_path.write_text(
            _page(title, f"<p><b>Report failed:</b> {html.escape(out['error'])}</p>"),
            encoding="utf-8",
        )
    return out


def build_drift_sections(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    shown_movie_ids: Sequence[int],
    until: datetime,
    out_dir: Path,
    window: timedelta = CURRENT_WINDOW,
) -> dict:
    """Both Evidently sections; HTML files `ratings.html` and `recommendations.html` in
    `out_dir`. `shown_movie_ids`: one entry per model impression in the window."""
    out_dir = Path(out_dir)
    reference, current = rating_frames(ratings, movies, until, window)
    rating_part = drift_section(
        reference,
        current,
        RATING_FEATURES,
        out_dir / "ratings.html",
        title="Data drift: MovieLens ratings vs web ratings of the last 7 days",
    )
    reference, current = recommendation_frames(ratings, movies, shown_movie_ids)
    recommendation_part = drift_section(
        reference,
        current,
        RECOMMENDATION_FEATURES,
        out_dir / "recommendations.html",
        title="Prediction drift: movies MovieLens users rated vs movies we recommended",
    )
    return {
        "engine": DRIFT_ENGINE,
        "window_start": (until - window).isoformat(),
        "window_end": until.isoformat(),
        "ratings": rating_part,
        "recommendations": recommendation_part,
        "drift_share": rating_part["drift_share"],
        "prediction_drift_share": recommendation_part["drift_share"],
    }
