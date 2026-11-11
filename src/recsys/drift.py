"""Data drift report of `retrain_daily` (spec 5.3).

Stage 4 ships a lightweight, dependency-free summary: rating distribution, number of ratings,
share of web ratings, compared with the previous run and web vs MovieLens. `build_drift_report`
is the single hook that stage 5 replaces with an Evidently report (HTML artifact +
`drift_share`); its input (two `summarize_ratings` dicts) and output keys (`engine`,
`metrics`, ...) are what the pipeline relies on.

Drift is a signal to look closer, not an alarm: nothing here fails the pipeline.
"""

from collections.abc import Mapping

import pandas as pd

DRIFT_ENGINE = "summary-v1"  # stage 5: "evidently"
RATING_VALUES = (1, 2, 3, 4, 5)


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


def build_drift_report(current: Mapping, previous: Mapping | None) -> dict:
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
        "engine": DRIFT_ENGINE,
        "current": dict(current),
        "previous": None if previous is None else dict(previous),
        "deltas": deltas,
        "web_vs_movielens_tvd": web_vs_ml,
        "metrics": metrics,
    }
