"""Evidently drift sections of drift_report (spec 5.3): features, windows, thresholds, and the
"too little data" / "Evidently failed" paths that must never fail the pipeline."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from synthetic import make_movies, make_ratings

from recsys import drift
from recsys.db.schema import WEB_USER_ID_START
from recsys.drift import (
    RATING_FEATURES,
    RECOMMENDATION_FEATURES,
    build_drift_sections,
    drift_section,
    rating_frames,
    recommendation_frames,
)

UNTIL = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
NOW_TS = int(UNTIL.timestamp())


def web(rows):
    """(user offset, movie, rating, seconds before UNTIL) -> web rating rows."""
    frame = pd.DataFrame(
        [(WEB_USER_ID_START + u, m, r, NOW_TS - ago) for u, m, r, ago in rows],
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )
    return frame.assign(source="web")


def snapshot(web_rows):
    ml = make_ratings().assign(source="movielens")
    return pd.concat([ml, web(web_rows)], ignore_index=True)


def comedy_haters(n_users=10, per_user=10):
    """Web visitors who rate only comedies, all with 1 star, today: obviously not MovieLens."""
    comedies = [m for m in range(1, 91) if m % 3 == 0]
    return [(u, comedies[i], 1, 3600) for u in range(n_users) for i in range(per_user)]


# ----- feature frames ------------------------------------------------------------------------


def test_rating_frames_reference_is_movielens_current_is_last_7_days_of_web():
    rows = [
        (0, 3, 5, 60),  # a minute ago
        (0, 6, 4, 7 * 86400 - 1),  # just inside the window
        (1, 9, 2, 7 * 86400 + 1),  # just outside
        (1, 12, 3, -60),  # after the snapshot moment (cannot happen, but not "current")
    ]
    ratings = snapshot(rows)
    reference, current = rating_frames(ratings, make_movies(), until=UNTIL)
    assert len(reference) == len(make_ratings())
    assert list(current["rating"]) == ["5", "4"]
    assert list(reference.columns) == list(current.columns) == list(RATING_FEATURES)
    # user_activity = number of ratings of the rating's author in the whole snapshot
    assert list(current["user_activity"]) == [2, 2]
    assert set(reference["user_activity"]) == {25}  # every synthetic user rated 25 movies
    assert list(current["release_year"]) == [1983.0, 1986.0]  # synthetic: 1980 + id % 20


def test_genre_is_one_of_the_movies_genres_and_reproducible():
    ratings, movies = snapshot(comedy_haters()), make_movies()
    first, _ = rating_frames(ratings, movies, until=UNTIL)
    again, _ = rating_frames(ratings, movies, until=UNTIL)
    pd.testing.assert_frame_equal(first, again)
    genres = dict(zip(movies["movie_id"], movies["genres"], strict=True))
    ml = make_ratings()
    for movie_id, genre in zip(ml["movie_id"], first["genre"], strict=True):
        assert genre in genres[movie_id]
    # multi-genre movies (every 5th has Romance too) contribute both genres
    assert {"Romance", "Comedy", "Drama", "Horror"} <= set(first["genre"])


def test_reference_is_a_fixed_size_sample():
    ratings = snapshot([])
    a, _ = rating_frames(ratings, make_movies(), until=UNTIL, reference_size=100)
    b, _ = rating_frames(ratings, make_movies(), until=UNTIL, reference_size=100)
    assert len(a) == 100
    pd.testing.assert_frame_equal(a, b)


def test_recommendation_frames_compare_shown_movies_with_what_movielens_rated():
    ratings, movies = snapshot([]), make_movies()
    shown = [3, 3, 6]  # one entry per impression
    reference, current = recommendation_frames(ratings, movies, shown)
    assert list(reference.columns) == list(current.columns) == list(RECOMMENDATION_FEATURES)
    counts = ratings.groupby("movie_id").size()
    assert list(current["popularity"]) == [counts[3], counts[3], counts[6]]
    assert len(reference) == len(make_ratings())


# ----- one Evidently section -----------------------------------------------------------------


def numeric_frame(n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "rating": rng.choice(["1", "2", "3", "4", "5"], size=n),
            "genre": rng.choice(["Comedy", "Drama", "Horror"], size=n),
            "release_year": rng.normal(1990 + shift, 5, size=n),
            "user_activity": rng.poisson(50 + shift, size=n).astype("int64"),
        }
    )


def test_same_distribution_has_no_drift(tmp_path):
    out = drift_section(
        numeric_frame(4000, seed=1),
        numeric_frame(2000, seed=2),
        RATING_FEATURES,
        tmp_path / "s.html",
    )
    assert out["status"] == "ok"
    assert out["drift_share"] == 0.0
    assert set(out["columns"]) == set(RATING_FEATURES)
    for column in out["columns"].values():
        assert column["drifted"] is False and column["score"] < column["threshold"] == 0.1
    assert (tmp_path / "s.html").stat().st_size > 10_000  # Evidently's interactive report


def test_obvious_drift_is_detected(tmp_path):
    reference, current = rating_frames(snapshot(comedy_haters()), make_movies(), until=UNTIL)
    out = drift_section(reference, current, RATING_FEATURES, tmp_path / "ratings.html")
    assert out["status"] == "ok" and out["n_current"] == 100
    assert out["columns"]["rating"]["drifted"] is True  # all 1 star vs mostly 4-5
    assert out["columns"]["genre"]["drifted"] is True  # comedies only
    assert out["columns"]["user_activity"]["drifted"] is True  # 10 vs 25 ratings per user
    assert out["columns"]["rating"]["method"] == "jensenshannon"  # fixed, not size-dependent
    assert out["columns"]["user_activity"]["method"] == "wasserstein"
    drifted = sum(c["drifted"] for c in out["columns"].values())
    assert out["drift_share"] == pytest.approx(drifted / len(RATING_FEATURES))
    assert out["drift_share"] >= 0.75


def test_too_little_current_data_is_not_a_verdict(tmp_path, monkeypatch):
    def no_evidently():
        raise AssertionError("Evidently must not run on 10 rows")

    monkeypatch.setattr(drift, "_import_evidently", no_evidently)
    out = drift_section(
        numeric_frame(4000), numeric_frame(10), RATING_FEATURES, tmp_path / "s.html"
    )
    assert out["status"] == "insufficient_data"
    assert out["drift_share"] is None and out["columns"] == {}
    assert out["n_current"] == 10 and out["min_current"] == drift.MIN_CURRENT_ROWS
    html = (tmp_path / "s.html").read_text(encoding="utf-8")
    assert "Insufficient data" in html and "10" in html


def test_evidently_failure_is_reported_not_raised(tmp_path, monkeypatch):
    def broken():
        raise RuntimeError("evidently exploded")

    monkeypatch.setattr(drift, "_import_evidently", broken)
    out = drift_section(
        numeric_frame(4000), numeric_frame(500), RATING_FEATURES, tmp_path / "s.html"
    )
    assert out["status"] == "error" and out["drift_share"] is None
    assert "evidently exploded" in out["error"]
    assert "evidently exploded" in (tmp_path / "s.html").read_text(encoding="utf-8")


# ----- both sections ---------------------------------------------------------------------------


def test_build_sections_without_web_data_or_impressions(tmp_path):
    sections = build_drift_sections(snapshot([]), make_movies(), [], until=UNTIL, out_dir=tmp_path)
    assert sections["ratings"]["status"] == "insufficient_data"
    assert sections["recommendations"]["status"] == "insufficient_data"
    assert sections["drift_share"] is None and sections["prediction_drift_share"] is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ratings.html", "recommendations.html"]
    assert sections["window_start"] == (UNTIL - timedelta(days=7)).isoformat()


def test_build_sections_with_drifted_web_ratings_and_popular_recommendations(tmp_path):
    shown = [m for m in range(1, 11)] * 10  # 100 impressions of the 10 lowest ids
    sections = build_drift_sections(
        snapshot(comedy_haters()), make_movies(), shown, until=UNTIL, out_dir=tmp_path
    )
    assert sections["ratings"]["status"] == "ok"
    assert sections["drift_share"] == sections["ratings"]["drift_share"] >= 0.75
    assert sections["recommendations"]["status"] == "ok"
    assert sections["prediction_drift_share"] == sections["recommendations"]["drift_share"]
