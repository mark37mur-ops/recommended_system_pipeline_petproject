"""Drift report (spec 5.3): the stage-4 summary and the Evidently sections."""

import pandas as pd
import pytest

from recsys.drift import SUMMARY_ENGINE, summarize_ratings, summary_report


def _ratings(rows):
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp", "source"])


@pytest.fixture
def current():
    return _ratings(
        [
            (1, 10, 5, 1, "movielens"),
            (1, 11, 4, 2, "movielens"),
            (2, 10, 3, 3, "movielens"),
            (1_000_000, 10, 1, 4, "web"),
        ]
    )


def test_summary_counts_and_shares(current):
    s = summarize_ratings(current)
    assert s["n_ratings"] == 4
    assert s["n_web_ratings"] == 1
    assert s["web_share"] == pytest.approx(0.25)
    assert s["n_users"] == 3
    assert s["n_web_users"] == 1
    assert s["rating_share"] == {"1": 0.25, "2": 0.0, "3": 0.25, "4": 0.25, "5": 0.25}
    assert s["web_rating_share"] == {"1": 1.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0}
    assert s["movielens_rating_share"]["5"] == pytest.approx(1 / 3)


def test_summary_without_source_column_counts_everything_as_movielens(current):
    s = summarize_ratings(current.drop(columns="source"))
    assert s["n_web_ratings"] == 0 and s["web_share"] == 0.0
    assert s["web_rating_share"] is None


def test_report_without_previous_run_has_no_deltas(current):
    report = summary_report(summarize_ratings(current), None)
    assert report["engine"] == SUMMARY_ENGINE
    assert report["previous"] is None
    assert report["deltas"] is None
    # web vs MovieLens (spec 5.3: reference = MovieLens, current = web)
    assert report["web_vs_movielens_tvd"] == pytest.approx(1.0)


def test_report_compares_with_previous_run(current):
    previous = summarize_ratings(current.iloc[:3])  # no web ratings yet
    report = summary_report(summarize_ratings(current), previous)
    deltas = report["deltas"]
    assert deltas["n_ratings_change"] == pytest.approx(1 / 3)
    assert deltas["web_share_change"] == pytest.approx(0.25)
    # previous: 3,4,5 one third each; current: 1,3,4,5 a quarter each -> TVD = 0.25
    assert deltas["rating_share_tvd"] == pytest.approx(0.25)


def test_report_metrics_are_flat_numbers(current):
    report = summary_report(summarize_ratings(current), summarize_ratings(current))
    metrics = report["metrics"]
    assert metrics["drift_rating_share_tvd"] == 0.0
    assert metrics["drift_web_share"] == pytest.approx(0.25)
    assert all(isinstance(v, float) for v in metrics.values())
