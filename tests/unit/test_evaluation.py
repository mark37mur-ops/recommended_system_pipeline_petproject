import pandas as pd
import pytest

from recsys.data.split import Split
from recsys.evaluation import (
    ProtocolViolation,
    compare_models,
    evaluate_recommendations,
    format_comparison,
    make_context,
    make_test_context,
    make_validation_context,
)

COLS = ["user_id", "movie_id", "rating", "timestamp"]


def _frame(rows: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame([(u, m, 4, 0) for u, m in rows], columns=COLS)


@pytest.fixture
def split() -> Split:
    return Split(
        als_train=_frame([(1, 1), (1, 2), (2, 1), (3, 5)]),
        ranker_train=_frame([(1, 3), (2, 2)]),
        test=_frame([(2, 4), (1, 4), (1, 6)]),
    )


def _pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(df["user_id"], df["movie_id"], strict=True))


def test_test_context_history_is_als_plus_ranker_and_target_is_test(split):
    ctx = make_test_context(split, n_items=10)
    assert _pairs(ctx.history) == _pairs(split.als_train) | _pairs(split.ranker_train)
    assert _pairs(ctx.target) == _pairs(split.test)
    assert ctx.k == 10 and ctx.n_items == 10


def test_validation_context_uses_als_train_and_ranker_train(split):
    ctx = make_validation_context(split, n_items=10, k=5)
    assert _pairs(ctx.history) == _pairs(split.als_train)
    assert _pairs(ctx.target) == _pairs(split.ranker_train)
    assert ctx.k == 5


def test_context_users_sorted_and_seen_from_history(split):
    ctx = make_test_context(split, n_items=10)
    assert ctx.users == [1, 2]
    assert ctx.seen == {1: {1, 2, 3}, 2: {1, 2}, 3: {5}}


def test_evaluate_recommendations_rejects_seen_item(split):
    ctx = make_test_context(split, n_items=10)
    with pytest.raises(ProtocolViolation, match="user 1"):
        evaluate_recommendations({1: [4, 2], 2: [4]}, ctx)


def test_seen_beyond_k_is_not_a_violation(split):
    ctx = make_context(split.als_train, split.test, n_items=10, k=1)
    result = evaluate_recommendations({1: [4, 1], 2: [4, 1]}, ctx)
    assert result["ndcg_at_1"] == pytest.approx(1.0)


def test_evaluate_recommendations_matches_metrics_evaluate(split):
    ctx = make_test_context(split, n_items=10)
    result = evaluate_recommendations({1: [4, 7], 2: [5]}, ctx)
    # user 1: hit at rank 1 of 2 relevant -> ndcg = 1 / (1 + 1/log2(3)); user 2: 0
    assert result["ndcg_at_10"] == pytest.approx(0.5 * 1 / (1 + 1 / 1.5849625007211562))
    assert result["recall_at_10"] == pytest.approx(0.25)
    assert result["coverage"] == pytest.approx(0.3)


def test_compare_models_passes_same_context_to_every_model(split):
    ctx = make_test_context(split, n_items=10)
    received = []

    def model(c):
        received.append(c)
        return {u: [9] for u in c.users}

    results = compare_models({"a": model, "b": model}, ctx)
    assert list(results) == ["a", "b"]
    assert received[0] is ctx and received[1] is ctx
    assert results["a"] == results["b"]


def test_format_comparison_lists_models_in_order():
    table = format_comparison(
        {
            "popularity": {"ndcg_at_10": 0.1, "recall_at_10": 0.2, "coverage": 0.3},
            "als_only": {"ndcg_at_10": 0.4, "recall_at_10": 0.5, "coverage": 0.6},
        }
    )
    lines = table.splitlines()
    assert "ndcg_at_10" in lines[0] and "coverage" in lines[0]
    assert lines.index(next(x for x in lines if "popularity" in x)) < lines.index(
        next(x for x in lines if "als_only" in x)
    )
    assert "0.4000" in table
