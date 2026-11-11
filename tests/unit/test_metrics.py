import math

import pandas as pd
import pytest

from recsys.metrics import coverage, evaluate, ndcg_at_k, recall_at_k


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
    assert res["ndcg_at_10"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
    assert res["recall_at_10"] == pytest.approx(0.5)
    assert res["coverage"] == pytest.approx(0.01)
