from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd


def _top_unique(recommended: Sequence[int], k: int) -> list[int]:
    """First k recommendations, with repeats collapsed to their first occurrence."""
    return list(dict.fromkeys(recommended[:k]))


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """NDCG@k with binary relevance.

    `recommended` is truncated to k first, then de-duplicated (first occurrence counts).
    Returns 0.0 when `relevant` is empty.
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(_top_unique(recommended, k))
        if item in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """Recall@k: hits among the top-k (truncated first, then de-duplicated) over |relevant|."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in _top_unique(recommended, k) if item in relevant)
    return hits / len(relevant)


def coverage(recommendations: Mapping[int, Sequence[int]], n_items: int) -> float:
    """Share of the catalogue (n_items) that appears in any user's recommendations."""
    shown = {item for recs in recommendations.values() for item in recs}
    return len(shown) / n_items


def evaluate(
    recommendations: Mapping[int, Sequence[int]],
    test: pd.DataFrame,
    n_items: int,
    k: int = 10,
) -> dict[str, float]:
    """Mean NDCG@k and Recall@k over users present in `test`, plus catalogue coverage.

    Conventions: relevance is binary, every item a user has in `test` is relevant; each
    user's recommendations are truncated to k first and then de-duplicated; a user with no
    recommendations scores 0. Coverage is the number of unique recommended items
    (truncated to k) over users present in `test`, divided by n_items.

    Keys are MLflow-compatible (no "@"): ndcg_at_{k}, recall_at_{k}, coverage.
    """
    relevant_by_user = test.groupby("user_id")["movie_id"].agg(set).to_dict()

    ndcgs: list[float] = []
    recalls: list[float] = []
    for user, relevant in relevant_by_user.items():
        recs = recommendations.get(user, [])
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))

    n_users = len(relevant_by_user)
    shown_recs = {
        user: recommendations[user][:k] for user in relevant_by_user if user in recommendations
    }
    return {
        f"ndcg_at_{k}": sum(ndcgs) / n_users if n_users else 0.0,
        f"recall_at_{k}": sum(recalls) / n_users if n_users else 0.0,
        "coverage": coverage(shown_recs, n_items),
    }
