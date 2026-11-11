from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd


def _top_unique(recommended: Sequence[int], k: int) -> list[int]:
    """First k recommendations, with repeats collapsed to their first occurrence."""
    return list(dict.fromkeys(recommended[:k]))


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int = 10) -> float:
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
    if not relevant:
        return 0.0
    hits = sum(1 for item in _top_unique(recommended, k) if item in relevant)
    return hits / len(relevant)


def coverage(recommendations: Mapping[int, Sequence[int]], n_items: int) -> float:
    shown = {item for recs in recommendations.values() for item in recs}
    return len(shown) / n_items


def evaluate(
    recommendations: Mapping[int, Sequence[int]],
    test: pd.DataFrame,
    n_items: int,
    k: int = 10,
) -> dict[str, float]:
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
        f"ndcg@{k}": sum(ndcgs) / n_users if n_users else 0.0,
        f"recall@{k}": sum(recalls) / n_users if n_users else 0.0,
        "coverage": coverage(shown_recs, n_items),
    }
