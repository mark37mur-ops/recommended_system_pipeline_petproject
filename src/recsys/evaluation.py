"""One evaluation protocol shared by every model (popularity, ALS-only, ALS + ranker).

A model is a function of an `EvalContext`: it may look at `ctx.history` only and must return
recommendations for `ctx.users`. Every model in a comparison receives the very same context,
and `evaluate_recommendations` refuses lists that contain an already-seen item in the top-k.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from recsys.data.split import Split
from recsys.metrics import evaluate


class ProtocolViolation(Exception):
    """A model broke the evaluation protocol (e.g. recommended an item from the history)."""


@dataclass(frozen=True)
class EvalContext:
    history: pd.DataFrame
    target: pd.DataFrame
    seen: dict[int, set[int]]
    users: list[int]
    n_items: int
    k: int


RecommendFn = Callable[[EvalContext], Mapping[int, Sequence[int]]]


def seen_by_user(ratings: pd.DataFrame) -> dict[int, set[int]]:
    return {
        int(user): {int(m) for m in group["movie_id"]} for user, group in ratings.groupby("user_id")
    }


def build_history(split: Split) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    """History visible to the model (als_train + ranker_train) and per-user seen movie sets."""
    history = pd.concat([split.als_train, split.ranker_train], ignore_index=True)
    return history, seen_by_user(history)


def make_context(
    history: pd.DataFrame, target: pd.DataFrame, n_items: int, k: int = 10
) -> EvalContext:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return EvalContext(
        history=history,
        target=target,
        seen=seen_by_user(history),
        users=sorted(int(u) for u in target["user_id"].unique()),
        n_items=n_items,
        k=k,
    )


def make_test_context(split: Split, n_items: int, k: int = 10) -> EvalContext:
    """Final evaluation: models see als_train + ranker_train, are scored on test."""
    history, _ = build_history(split)
    return make_context(history, split.test, n_items, k)


def make_validation_context(split: Split, n_items: int, k: int = 10) -> EvalContext:
    """Model selection (never touches test): models see als_train, are scored on ranker_train."""
    return make_context(split.als_train, split.ranker_train, n_items, k)


def evaluate_recommendations(
    recommendations: Mapping[int, Sequence[int]], ctx: EvalContext
) -> dict[str, float]:
    for user in ctx.users:
        top = recommendations.get(user, [])[: ctx.k]
        leaked = set(top) & ctx.seen.get(user, set())
        if leaked:
            raise ProtocolViolation(
                f"user {user}: recommended already-seen items {sorted(leaked)[:5]}"
            )
    return evaluate(recommendations, ctx.target, n_items=ctx.n_items, k=ctx.k)


def compare_models(
    models: Mapping[str, RecommendFn], ctx: EvalContext
) -> dict[str, dict[str, float]]:
    """Evaluate every model on the same context, in the given order."""
    return {name: evaluate_recommendations(fn(ctx), ctx) for name, fn in models.items()}


def format_comparison(results: Mapping[str, Mapping[str, float]], k: int = 10) -> str:
    columns = [f"ndcg_at_{k}", f"recall_at_{k}", "coverage"]
    width = max([len("model"), *(len(name) for name in results)])
    header = f"{'model':<{width}} | " + " | ".join(f"{c:>12}" for c in columns)
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        lines.append(f"{name:<{width}} | " + " | ".join(f"{metrics[c]:>12.4f}" for c in columns))
    return "\n".join(lines)
