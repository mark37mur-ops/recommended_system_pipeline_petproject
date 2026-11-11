"""Hooks for service events. Stage 5 subclasses `Observer` to export Prometheus metrics
(spec 5.1: recommendations_served_total{path}, ratings_received_total{source},
cache_requests_total{result}, model_info) and to log recommendation impressions (spec 5.2).
The default does nothing."""

from collections.abc import Sequence


class Observer:
    def recommendations_served(
        self,
        user_id: int | None,
        path: str,
        movie_ids: Sequence[int],
        model_version: str | None,
        cached: bool,
    ) -> None:
        pass

    def rating_received(self, source: str) -> None:
        pass

    def cache_lookup(self, hit: bool) -> None:
        pass

    def model_error(self, exc: BaseException) -> None:
        pass

    def model_loaded(self, version: str, loaded_at: float) -> None:
        pass
