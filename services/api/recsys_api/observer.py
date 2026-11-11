"""Hooks for service events. The default `Observer` does nothing; stage 5 plugs in
`recsys_api.metrics.MetricsObserver` (Prometheus, spec 5.1) and
`recsys_api.impressions.ImpressionObserver` (recommendation_impressions, spec 5.2), combined by
`CompositeObserver`. The service and the model holder only ever see one Observer."""

import logging
from collections.abc import Iterable, Sequence

log = logging.getLogger(__name__)


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

    def model_loaded(self, version: str, loaded_at: float, created_at: float | None = None) -> None:
        pass


class CompositeObserver(Observer):
    """Fans every event out to several observers. Monitoring must never break a request: an
    observer that raises is logged and skipped, the others still get the event."""

    def __init__(self, observers: Iterable[Observer]) -> None:
        self.observers = list(observers)

    def _each(self, method: str, *args, **kwargs) -> None:
        for observer in self.observers:
            try:
                getattr(observer, method)(*args, **kwargs)
            except Exception:  # noqa: BLE001 - see class docstring
                log.exception("observer %s.%s failed", type(observer).__name__, method)

    def recommendations_served(self, user_id, path, movie_ids, model_version, cached) -> None:
        self._each("recommendations_served", user_id, path, movie_ids, model_version, cached)

    def rating_received(self, source: str) -> None:
        self._each("rating_received", source)

    def cache_lookup(self, hit: bool) -> None:
        self._each("cache_lookup", hit)

    def model_error(self, exc: BaseException) -> None:
        self._each("model_error", exc)

    def model_loaded(self, version: str, loaded_at: float, created_at: float | None = None) -> None:
        self._each("model_loaded", version, loaded_at, created_at)
