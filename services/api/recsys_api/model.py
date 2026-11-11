"""The served model and its hot swap (spec 3.2).

`ModelHolder.current` is a single reference to an immutable `LoadedModel`. A request reads it
once and works with that object; `reload` loads the new champion off to the side, validates it
and then replaces the reference with one assignment (atomic in CPython). Any failure leaves the
previous model in place. Only one reload runs at a time; a concurrent one reports `busy`.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from recsys_api.observer import Observer

log = logging.getLogger(__name__)

ReloadStatus = Literal["loaded", "unchanged", "failed", "busy"]


class Champion(Protocol):
    recommender: Any
    version: str
    run_id: str | None


@dataclass(frozen=True)
class LoadedModel:
    recommender: Any  # recsys.models.two_stage.TwoStageRecommender in production
    version: str
    run_id: str | None
    loaded_at: float


@dataclass(frozen=True)
class ReloadResult:
    status: ReloadStatus
    version: str | None  # version served after the call
    previous_version: str | None
    error: str | None = None


class ModelHolder:
    def __init__(
        self,
        loader: Callable[[], Champion],
        version_probe: Callable[[], str | None],
        *,
        validate: Callable[[Any], None] | None = None,
        clock: Callable[[], float] = time.time,
        observer: Observer | None = None,
    ) -> None:
        self._loader = loader
        self._probe = version_probe
        self._validate = validate
        self._clock = clock
        self._observer = observer or Observer()
        self._lock = threading.Lock()
        self._current: LoadedModel | None = None
        self.last_error: str | None = None

    @property
    def current(self) -> LoadedModel | None:
        return self._current

    def reload(self, force: bool = False) -> ReloadResult:
        """Load the current champion unless it is already served (`force` reloads anyway)."""
        if not self._lock.acquire(blocking=False):
            current = self._current
            return ReloadResult("busy", current and current.version, current and current.version)
        previous = self._current
        previous_version = previous.version if previous else None
        try:
            version = self._probe()
            if version is None:
                raise LookupError("registered model has no 'champion' alias")
            if not force and previous is not None and version == previous.version:
                return ReloadResult("unchanged", previous_version, previous_version)
            champion = self._loader()
            if self._validate is not None:
                self._validate(champion.recommender)
            loaded = LoadedModel(
                champion.recommender, str(champion.version), champion.run_id, self._clock()
            )
            self._current = loaded  # the swap
            self.last_error = None
            self._observer.model_loaded(loaded.version, loaded.loaded_at)
            log.info("model version %s loaded (previous %s)", loaded.version, previous_version)
            return ReloadResult("loaded", loaded.version, previous_version)
        except Exception as exc:  # noqa: BLE001 - any failure keeps the old model
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("model reload failed, keeping version %s: %s", previous_version, exc)
            return ReloadResult("failed", previous_version, previous_version, self.last_error)
        finally:
            self._lock.release()
