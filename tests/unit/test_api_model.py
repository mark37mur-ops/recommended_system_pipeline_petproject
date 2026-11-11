import threading

from api_fakes import FakeChampion, StubRecommender

from recsys_api.model import ModelHolder


class Registry:
    """Fake MLflow registry: `version` is the champion alias, `fail` breaks loading."""

    def __init__(self, version: str | None = "1") -> None:
        self.version = version
        self.fail = False
        self.loads = 0

    def probe(self) -> str | None:
        return self.version

    def load(self) -> FakeChampion:
        self.loads += 1
        if self.fail:
            raise ConnectionError("mlflow is down")
        return FakeChampion(StubRecommender(), self.version)


def test_first_reload_loads_and_exposes_version():
    registry = Registry("3")
    holder = ModelHolder(registry.load, registry.probe, clock=lambda: 123.0)
    assert holder.current is None
    result = holder.reload()
    assert (result.status, result.version, result.previous_version) == ("loaded", "3", None)
    assert holder.current.version == "3" and holder.current.loaded_at == 123.0


def test_same_version_is_unchanged_unless_forced():
    registry = Registry("3")
    holder = ModelHolder(registry.load, registry.probe)
    holder.reload()
    first = holder.current
    assert holder.reload().status == "unchanged" and registry.loads == 1
    assert holder.current is first
    assert holder.reload(force=True).status == "loaded" and registry.loads == 2


def test_failed_reload_keeps_the_old_model():
    registry = Registry("3")
    holder = ModelHolder(registry.load, registry.probe)
    holder.reload()
    old = holder.current
    registry.version, registry.fail = "4", True
    result = holder.reload()
    assert result.status == "failed" and result.version == "3" and "mlflow is down" in result.error
    assert holder.current is old and "mlflow is down" in holder.last_error


def test_validation_failure_keeps_the_old_model():
    registry = Registry("3")

    def validate(recommender):
        if registry.version == "4":
            raise ValueError("smoke prediction failed")

    holder = ModelHolder(registry.load, registry.probe, validate=validate)
    holder.reload()
    registry.version = "4"
    assert holder.reload().status == "failed" and holder.current.version == "3"


def test_missing_champion_is_a_failure():
    registry = Registry(None)
    holder = ModelHolder(registry.load, registry.probe)
    result = holder.reload()
    assert result.status == "failed" and "champion" in result.error
    assert registry.loads == 0 and holder.current is None


def test_probe_error_is_a_failure_not_an_exception():
    def probe():
        raise ConnectionError("registry unreachable")

    holder = ModelHolder(Registry().load, probe)
    assert holder.reload().status == "failed"


def test_concurrent_reload_reports_busy():
    started, release = threading.Event(), threading.Event()
    registry = Registry("3")

    def slow_load():
        started.set()
        release.wait(5)
        return registry.load()

    holder = ModelHolder(slow_load, registry.probe)
    worker = threading.Thread(target=holder.reload)
    worker.start()
    started.wait(5)
    assert holder.reload().status == "busy"
    release.set()
    worker.join(5)
    assert holder.current.version == "3"


def test_reader_keeps_a_consistent_reference_during_swap():
    registry = Registry("1")
    holder = ModelHolder(registry.load, registry.probe)
    holder.reload()
    snapshot = holder.current
    registry.version = "2"
    holder.reload()
    assert snapshot.version == "1" and holder.current.version == "2"
