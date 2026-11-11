import io
import json
import logging

import pytest

from recsys.logs import JsonFormatter, configure_logging


def _record(msg="hello %s", args=("world",), exc_info=None, **extra):
    record = logging.LogRecord("recsys.test", logging.WARNING, __file__, 1, msg, args, exc_info)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_one_json_object_per_line_with_standard_fields():
    line = JsonFormatter().format(_record())
    assert "\n" not in line
    data = json.loads(line)
    assert data["level"] == "WARNING"
    assert data["logger"] == "recsys.test"
    assert data["message"] == "hello world"
    assert data["ts"].endswith("+00:00")  # UTC, ISO 8601


def test_extra_fields_become_keys_and_odd_values_are_stringified():
    data = json.loads(JsonFormatter().format(_record(endpoint="/ready", status=200, obj={1, 2})))
    assert data["endpoint"] == "/ready" and data["status"] == 200
    assert isinstance(data["obj"], str)
    assert "args" not in data and "msg" not in data  # LogRecord internals stay out


def test_exception_is_one_field():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record("failed", (), exc_info=sys.exc_info())
    data = json.loads(JsonFormatter().format(record))
    assert data["message"] == "failed"
    assert "ValueError: boom" in data["exc"]


@pytest.fixture
def clean_root():
    """A stand-alone logger playing the root: pytest attaches capture handlers to the real
    root (and to loggers it has seen) for every test phase."""
    return logging.Logger("isolated-root")


def test_configure_logging_installs_one_json_handler(clean_root):
    stream = io.StringIO()
    assert configure_logging("INFO", "json", stream=stream, root=clean_root) is True
    assert configure_logging("INFO", "json", stream=stream, root=clean_root) is False  # once
    clean_root.info("ready", extra={"version": "6"})
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["version"] == "6"


def test_configure_logging_leaves_a_configured_root_alone(clean_root):
    existing = logging.StreamHandler(io.StringIO())
    clean_root.addHandler(existing)  # e.g. pytest's or an embedding application's
    assert configure_logging("INFO", "json", root=clean_root) is False
    assert clean_root.handlers == [existing]


def test_text_format_for_local_debugging(clean_root):
    stream = io.StringIO()
    configure_logging("INFO", "text", stream=stream, root=clean_root)
    clean_root.info("plain")
    assert stream.getvalue().strip().endswith("INFO isolated-root: plain")


def test_unknown_format_is_rejected(clean_root):
    with pytest.raises(ValueError, match="log format"):
        configure_logging("INFO", "xml", root=clean_root)
