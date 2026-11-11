"""Structured logs (spec 5.6): one JSON object per line on stdout.

Docker collects a container's stdout; with one JSON object per line, `docker compose logs api`
stays readable and a log shipper (Loki in v2) can parse fields without regexes. Stdlib only:
fields passed with `log.info("...", extra={"status": 200})` become top-level keys.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TextIO

# Attributes every LogRecord has; anything else on a record came in through `extra=`.
# color_message: uvicorn's copy of the message with terminal colour codes.
_RECORD_FIELDS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message",
    "asctime",
    "color_message",
}
TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_MARKER = "_recsys_handler"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RECORD_FIELDS and not key.startswith("_"):
                data[key] = value
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            data["stack"] = self.formatStack(record.stack_info)
        # default=str: an odd value in `extra` must never make the log call itself fail.
        return json.dumps(data, default=str, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    stream: TextIO | None = None,
    root: logging.Logger | None = None,
) -> bool:
    """Send every logger to one stdout handler in `fmt` ("json" or "text").

    Does nothing (returns False) if the root logger already has handlers: someone else (pytest,
    an embedding application, an earlier call) owns logging then."""
    if fmt not in ("json", "text"):
        raise ValueError(f"unknown log format {fmt!r} (json or text)")
    root = root or logging.getLogger()  # `root` only for tests (pytest owns the real one)
    if root.handlers:
        return False
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT))
    setattr(handler, _MARKER, True)
    root.addHandler(handler)
    root.setLevel(level)
    return True
