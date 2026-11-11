"""reload_api against a real HTTP server on 127.0.0.1 that plays the API's admin endpoint."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from recsys.pipelines.reload import ReloadError, reload_api
from recsys.pipelines.retrain import PipelineSettings
from recsys.pipelines.run_dir import RunDir, read_json, write_json_atomic

TOKEN = "t" * 43


class FakeApi:
    """Answers POST /admin/reload-model with a canned (status, body); records requests."""

    def __init__(self) -> None:
        self.reply: tuple[int, dict] = (200, {})
        self.requests: list[dict] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (http.server naming)
                api.requests.append({"path": self.path, "auth": self.headers.get("Authorization")})
                status, body = api.reply
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def api():
    fake = FakeApi()
    yield fake
    fake.close()


@pytest.fixture
def run_dir(tmp_path) -> str:
    rd = RunDir(tmp_path / "run")
    write_json_atomic(rd.promotion_json, {"version": "5", "promoted": True})
    return str(rd.root)


def _settings(url: str, token: str | None = TOKEN) -> PipelineSettings:
    return PipelineSettings(api_url=url, admin_token=token)


def _body(status: str, version: str | None = "5", error: str | None = None) -> dict:
    return {
        "status": status,
        "version": version,
        "previous_version": "4",
        "error": error,
        "catalog_size": 3883,
    }


@pytest.mark.parametrize("status", ["loaded", "unchanged"])
def test_reload_succeeds_when_the_api_serves_the_promoted_version(api, run_dir, status):
    api.reply = (200, _body(status))
    assert reload_api(run_dir, _settings(api.url + "/")) == run_dir
    assert api.requests == [{"path": "/admin/reload-model", "auth": f"Bearer {TOKEN}"}]
    assert read_json(RunDir(run_dir).reload_json)["version"] == "5"


def test_reload_fails_when_the_api_serves_another_version(api, run_dir):
    api.reply = (200, _body("loaded", version="6"))
    with pytest.raises(ReloadError, match="v6.*v5|expected v5"):
        reload_api(run_dir, _settings(api.url))


@pytest.mark.parametrize(
    ("code", "body", "message"),
    [
        (502, _body("failed", version="4", error="MLflow unreachable"), "MLflow unreachable"),
        (409, _body("busy", version="4"), "busy"),
        (401, {"detail": "bad admin token"}, "bad admin token"),
        (503, {"detail": "ADMIN_TOKEN is not configured"}, "ADMIN_TOKEN"),
    ],
)
def test_reload_errors_carry_the_api_answer(api, run_dir, code, body, message):
    api.reply = (code, body)
    with pytest.raises(ReloadError, match=message):
        reload_api(run_dir, _settings(api.url))


def test_unreachable_api_is_an_error(run_dir):
    with socket.socket() as s:  # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(ReloadError, match="unreachable"):
        reload_api(run_dir, _settings(f"http://127.0.0.1:{port}"), timeout=2)


def test_missing_token_fails_before_any_request(api, run_dir, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    with pytest.raises(ReloadError, match="ADMIN_TOKEN"):
        reload_api(run_dir, _settings(api.url, token=None))
    assert api.requests == []


def test_settings_come_from_the_environment(api, run_dir, monkeypatch):
    monkeypatch.setenv("API_URL", api.url)
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    api.reply = (200, _body("loaded"))
    assert reload_api(run_dir) == run_dir
    assert api.requests[0]["auth"] == f"Bearer {TOKEN}"
