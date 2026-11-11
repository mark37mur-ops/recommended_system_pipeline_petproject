"""Publish the read-only public copy of the Grafana dashboard (spec 5.4), idempotently.

Grafana's "public dashboards" (shared dashboards) serve one dashboard at
/public-dashboards/<access token> without a login. Unlike anonymous Viewer access, the public
page can only run the saved queries of its own panels, never an arbitrary query against a
data source. File provisioning cannot create public dashboards, so this one-shot job (compose
service `grafana-publish`) does it through Grafana's HTTP API after every start:
- not published yet -> create it (enabled, time picker on, no annotations);
- published but paused -> enable it again;
- published -> nothing to do.
It prints the link. Standard library only: it runs in the API image.

Env: GRAFANA_URL (http://grafana:3000), GRAFANA_ADMIN_USER, GRAFANA_ADMIN_PASSWORD,
DASHBOARD_UID (recsys-overview), GRAFANA_PUBLIC_URL (base of the printed link).
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable

Api = Callable[[str, str, dict | None], tuple[int, dict]]
SETTINGS = {
    "isEnabled": True,
    "share": "public",
    "timeSelectionEnabled": True,
    "annotationsEnabled": False,
}


def http_api(base_url: str, user: str, password: str, timeout: float = 10.0) -> Api:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()

    def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=data,
            method=method,
            headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                return exc.code, json.loads(raw) if raw else {}
            except ValueError:
                return exc.code, {"message": raw.decode(errors="replace")}

    return call


def wait_for_dashboard(api: Api, uid: str, timeout: float = 120.0, step: float = 2.0) -> None:
    """Grafana up and the provisioned dashboard loaded (provisioning runs after start-up)."""
    deadline = time.monotonic() + timeout
    last = "no answer"
    while time.monotonic() < deadline:
        try:
            status, body = api("GET", f"/api/dashboards/uid/{uid}", None)
            if status == 200:
                return
            last = f"HTTP {status}: {body.get('message', body)}"
        except OSError as exc:  # connection refused while Grafana starts
            last = str(exc)
        time.sleep(step)
    raise RuntimeError(f"dashboard {uid} not available after {timeout:.0f}s ({last})")


def ensure_public_dashboard(api: Api, uid: str) -> tuple[str, dict]:
    """('created' | 'enabled' | 'unchanged', public dashboard config)."""
    path = f"/api/dashboards/uid/{uid}/public-dashboards"
    status, current = api("GET", path, None)
    if status == 200 and current.get("accessToken"):
        if current.get("isEnabled"):
            return "unchanged", current
        status, updated = api("PATCH", f"{path}/{current['uid']}", {"isEnabled": True})
        if status != 200:
            raise RuntimeError(f"enabling the public dashboard failed: HTTP {status} {updated}")
        return "enabled", current | updated
    if status not in (200, 404):
        raise RuntimeError(f"reading the public dashboard failed: HTTP {status} {current}")
    status, created = api("POST", path, dict(SETTINGS))
    if status != 200:
        raise RuntimeError(f"creating the public dashboard failed: HTTP {status} {created}")
    return "created", created


def main() -> int:
    url = os.environ.get("GRAFANA_URL", "http://grafana:3000")
    uid = os.environ.get("DASHBOARD_UID", "recsys-overview")
    public = os.environ.get("GRAFANA_PUBLIC_URL", "http://localhost:3000").rstrip("/")
    api = http_api(
        url, os.environ.get("GRAFANA_ADMIN_USER", "admin"), os.environ["GRAFANA_ADMIN_PASSWORD"]
    )
    try:
        wait_for_dashboard(api, uid)
        outcome, config = ensure_public_dashboard(api, uid)
    except RuntimeError as exc:
        print(f"publish_dashboard: {exc}", file=sys.stderr)
        return 1
    print(f"public dashboard {outcome}: {public}/public-dashboards/{config['accessToken']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
