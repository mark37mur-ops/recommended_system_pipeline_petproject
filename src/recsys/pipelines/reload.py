"""Last task of the `promote` branch: make the API load the new champion (spec 3.2, 4.2).

Contract of the API (`POST /admin/reload-model`, Bearer ADMIN_TOKEN): 200 `loaded` (new
version) or `unchanged` (already serving the alias), 502 `failed` (old model keeps serving),
409 `busy`, 401 bad token, 503 no token configured.

Decision 10: the alias has already moved and stays moved, the Registry is the source of truth.
Anything but "the API now serves exactly the promoted version" raises ReloadError, so Airflow
retries and then shows the task red; the model is never un-promoted from here. The standard
library client keeps the task free of extra dependencies.
"""

import json
import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from recsys.pipelines.run_dir import RunDir, read_json, write_json_atomic
from recsys.pipelines.settings import PipelineSettings, prepare

log = logging.getLogger("recsys.retrain")
DEFAULT_API_URL = "http://api:8000"


class ReloadError(RuntimeError):
    """The API did not confirm that it serves the promoted version."""


def _answer(body: bytes) -> dict:
    try:
        parsed = json.loads(body or b"{}")
    except ValueError:
        return {"detail": body[:200].decode("utf-8", "replace")}
    return parsed if isinstance(parsed, dict) else {"detail": parsed}


def reload_api(
    run_dir: str, settings: PipelineSettings | None = None, timeout: float = 60.0
) -> str:
    """POST /admin/reload-model and check the API now serves the version this run promoted.

    `settings.api_url` / `settings.admin_token`, else
    $API_URL (default http://api:8000) and $ADMIN_TOKEN. The call is synchronous on the API
    side (~1 s for a real model, up to ~16 s when MLflow is slow), hence the long timeout."""
    settings = prepare(settings)
    rd = RunDir(Path(run_dir))
    expected = str(read_json(rd.promotion_json)["version"])
    base = (settings.api_url or os.environ.get("API_URL") or DEFAULT_API_URL).rstrip("/")
    token = settings.admin_token or os.environ.get("ADMIN_TOKEN") or ""
    if not token.strip():
        raise ReloadError("ADMIN_TOKEN is not set: cannot call the API admin endpoint")

    url = f"{base}/admin/reload-model"
    request = Request(url, data=b"", method="POST", headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(request, timeout=timeout) as response:
            code, answer = response.status, _answer(response.read())
    except HTTPError as exc:
        code, answer = exc.code, _answer(exc.read())
    except (URLError, OSError) as exc:  # refused, DNS, timeout
        raise ReloadError(f"API unreachable at {url}: {exc}") from exc

    if code != 200:
        reason = answer.get("error") or answer.get("detail") or answer.get("status")
        raise ReloadError(f"API reload answered {code}: {reason} ({answer})")
    served = None if answer.get("version") is None else str(answer["version"])
    if answer.get("status") not in {"loaded", "unchanged"} or served != expected:
        raise ReloadError(
            f"API serves v{served} ({answer.get('status')}), expected v{expected}: {answer}"
        )
    write_json_atomic(rd.reload_json, answer)
    log.info(
        "API %s model v%s (previous v%s)", answer["status"], served, answer.get("previous_version")
    )
    return run_dir
