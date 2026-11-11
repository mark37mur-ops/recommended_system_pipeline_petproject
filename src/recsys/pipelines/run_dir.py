"""Layout of one pipeline run on disk: `/data/runs/{run_id}/` (spec 4.1).

Tasks of `retrain_daily` hand data to each other through files in this folder, never through
XCom: XCom carries only the folder path. Every output is written atomically (a temp sibling is
filled, then renamed into place), so a task killed half-way leaves either the previous complete
output or nothing, and a retry simply writes it again.
"""

import json
import os
import re
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RUNS_ROOT = Path("/data/runs")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def resolve_runs_root(root: Path | str | None = None) -> Path:
    """Explicit argument > $RUNS_DIR > /data/runs (the shared volume in the Airflow image)."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get("RUNS_DIR") or DEFAULT_RUNS_ROOT)


def safe_run_id(run_id: str) -> str:
    """Airflow run ids look like `manual__2026-11-11T10:00:00+00:00`: keep a folder-safe form."""
    safe = _UNSAFE.sub("-", run_id).strip("-")
    if safe in {"", ".", ".."}:
        raise ValueError(f"run id {run_id!r} does not give a usable folder name")
    return safe


@dataclass(frozen=True)
class RunDir:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))  # accept the str from XCom

    @classmethod
    def for_run(cls, run_id: str, runs_root: Path | str | None = None) -> "RunDir":
        return cls(resolve_runs_root(runs_root) / safe_run_id(run_id))

    # extract_ratings
    @property
    def extract_dir(self) -> Path:
        return self.root / "extract"

    @property
    def ratings_path(self) -> Path:
        return self.extract_dir / "ratings.parquet"

    @property
    def movies_path(self) -> Path:
        return self.extract_dir / "movies.parquet"

    @property
    def users_path(self) -> Path:
        return self.extract_dir / "users.parquet"

    @property
    def extract_json(self) -> Path:
        return self.extract_dir / "extract.json"

    # validate_data / drift_report
    @property
    def validation_json(self) -> Path:
        return self.root / "validation.json"

    @property
    def drift_json(self) -> Path:
        return self.root / "drift.json"

    @property
    def drift_dir(self) -> Path:  # Evidently HTML reports (stage 5)
        return self.root / "drift"

    # split
    @property
    def split_dir(self) -> Path:
        return self.root / "split"

    def split_path(self, part: str) -> Path:
        return self.split_dir / f"{part}.parquet"

    # train_als / train_ranker
    @property
    def als_dir(self) -> Path:
        return self.root / "als"

    @property
    def ranker_dir(self) -> Path:
        return self.root / "ranker"

    @property
    def ranker_json(self) -> Path:
        return self.ranker_dir / "diagnostics.json"

    # evaluate / decide / promote
    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def evaluate_json(self) -> Path:
        return self.root / "evaluate.json"

    @property
    def decision_json(self) -> Path:
        return self.root / "decision.json"

    @property
    def promotion_json(self) -> Path:
        return self.root / "promotion.json"

    @property
    def reload_json(self) -> Path:
        return self.root / "reload.json"

    @property
    def mlflow_json(self) -> Path:
        return self.root / "mlflow_run.json"

    @property
    def scratch_dir(self) -> Path:
        """Temp space for MLflow uploads/downloads (never the system temp dir)."""
        return self.root / "tmp"


@contextmanager
def atomic_dir(target: Path) -> Iterator[Path]:
    """Yield an empty temp directory next to `target`; on success it replaces `target`."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex[:8]}")
    tmp.mkdir()
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    old = None
    if target.exists():
        # A directory cannot be renamed onto a non-empty one: move the old output aside first.
        old = target.with_name(f".{target.name}.old-{uuid.uuid4().hex[:8]}")
        os.replace(target, old)
    os.replace(tmp, target)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def write_json_atomic(path: Path, obj: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
