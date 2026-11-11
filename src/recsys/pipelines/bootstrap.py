"""Tasks of the manual `bootstrap_data` DAG (spec 4.2): download -> checksum -> Postgres.

Each step is idempotent: the download is skipped when a valid archive is already there, the
loader upserts (a re-run on a loaded database changes nothing, web ratings are never touched).
Paths pass between tasks as strings (XCom), the files live on the shared /data volume.
"""

import logging
import os
from dataclasses import asdict
from pathlib import Path

from recsys.data.download import ML1M_MD5, ML1M_URL, extract_archive, fetch_archive, file_md5
from recsys.db.load_movielens import LoadStats, load_movielens
from recsys.db.schema import make_engine

log = logging.getLogger("recsys.bootstrap")
DEFAULT_RAW_DIR = Path("/data/raw")

__all__ = ["ChecksumError", "LoadStats", "download", "load_postgres", "verify_checksum"]


class ChecksumError(ValueError):
    pass


def _logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def download(raw_dir: str | Path | None = None, url: str = ML1M_URL, md5: str = ML1M_MD5) -> str:
    """Fetch ml-1m.zip into raw_dir ($RAW_DIR, default /data/raw) unless a valid copy exists."""
    _logging()
    raw_dir = Path(raw_dir or os.environ.get("RAW_DIR") or DEFAULT_RAW_DIR)
    archive = fetch_archive(raw_dir, url=url, md5=md5)
    log.info("archive ready: %s (%d bytes)", archive, archive.stat().st_size)
    return str(archive)


def verify_checksum(archive: str, md5: str = ML1M_MD5) -> str:
    """Separate, explicit check of the file that is about to be loaded (spec 4.2)."""
    _logging()
    actual = file_md5(Path(archive))
    if actual != md5:
        raise ChecksumError(f"MD5 mismatch for {archive}: expected {md5}, got {actual}")
    log.info("MD5 ok: %s", actual)
    return archive


def load_postgres(archive: str, database_url: str | None = None) -> dict[str, int]:
    """Unpack next to the archive and upsert movies, users, ratings into the `app` database."""
    _logging()
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (the `app` database)")
    archive_path = Path(archive)
    raw = extract_archive(archive_path, archive_path.parent)
    engine = make_engine(url)
    try:
        stats = load_movielens(engine, raw)
    finally:
        engine.dispose()
    log.info("loaded: %s", stats)
    return asdict(stats)
