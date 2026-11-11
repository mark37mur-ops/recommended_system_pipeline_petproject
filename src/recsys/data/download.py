"""Download and unpack MovieLens 1M with MD5 verification."""

import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlopen

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
# Official checksum from https://files.grouplens.org/datasets/movielens/ml-1m.zip.md5
ML1M_MD5 = "c4d9eecfca2ab87c1945afe126590906"


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_ml1m(dest_dir: Path, url: str = ML1M_URL, md5: str = ML1M_MD5) -> Path:
    """Ensure ml-1m is downloaded and extracted; return ``dest_dir / "ml-1m"``."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "ml-1m.zip"

    if not (archive.exists() and file_md5(archive) == md5):
        part = archive.with_name(archive.name + ".part")
        try:
            with urlopen(url) as resp, part.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            actual = file_md5(part)
            if actual != md5:
                raise ValueError(f"MD5 mismatch for {url}: expected {md5}, got {actual}")
            part.replace(archive)
        finally:
            part.unlink(missing_ok=True)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
    return dest_dir / "ml-1m"
