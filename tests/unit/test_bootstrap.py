import hashlib
import zipfile

import pytest

from recsys.pipelines import bootstrap
from recsys.pipelines.bootstrap import ChecksumError, download, load_postgres, verify_checksum


@pytest.fixture
def source(tmp_path_factory):
    src = tmp_path_factory.mktemp("src") / "ml-1m.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("ml-1m/ratings.dat", "1::1::5::978300760\n")
    return src, hashlib.md5(src.read_bytes()).hexdigest()


def test_download_then_verify(tmp_path, source):
    src, md5 = source
    archive = download(tmp_path / "raw", url=src.as_uri(), md5=md5)
    assert archive == str(tmp_path / "raw" / "ml-1m.zip")
    assert verify_checksum(archive, md5=md5) == archive


def test_download_twice_does_not_fetch_again(tmp_path, source, monkeypatch):
    src, md5 = source
    download(tmp_path / "raw", url=src.as_uri(), md5=md5)
    monkeypatch.setattr("recsys.data.download.urlopen", lambda *a, **k: pytest.fail("fetched"))
    download(tmp_path / "raw", url=src.as_uri(), md5=md5)


def test_verify_checksum_fails_on_a_changed_archive(tmp_path, source):
    src, md5 = source
    archive = download(tmp_path / "raw", url=src.as_uri(), md5=md5)
    with open(archive, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(ChecksumError, match="MD5"):
        verify_checksum(archive, md5=md5)


def test_download_defaults_to_raw_dir_from_environment(tmp_path, source, monkeypatch):
    src, md5 = source
    monkeypatch.setenv("RAW_DIR", str(tmp_path / "env-raw"))
    assert download(url=src.as_uri(), md5=md5) == str(tmp_path / "env-raw" / "ml-1m.zip")


def test_load_postgres_extracts_and_loads(tmp_path, source, monkeypatch):
    src, md5 = source
    archive = download(tmp_path / "raw", url=src.as_uri(), md5=md5)
    seen = {}

    def fake_load(engine, raw_dir):
        seen["raw_dir"] = raw_dir
        seen["url"] = str(engine.url)
        return bootstrap.LoadStats(movies=1, users=2, ratings=3)

    monkeypatch.setattr(bootstrap, "load_movielens", fake_load)
    stats = load_postgres(archive, database_url="sqlite://")
    assert stats == {"movies": 1, "users": 2, "ratings": 3}
    assert (seen["raw_dir"] / "ratings.dat").exists()
    assert seen["url"] == "sqlite://"
