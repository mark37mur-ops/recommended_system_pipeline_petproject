import hashlib
import zipfile
from types import SimpleNamespace

import pytest

from recsys.data.download import download_ml1m


@pytest.fixture
def fake_zip(tmp_path_factory):
    src = tmp_path_factory.mktemp("fake_src")
    path = src / "ml-1m.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ml-1m/ratings.dat", "1::1::5::978300760\n")
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    return SimpleNamespace(uri=path.as_uri(), md5=md5)


def _fail(*args, **kwargs):
    raise AssertionError("network must not be touched")


def test_download_extracts_and_verifies(tmp_path, fake_zip):
    out = download_ml1m(tmp_path, url=fake_zip.uri, md5=fake_zip.md5)
    assert (out / "ratings.dat").exists()


def test_download_skips_when_archive_valid(tmp_path, fake_zip, monkeypatch):
    download_ml1m(tmp_path, url=fake_zip.uri, md5=fake_zip.md5)
    monkeypatch.setattr("recsys.data.download.urlopen", _fail)
    download_ml1m(tmp_path, url=fake_zip.uri, md5=fake_zip.md5)


def test_download_redownloads_corrupted_archive(tmp_path, fake_zip):
    (tmp_path / "ml-1m.zip").write_bytes(b"garbage")
    out = download_ml1m(tmp_path, url=fake_zip.uri, md5=fake_zip.md5)
    assert (out / "ratings.dat").exists()


def test_download_raises_on_md5_mismatch(tmp_path, fake_zip):
    with pytest.raises(ValueError, match="MD5"):
        download_ml1m(tmp_path, url=fake_zip.uri, md5="0" * 32)
    assert not (tmp_path / "ml-1m.zip.part").exists()
