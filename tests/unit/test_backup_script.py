"""infra/scripts/backup.sh against fake docker (shell_fakes): one archive with both database
dumps and the MLflow artifacts, only the newest BACKUP_KEEP archives stay, and a failed dump
leaves no archive (and no temporary files) behind."""

import tarfile
from pathlib import Path

import pytest
from shell_fakes import ShellHarness, find_bash

BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash (Git's bash on Windows)")


@pytest.fixture
def app(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    return app


@pytest.fixture
def shell(tmp_path: Path) -> ShellHarness:
    return ShellHarness(tmp_path, BASH)


def archives(app: Path) -> list[Path]:
    return sorted((app / "backups").glob("backup-*.tar"))


def test_archive_holds_both_dumps_and_the_artifacts(app, shell):
    result = shell.run("backup.sh", cwd=app)

    assert result.returncode == 0, result.output
    [archive] = archives(app)
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == ["app.dump", "mlartifacts.tar.gz", "mlflow.dump"]
        assert tar.extractfile("app.dump").read() == b"DUMP app\n"
        assert tar.extractfile("mlflow.dump").read() == b"DUMP mlflow\n"
    dumps = [c for c in result.calls if "pg_dump" in c]
    assert all("exec -T postgres pg_dump -U postgres -Fc" in c for c in dumps)
    assert any("exec -T mlflow tar -C /mlartifacts" in c for c in result.calls)


def test_keeps_only_the_newest_archives(app, shell):
    backups = app / "backups"
    backups.mkdir()
    old = [backups / f"backup-202001{day:02d}T000000Z.tar" for day in range(1, 9)]
    for path in old:
        path.write_text("old")

    result = shell.run("backup.sh", cwd=app, BACKUP_KEEP="3")

    assert result.returncode == 0, result.output
    names = [p.name for p in archives(app)]
    assert len(names) == 3
    assert names[:2] == [old[-2].name, old[-1].name]  # the two newest old ones
    assert names[2].startswith("backup-20")  # + the new one (current year)
    assert names[2] not in {p.name for p in old}


def test_failed_dump_leaves_nothing_behind(app, shell):
    result = shell.run("backup.sh", cwd=app, FAKE_PG_DUMP_FAIL="1")

    assert result.returncode != 0
    assert archives(app) == []
    assert list((app / "backups").iterdir()) == []  # no .tmp-* directory either


def test_rejects_a_bad_keep_value(app, shell):
    result = shell.run("backup.sh", cwd=app, BACKUP_KEEP="0")

    assert result.returncode == 1
    assert result.calls == []
