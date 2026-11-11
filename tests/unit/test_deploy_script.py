"""infra/scripts/deploy.sh against fake docker/curl/git (shell_fakes): the new tag goes live
only when /ready answers, a failed pull or a version that never gets ready is rolled back
(tag in .env and git checkout), and nothing runs for a malformed tag."""

import os
import stat
from pathlib import Path

import pytest
from shell_fakes import ShellHarness, find_bash, posix

BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash (Git's bash on Windows)")

OLD, NEW = "sha-1111111", "sha-2222222"
OLD_SHA = "1111111aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
NEW_SHA = "2222222bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
PREFIX = "ghcr.io/owner/repo"
FAST = {"READY_TIMEOUT": "1", "READY_INTERVAL": "0.2"}


def env_file(tag: str | None) -> str:
    lines = ["# server .env", "POSTGRES_PASSWORD=secret-value", "SITE_ADDRESS=example.test"]
    lines += [f"IMAGE_PREFIX={PREFIX}"]
    if tag is not None:
        lines.append(f"IMAGE_TAG={tag}")
    lines.append("EDGE_SUBNET=10.203.17.0/24")
    return "\n".join(lines) + "\n"


@pytest.fixture
def app(tmp_path: Path) -> Path:
    """A fake checkout: .env with the running version, a .git marker, the fake already 'up'."""
    app = tmp_path / "app"
    app.mkdir()
    (app / ".git").mkdir()
    (app / ".env").write_text(env_file(OLD), newline="\n")
    return app


@pytest.fixture
def shell(tmp_path: Path) -> ShellHarness:
    harness = ShellHarness(tmp_path, BASH)
    (harness.state / "running").write_text(OLD + "\n")
    return harness


def env_value(app: Path, key: str) -> str | None:
    for line in (app / ".env").read_text().splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    return None


def ups(calls: list[str]) -> list[str]:
    return [c.removeprefix("up ") for c in calls if c.startswith("up ")]


def test_success_switches_tag_and_checkout(app, shell):
    result = shell.run("deploy.sh", NEW, NEW_SHA, cwd=app, FAKE_GIT_HEAD=OLD_SHA, **FAST)

    assert result.returncode == 0, result.output
    assert env_value(app, "IMAGE_TAG") == NEW
    assert ups(result.calls) == [NEW]
    assert "git fetch --quiet origin" in result.calls
    assert result.file("head") == NEW_SHA
    assert "curl https://example.test/ready" in result.calls
    compose = [c for c in result.calls if c.startswith("docker compose")]
    assert all("-f docker-compose.yml -f docker-compose.prod.yml" in c for c in compose)
    assert any(" pull " in f"{c} " for c in compose)


def test_env_file_keeps_other_lines_and_mode(app, shell):
    before = (app / ".env").read_text().splitlines()
    (app / ".env").chmod(0o600)

    result = shell.run("deploy.sh", NEW, cwd=app, **FAST)

    assert result.returncode == 0, result.output
    after = (app / ".env").read_text().splitlines()
    assert after == [line.replace(f"IMAGE_TAG={OLD}", f"IMAGE_TAG={NEW}") for line in before]
    if os.name != "nt":  # Windows has no POSIX modes
        assert stat.S_IMODE((app / ".env").stat().st_mode) == 0o600


def test_git_ref_defaults_to_the_sha_in_the_tag(app, shell):
    result = shell.run("deploy.sh", NEW, cwd=app, FAKE_GIT_HEAD=OLD_SHA, **FAST)

    assert result.returncode == 0, result.output
    assert result.file("head") == "2222222"


def test_not_ready_rolls_back_tag_and_checkout(app, shell):
    result = shell.run(
        "deploy.sh", NEW, NEW_SHA, cwd=app, FAKE_GIT_HEAD=OLD_SHA, FAKE_UNREADY_TAGS=NEW, **FAST
    )

    assert result.returncode == 1
    assert env_value(app, "IMAGE_TAG") == OLD
    assert ups(result.calls) == [NEW, OLD]
    assert result.file("head") == OLD_SHA
    assert result.file("running") == OLD
    assert "rolled back to sha-1111111" in result.output
    assert "fake api log line" in result.output  # the API logs of the failed version


def test_failed_pull_rolls_back_without_starting_the_new_version(app, shell):
    result = shell.run("deploy.sh", NEW, cwd=app, FAKE_BAD_PULL_TAGS=NEW, **FAST)

    assert result.returncode == 1
    assert env_value(app, "IMAGE_TAG") == OLD
    assert ups(result.calls) == [OLD]  # only the rollback's `up`
    assert result.file("running") == OLD


def test_rollback_that_does_not_get_ready_says_so(app, shell):
    result = shell.run("deploy.sh", NEW, cwd=app, FAKE_UNREADY_TAGS=f"{NEW} {OLD}", **FAST)

    assert result.returncode == 1
    assert env_value(app, "IMAGE_TAG") == OLD
    assert "did not become ready either" in result.output


@pytest.mark.parametrize("previous", [None, "latest"])
def test_first_deploy_has_nothing_to_roll_back_to(tmp_path, shell, previous):
    app = tmp_path / "first"
    app.mkdir()
    (app / ".env").write_text(env_file(previous), newline="\n")

    result = shell.run("deploy.sh", NEW, cwd=app, FAKE_UNREADY_TAGS=NEW, **FAST)

    assert result.returncode == 1
    assert "nothing to roll back to" in result.output
    assert ups(result.calls) == [NEW]  # no second `up` with an empty or moving tag


@pytest.mark.parametrize(
    "args",
    [[], ["sha-1; rm -rf /"], ["../../etc"], [NEW, "main; reboot"], [NEW, "not-hex"]],
)
def test_malformed_input_is_refused_before_anything_runs(app, shell, args):
    result = shell.run("deploy.sh", *args, cwd=app, **FAST)

    assert result.returncode == 1
    assert result.calls == []
    assert env_value(app, "IMAGE_TAG") == OLD


def test_without_git_sync_the_checkout_is_not_touched(app, shell):
    result = shell.run("deploy.sh", NEW, cwd=app, DEPLOY_GIT_SYNC="0", **FAST)

    assert result.returncode == 0, result.output
    assert not [c for c in result.calls if c.startswith("git ")]


def test_success_prunes_images_older_than_the_previous_tag(app, shell, tmp_path):
    images = tmp_path / "images.txt"
    lines = [
        f"{PREFIX}-{name}:{tag}"
        for name in ("api", "airflow", "mlflow")
        for tag in ("sha-0000000", OLD, NEW, "latest")
    ]
    images.write_text("\n".join(lines) + "\n")

    result = shell.run("deploy.sh", NEW, cwd=app, FAKE_IMAGES=posix(images), **FAST)

    assert result.returncode == 0, result.output
    removed = sorted(c.removeprefix("docker image rm ") for c in result.calls if " rm " in c)
    assert removed == sorted(f"{PREFIX}-{n}:sha-0000000" for n in ("api", "airflow", "mlflow"))


def test_missing_env_file_is_an_error(tmp_path, shell):
    empty = tmp_path / "empty"
    empty.mkdir()

    result = shell.run("deploy.sh", NEW, cwd=empty, **FAST)

    assert result.returncode == 1
    assert "no .env" in result.output
    assert result.calls == []
