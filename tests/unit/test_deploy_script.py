"""infra/scripts/deploy.sh against fake docker/curl/git (shell_fakes): the new tag goes live
only when it passes the smoke test (/ready and 10 recommendations); a failed pull, a version
that fails the smoke test or a deploy killed half-way is rolled back (tag in .env and git
checkout); an older commit is not deployed over a newer one unless asked; `--detach` runs
the deploy apart from the SSH session; nothing runs for a malformed tag."""

import os
import shutil
import stat
import time
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
    assert "curl https://example.test/recommendations?k=10" in result.calls
    # Through Caddy on this machine, without depending on DNS (s7-review Minor-12).
    curl_args = [c for c in result.calls if c.startswith("curl-args ")]
    assert curl_args
    assert all("--resolve example.test:443:127.0.0.1" in c for c in curl_args)
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


def test_rollback_that_does_not_come_up_either_says_so(app, shell):
    result = shell.run("deploy.sh", NEW, cwd=app, FAKE_DOWN_TAGS=f"{NEW} {OLD}", **FAST)

    assert result.returncode == 1
    assert env_value(app, "IMAGE_TAG") == OLD
    assert "did not pass the smoke test either" in result.output


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


def test_env_file_is_replaced_without_leftovers(app, shell):
    result = shell.run("deploy.sh", NEW, cwd=app, **FAST)

    assert result.returncode == 0, result.output
    assert sorted(p.name for p in app.iterdir() if p.name.startswith(".env")) == [".env"]


def test_failed_image_prune_does_not_fail_a_good_deploy(app, shell):
    result = shell.run("deploy.sh", NEW, cwd=app, FAKE_IMAGE_LS_FAIL="1", **FAST)

    assert result.returncode == 0, result.output
    assert env_value(app, "IMAGE_TAG") == NEW
    assert "image prune failed" in result.output


# ----- smoke test (spec 6.5: /ready = 200 and /recommendations returns 10 movies) ----------


@pytest.mark.parametrize("broken", ["FAKE_SHORT_RECS_TAGS", "FAKE_BAD_RECS_TAGS"])
def test_ready_but_broken_recommendations_rolls_back(app, shell, broken):
    result = shell.run("deploy.sh", NEW, cwd=app, **{broken: NEW}, **FAST)

    assert result.returncode == 1
    assert env_value(app, "IMAGE_TAG") == OLD
    assert ups(result.calls) == [NEW, OLD]
    assert "rolled back to sha-1111111" in result.output


def test_first_deploy_without_a_model_passes_on_the_popularity_fallback(app, shell):
    """Nothing runs yet and there is no champion (bootstrap_data and retrain_daily run only
    once the stack is up): /ready stays 503 by design (spec 3.5), so the gate is "serving"."""
    (shell.state / "running").unlink()

    result = shell.run(
        "deploy.sh", NEW, cwd=app, FAKE_UNREADY_TAGS=NEW, FAKE_EMPTY_RECS_TAGS=NEW, **FAST
    )

    assert result.returncode == 0, result.output
    assert env_value(app, "IMAGE_TAG") == NEW
    assert "curl https://example.test/health" in result.calls
    assert "curl https://example.test/recommendations?k=10" in result.calls
    assert "smoke gate: serving" in result.output


def test_degraded_stack_still_gates_on_a_working_api(app, shell):
    result = shell.run(
        "deploy.sh", NEW, cwd=app, FAKE_UNREADY_TAGS=f"{OLD} {NEW}", FAKE_DOWN_TAGS=NEW, **FAST
    )

    assert result.returncode == 1
    assert "smoke gate: serving" in result.output
    assert env_value(app, "IMAGE_TAG") == OLD
    assert ups(result.calls) == [NEW, OLD]


def test_a_ready_stack_is_not_replaced_by_one_without_a_model(app, shell):
    result = shell.run("deploy.sh", NEW, cwd=app, FAKE_UNREADY_TAGS=NEW, **FAST)

    assert result.returncode == 1
    assert "smoke gate: full" in result.output
    assert env_value(app, "IMAGE_TAG") == OLD


def test_smoke_gate_can_be_forced_to_full(app, shell):
    (shell.state / "running").unlink()

    result = shell.run("deploy.sh", NEW, cwd=app, SMOKE_GATE="full", FAKE_UNREADY_TAGS=NEW, **FAST)

    assert result.returncode == 1
    assert "smoke gate: full" in result.output


# ----- interrupted deploys (s7-review I-1) -------------------------------------------------


@pytest.mark.parametrize("signal", ["TERM", "HUP"])
def test_signal_while_waiting_rolls_back(app, shell, signal):
    result = shell.run(
        "deploy.sh",
        NEW,
        NEW_SHA,
        cwd=app,
        FAKE_GIT_HEAD=OLD_SHA,
        FAKE_SIGNAL=signal,
        FAKE_SIGNAL_TAG=NEW,
        READY_TIMEOUT="30",
        READY_INTERVAL="0.2",
    )

    assert f"signal {signal}" in result.calls
    assert result.returncode == 1
    assert f"interrupted by SIG{signal}" in result.output
    assert env_value(app, "IMAGE_TAG") == OLD
    assert ups(result.calls) == [NEW, OLD]
    assert result.file("head") == OLD_SHA
    assert result.file("running") == OLD


def wait_status(app: Path, run_id: str, timeout: float = 60) -> str:
    status = app / ".deploy" / f"{run_id}.status"
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        text = status.read_text().strip() if status.exists() else ""
        if text.startswith("done"):
            return text
        time.sleep(0.2)
    raise AssertionError(f"no result in {status}: {text!r}")


def test_detach_returns_at_once_and_the_deploy_finishes_on_its_own(app, shell):
    started = shell.run("deploy.sh", "--detach", "77-1", NEW, NEW_SHA, cwd=app, **FAST)

    assert started.returncode == 0, started.output
    assert "started" in started.output
    assert wait_status(app, "77-1") == "done 0"
    assert "deployed sha-2222222" in (app / ".deploy" / "77-1.log").read_text()
    assert env_value(app, "IMAGE_TAG") == NEW


def test_detached_deploy_that_is_killed_rolls_back_and_says_so(app, shell):
    started = shell.run(
        "deploy.sh",
        "--detach",
        "78-1",
        NEW,
        cwd=app,
        FAKE_SIGNAL="TERM",
        FAKE_SIGNAL_TAG=NEW,
        READY_TIMEOUT="30",
        READY_INTERVAL="0.2",
    )

    assert started.returncode == 0, started.output
    assert wait_status(app, "78-1") == "done 1"
    assert "interrupted by SIGTERM" in (app / ".deploy" / "78-1.log").read_text()
    assert env_value(app, "IMAGE_TAG") == OLD


def test_detach_refuses_a_reused_id_and_bad_input(app, shell):
    first = shell.run("deploy.sh", "--detach", "79-1", NEW, cwd=app, **FAST)
    assert first.returncode == 0, first.output
    wait_status(app, "79-1")

    again = shell.run("deploy.sh", "--detach", "79-1", NEW, cwd=app, **FAST)
    assert again.returncode == 1
    assert "already" in again.output

    for args in (["--detach", "../x", NEW], ["--detach", "80-1", "sha-1; reboot"]):
        bad = shell.run("deploy.sh", *args, cwd=app, **FAST)
        assert bad.returncode == 1
        assert not (app / ".deploy" / "80-1.status").exists()


@pytest.mark.skipif(shutil.which("flock") is None, reason="needs flock (util-linux)")
def test_second_deploy_waits_for_the_lock_then_gives_up(app, shell):
    import fcntl

    with open(app / ".deploy.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        result = shell.run("deploy.sh", NEW, cwd=app, DEPLOY_LOCK_TIMEOUT="1", **FAST)

    assert result.returncode == 1
    assert "another deploy is still running" in result.output
    assert not [c for c in result.calls if c.startswith("docker ")]


# ----- order of versions (s7-review I-2) ---------------------------------------------------


def test_older_commit_is_not_deployed_over_a_newer_one(app, shell):
    (app / ".env").write_text(env_file(NEW), newline="\n")
    (shell.state / "running").write_text(NEW + "\n")

    result = shell.run(
        "deploy.sh",
        OLD,
        OLD_SHA,
        cwd=app,
        FAKE_GIT_HEAD=NEW_SHA,
        FAKE_GIT_ANCESTORS="1111111:2222222",
        **FAST,
    )

    assert result.returncode == 3
    assert "older than the running" in result.output
    assert ups(result.calls) == []
    assert env_value(app, "IMAGE_TAG") == NEW
    assert result.file("head") == NEW_SHA


def test_older_commit_is_deployed_when_asked_for(app, shell):
    (app / ".env").write_text(env_file(NEW), newline="\n")
    (shell.state / "running").write_text(NEW + "\n")

    result = shell.run(
        "deploy.sh",
        "--allow-older",
        OLD,
        OLD_SHA,
        cwd=app,
        FAKE_GIT_HEAD=NEW_SHA,
        FAKE_GIT_ANCESTORS="1111111:2222222",
        **FAST,
    )

    assert result.returncode == 0, result.output
    assert env_value(app, "IMAGE_TAG") == OLD
    assert result.file("head") == OLD_SHA


def test_newer_commit_is_deployed(app, shell):
    result = shell.run(
        "deploy.sh",
        NEW,
        NEW_SHA,
        cwd=app,
        FAKE_GIT_HEAD=OLD_SHA,
        FAKE_GIT_ANCESTORS="1111111:2222222",
        **FAST,
    )

    assert result.returncode == 0, result.output
    assert env_value(app, "IMAGE_TAG") == NEW
