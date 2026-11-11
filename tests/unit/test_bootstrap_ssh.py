"""bootstrap-server.sh's SSH and firewall steps against fake sshd/systemctl/ufw (shell_fakes):
password login is switched off only when a HUMAN can still get in as root with a key (the CI
key DEPLOY_PUBKEY does not count: `deploy` has no sudo), a configuration sshd rejects is
removed again before anything reloads, and the firewall allows SSH before it is switched on
(s7-review I-3, Minor-5)."""

from pathlib import Path

import pytest
from shell_fakes import ShellHarness, find_bash, posix

BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash (Git's bash on Windows)")

CI_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICIcIcIcIcIcIcIcIcIcIcIcIc github-actions"
ADMIN_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAdMiNaDmInAdMiNaDmInAdMiN owner@laptop"


@pytest.fixture
def server(tmp_path: Path) -> dict:
    """A fake root: root's authorized_keys, the sshd drop-in directory, deploy's home."""
    root = tmp_path / "root-ssh"
    dropin = tmp_path / "sshd_config.d"
    home = tmp_path / "home-deploy"
    for d in (root, dropin, home / ".ssh"):
        d.mkdir(parents=True)
    return {"keys": root / "authorized_keys", "dropin": dropin, "home": home}


@pytest.fixture
def shell(tmp_path: Path, server) -> ShellHarness:
    harness = ShellHarness(tmp_path, BASH)
    # getent passwd deploy -> a home directory under tmp (create_user put the CI key there).
    getent = harness.bin / "getent"
    getent.write_text(
        f'#!/usr/bin/env bash\necho "deploy:x:1000:1000::{posix(server["home"])}:/bin/bash"\n',
        newline="\n",
    )
    getent.chmod(0o755)
    (server["home"] / ".ssh" / "authorized_keys").write_text(CI_KEY + "\n", newline="\n")
    return harness


def harden(shell, server, tmp_path, **env):
    return shell.call(
        "bootstrap-server.sh",
        "harden_ssh",
        cwd=tmp_path,
        ROOT_KEYS=posix(server["keys"]),
        SSHD_DROPIN_DIR=posix(server["dropin"]),
        DEPLOY_PUBKEY=CI_KEY,
        **env,
    )


def hardening(server) -> Path:
    return server["dropin"] / "00-hardening.conf"


@pytest.mark.parametrize(
    "root_keys",
    [None, "", CI_KEY + "\n", "# comment\n\n" + CI_KEY + "\n"],
    ids=["no-file", "empty", "ci-key", "comment-and-ci-key"],
)
def test_only_the_ci_key_leaves_password_login_on(shell, server, tmp_path, root_keys):
    if root_keys is not None:
        server["keys"].write_text(root_keys, newline="\n")

    result = harden(shell, server, tmp_path)

    assert result.returncode == 0, result.output
    assert not hardening(server).exists()
    assert not [c for c in result.calls if c.startswith(("sshd", "systemctl"))]
    assert "password login left ON" in result.output


def test_an_admin_key_switches_password_login_off(shell, server, tmp_path):
    server["keys"].write_text(ADMIN_KEY + "\n" + CI_KEY + "\n", newline="\n")

    result = harden(shell, server, tmp_path)

    assert result.returncode == 0, result.output
    text = hardening(server).read_text()
    assert "PasswordAuthentication no" in text
    assert "PermitRootLogin prohibit-password" in text
    assert result.calls == ["sshd -t", "systemctl reload ssh"]


def test_admin_pubkey_is_added_to_root_once(shell, server, tmp_path):
    for _ in range(2):
        result = shell.call(
            "bootstrap-server.sh",
            "add_admin_key",
            cwd=tmp_path,
            ROOT_KEYS=posix(server["keys"]),
            ADMIN_PUBKEY=ADMIN_KEY,
        )
        assert result.returncode == 0, result.output

    assert server["keys"].read_text().splitlines() == [ADMIN_KEY]


def test_config_rejected_by_sshd_is_removed_and_nothing_reloads(shell, server, tmp_path):
    server["keys"].write_text(ADMIN_KEY + "\n", newline="\n")

    result = harden(shell, server, tmp_path, FAKE_SSHD_FAIL="1")

    assert result.returncode != 0
    assert not hardening(server).exists()
    assert "systemctl reload ssh" not in result.calls


def test_firewall_allows_ssh_before_it_is_enabled(shell, tmp_path):
    result = shell.call("bootstrap-server.sh", "setup_firewall", cwd=tmp_path)

    assert result.returncode == 0, result.output
    calls = [c.removeprefix("ufw ") for c in result.calls]
    assert calls.index("allow 22/tcp") < calls.index("--force enable")
    assert calls[-1] == "--force enable"
