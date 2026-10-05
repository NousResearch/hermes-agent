"""Regression tests for issue #133436.

A named profile re-homes the scratch TMPDIR to ``~/.hermes/profiles/<name>/cache/scratch``,
so ``<scratch>/hermes-ssh/<16hex>.sock`` plus OpenSSH's 17-byte ControlMaster suffix blows
the AF_UNIX ``sun_path`` cap and every terminal call fails with
``unix_listener: path ... too long for Unix domain socket``. The control dir must move to a
short per-user root — and, because the name is predictable, must never be adopted from an
untrusted shape (symlink / foreign uid / non-directory; #80284, #127862).
"""

import os
import shutil
import stat as stat_module
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tools.environments.ssh as ssh_env
from tools.environments.ssh import SSHEnvironment, _ensure_owned_dir

# OpenSSH appends ``.XXXXXXXXXXXXXXXX`` (17 bytes) to the ControlPath in ControlMaster mode;
# the macOS sun_path field is 104 bytes including the NUL terminator (103 usable, the stricter
# of the two platforms we support).
_SSH_CONTROLMASTER_SUFFIX = 17
_MAX_SUN_PATH = 103

_POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt", reason="AF_UNIX ControlMaster paths are POSIX-only"
)


def _mock_connection(monkeypatch):
    monkeypatch.setattr(
        "tools.environments.ssh.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 0),
    )
    monkeypatch.setattr(
        "tools.environments.ssh.subprocess.Popen",
        lambda *a, **k: MagicMock(stdout=iter([]), stderr=iter([]), stdin=MagicMock()),
    )
    monkeypatch.setattr("tools.environments.base.time.sleep", lambda _: None)


def _sun_path_ok(socket: Path) -> bool:
    return len(str(socket)) + _SSH_CONTROLMASTER_SUFFIX <= _MAX_SUN_PATH


@pytest.fixture
def _clean_fallback_dir():
    yield
    shutil.rmtree(Path("/tmp") / f"hermes-ssh-{os.geteuid()}", ignore_errors=True)


@_POSIX_ONLY
def test_deep_profile_scratch_falls_back_under_sun_path(
    monkeypatch, _clean_fallback_dir
):
    """#133436: the issue's 61-byte profile scratch must not back the socket path."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    deep = "/home/raffymontemayor/.hermes/profiles/video-agent/cache/scratch"
    monkeypatch.setattr(ssh_env.tempfile, "gettempdir", lambda: deep)
    _mock_connection(monkeypatch)

    env = SSHEnvironment(host="h", user="u", port=22)

    assert deep not in str(env.control_dir)
    assert env.control_dir == Path("/tmp") / f"hermes-ssh-{os.geteuid()}"
    assert _sun_path_ok(env.control_socket)


@_POSIX_ONLY
def test_xdg_runtime_dir_is_preferred(monkeypatch, tmp_path):
    """A set $XDG_RUNTIME_DIR (per-user 0700 tmpfs) wins and gets a 0700 hermes-ssh inside."""
    xdg = tmp_path / "run-user"
    xdg.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))
    monkeypatch.setattr(
        ssh_env.tempfile,
        "gettempdir",
        lambda: "/var/folders/2t/wbkw5yb158jc3zhswgl7tz9c0000gn/T",
    )
    _mock_connection(monkeypatch)

    env = SSHEnvironment(host="h", user="u")

    assert env.control_dir == xdg / "hermes-ssh"
    assert stat_module.S_IMODE(env.control_dir.lstat().st_mode) == 0o700


@_POSIX_ONLY
def test_short_enough_scratch_dir_keeps_scratch_root(monkeypatch, tmp_path):
    """A short scratch root keeps today's (already per-user, policy-managed) location."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(ssh_env.tempfile, "gettempdir", lambda: str(tmp_path))
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "SOCKET_TMPDIR_MAX_LEN", 4096)
    _mock_connection(monkeypatch)

    env = SSHEnvironment(host="h", user="u")

    assert env.control_dir == tmp_path / "hermes-ssh"


@_POSIX_ONLY
def test_refuses_symlinked_control_dir(tmp_path):
    """A symlink planted at the predictable name must not be adopted (#80284, #127862)."""
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    link = tmp_path / "hermes-ssh"
    link.symlink_to(attacker)

    with pytest.raises(PermissionError):
        _ensure_owned_dir(link)


@_POSIX_ONLY
def test_refuses_non_directory_placeholder(tmp_path):
    placeholder = tmp_path / "hermes-ssh"
    placeholder.write_text("not a directory")

    with pytest.raises(OSError):
        _ensure_owned_dir(placeholder)


@_POSIX_ONLY
def test_dir_owned_by_another_uid_is_rejected(monkeypatch, tmp_path):
    target = tmp_path / "hermes-ssh"
    target.mkdir(mode=0o700)
    monkeypatch.setattr(ssh_env.os, "geteuid", lambda: 4242)

    with pytest.raises(PermissionError):
        _ensure_owned_dir(target)


@_POSIX_ONLY
def test_existing_owned_dir_is_tightened_to_0700(tmp_path):
    """A dir created by an older version without a mode is reused but chmod'd 0700."""
    target = tmp_path / "hermes-ssh"
    target.mkdir(mode=0o755)

    assert _ensure_owned_dir(target) == target
    assert stat_module.S_IMODE(target.lstat().st_mode) == 0o700
