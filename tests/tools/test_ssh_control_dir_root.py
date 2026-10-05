"""ControlMaster listeners must fit even when profile scratch or XDG roots are long.

Related: #133436, salvage of #133444 (5d615ce). Tests only remove their own directories.
"""

import os
import socket
import stat
import tempfile
from pathlib import Path

import pytest

from hermes_constants import socket_safe_tmpdir
from tools.environments import ssh as ssh_env
from tools.environments.base import EnvironmentConnectionError


@pytest.fixture
def short_root(monkeypatch):
    # tmp_path itself can be too long for a real AF_UNIX listener.
    with tempfile.TemporaryDirectory(prefix="hssh-", dir=socket_safe_tmpdir()) as root:
        root = Path(root)
        monkeypatch.setattr(ssh_env, "_CONTROL_FALLBACK_ROOT", root, raising=False)
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        yield root


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("candidate", ["scratch", "xdg", "unicode"])
def test_long_root_falls_back_to_bindable_listener(monkeypatch, tmp_path, short_root, candidate):
    long_root = short_root / ("界" * 15) if candidate == "unicode" else tmp_path / ("x" * 100)
    long_root.mkdir()
    monkeypatch.setattr(ssh_env.tempfile, "gettempdir", lambda: str(long_root))
    if candidate == "xdg":
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(long_root))
    control_dir = ssh_env._open_control_dir()
    listener = control_dir / ("a" * 16 + ".sock." + "b" * 16)
    assert len(os.fsencode(listener)) <= 103
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(listener))
    assert stat.S_IMODE(control_dir.stat().st_mode) == 0o700


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("candidate", ["scratch", "xdg"])
def test_short_root_is_reused(monkeypatch, short_root, candidate):
    monkeypatch.setattr(ssh_env.tempfile, "gettempdir", lambda: str(short_root))
    if candidate == "xdg":
        xdg = short_root / "xdg"
        xdg.mkdir()
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))
        expected = xdg / "hermes-ssh"
    else:
        expected = short_root / "hermes-ssh"
    assert ssh_env._open_control_dir() == expected
    assert ssh_env._open_control_dir() == expected


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shape", ["symlink", "file", "foreign", "owned"])
def test_control_dir_ownership(monkeypatch, tmp_path, shape):
    target = tmp_path / "control"
    if shape == "symlink":
        real = tmp_path / "other"
        real.mkdir()
        target.symlink_to(real)
    elif shape == "file":
        target.write_text("placeholder")
    else:
        target.mkdir(mode=0o755)
        if shape == "foreign":
            monkeypatch.setattr(ssh_env.os, "geteuid", lambda: target.stat().st_uid + 1)
    if shape == "owned":
        assert ssh_env._ensure_owned_dir(target) == target
        assert stat.S_IMODE(target.stat().st_mode) == 0o700
    else:
        with pytest.raises(OSError):
            ssh_env._ensure_owned_dir(target)


@pytest.mark.platforms("posix")
def test_unusable_roots_report_connection_error(monkeypatch, tmp_path):
    deep = tmp_path / ("x" * 100)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(ssh_env.tempfile, "gettempdir", lambda: str(deep))
    monkeypatch.setattr(ssh_env, "_CONTROL_FALLBACK_ROOT", deep)
    with pytest.raises(EnvironmentConnectionError, match="No usable SSH control-socket directory"):
        ssh_env._open_control_dir()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("deep", [False, True])
def test_real_ssh_with_short_and_long_temp_roots(monkeypatch, tmp_path, short_root, deep):
    host = os.getenv("HERMES_TEST_SSH_HOST")
    user = os.getenv("HERMES_TEST_SSH_USER")
    if not host or not user:
        pytest.skip("requires the disposable sshd from scripts/ci/start_test_sshd.sh")
    root = tmp_path / ("x" * 100) if deep else short_root
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(ssh_env.tempfile, "gettempdir", lambda: str(root))
    env = ssh_env.SSHEnvironment(
        host=host, user=user, port=int(os.getenv("HERMES_TEST_SSH_PORT", "22")),
        key_path=os.getenv("HERMES_TEST_SSH_KEY", ""), probe_only=True,
    )
    try:
        first = env._run_ssh("printf socket-ok", timeout=10)
        second = env._run_ssh("printf reused-ok", timeout=10)
        assert first.returncode == second.returncode == 0
        assert (first.stdout, second.stdout) == ("socket-ok", "reused-ok")
        assert env.control_socket.is_socket()
    finally:
        env.cleanup()
    assert not env.control_socket.exists()
