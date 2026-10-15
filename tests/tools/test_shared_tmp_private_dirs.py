"""Per-session socket/control dirs live in a SHARED temp root (``/tmp``) under predictable names.
A directory there is used only once it is proven ours (real dir, our uid, owner-only), and marker
files written into it never follow a planted symlink. ``tmp_path`` stands in for the shared root."""

import os
import stat

import pytest

import tools.browser_tool as bt
import tools.environments.ssh as ssh_mod
from tools import browser_tool_lifecycle as bt_lifecycle
from tools import browser_tool_session as bt_session


def _browser_entry(root, monkeypatch):
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(root))
    return (lambda: root / "agent-browser-h_probe",
            lambda: bt_session._prepare_session_socket_dir("h_probe"))


def _ssh_entry(root, monkeypatch):
    monkeypatch.setattr(ssh_mod, "socket_safe_tmpdir", lambda: str(root), raising=False)
    monkeypatch.setattr(ssh_mod.tempfile, "gettempdir", lambda: str(root))
    monkeypatch.setattr(ssh_mod, "_ensure_ssh_available", lambda: None)
    monkeypatch.setattr(ssh_mod.SSHEnvironment, "_establish_connection", lambda self: None)
    return (lambda: root / f"hermes-ssh-{os.getuid()}",  # windows-footgun: ok — posix-only test
            lambda: ssh_mod.SSHEnvironment(host="h", user="u", probe_only=True).control_dir)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("entry", [_browser_entry, _ssh_entry], ids=["browser", "ssh"])
@pytest.mark.parametrize("planted", ["symlink", "foreign_owner", "loose_mode"])
def test_shared_tmp_dir_is_used_only_when_proven_ours(tmp_path, monkeypatch, entry, planted):
    root = tmp_path / "shared-tmp"
    victim = tmp_path / "victim"
    root.mkdir()
    victim.mkdir()
    planted_path, use_dir = entry(root, monkeypatch)
    if planted == "foreign_owner":
        real_uid = os.getuid()  # windows-footgun: ok — posix-only test
        monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    target = planted_path()
    if planted == "symlink":
        target.symlink_to(victim, target_is_directory=True)
    else:
        target.mkdir()
        target.chmod(0o777)

    if planted == "loose_mode":
        use_dir()
        assert stat.S_IMODE(os.lstat(target).st_mode) == 0o700
    else:
        with pytest.raises(PermissionError):
            use_dir()
        assert not any(target.iterdir()) and not any(victim.iterdir())


@pytest.mark.platforms("posix")
def test_owner_pid_write_replaces_a_planted_symlink(tmp_path):
    socket_dir = tmp_path / "agent-browser-h_probe"
    socket_dir.mkdir(mode=0o700)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    (socket_dir / "h_probe.owner_pid").symlink_to(victim)

    bt_lifecycle._write_owner_pid(str(socket_dir), "h_probe")

    assert victim.read_text(encoding="utf-8-sig") == "keep"
    marker = socket_dir / "h_probe.owner_pid"
    assert not marker.is_symlink()
    assert marker.read_text(encoding="utf-8-sig") == str(os.getpid())
