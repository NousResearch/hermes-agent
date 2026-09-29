"""Per-session socket/control dirs live in a SHARED temp root (``/tmp``) under predictable names.
A directory there is used only once it is proven ours (real dir, our uid, owner-only), and marker
files written into it never follow a planted symlink. ``tmp_path`` stands in for the shared root."""

import os
import stat

import pytest

import tools.environments.ssh as ssh_mod


def _ssh_entry(root, monkeypatch):
    monkeypatch.setattr(ssh_mod, "socket_safe_tmpdir", lambda: str(root), raising=False)
    monkeypatch.setattr(ssh_mod.tempfile, "gettempdir", lambda: str(root))
    monkeypatch.setattr(ssh_mod, "_ensure_ssh_available", lambda: None)
    monkeypatch.setattr(ssh_mod.SSHEnvironment, "_establish_connection", lambda self: None)
    return (lambda: root / f"hermes-ssh-{os.getuid()}",  # windows-footgun: ok — posix-only test
            lambda: ssh_mod.SSHEnvironment(host="h", user="u", probe_only=True).control_dir)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("entry", [_ssh_entry], ids=["ssh"])
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
