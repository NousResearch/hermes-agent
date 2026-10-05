"""The SSH ControlMaster dir sits under a predictable name, in the shared ``/tmp`` whenever the
scratch TMPDIR is too deep for ``sun_path`` (macOS always). It is used only once proven ours (a
real directory, our uid, owner-only); a squatted name never hosts our master socket and never
takes the SSH backend down (#80284, #133436). ``tmp_path`` stands in for the shared root."""

import os
import stat

import pytest

import tools.environments.ssh as ssh_mod


@pytest.fixture
def shared_root(tmp_path, monkeypatch):
    root = tmp_path / "shared-tmp"
    root.mkdir()
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(ssh_mod, "socket_safe_tmpdir", lambda: str(root))
    monkeypatch.setattr(ssh_mod, "_ensure_ssh_available", lambda: None)
    monkeypatch.setattr(ssh_mod.SSHEnvironment, "_establish_connection", lambda self: None)
    return root


def _control_dir():
    return ssh_mod.SSHEnvironment(host="h", user="u", probe_only=True).control_dir


def _mode(path):
    return stat.S_IMODE(os.lstat(path).st_mode)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("planted", ["symlink", "foreign_owner", "plain_file"])
def test_squatted_name_is_refused_and_backend_still_gets_a_private_dir(
        shared_root, tmp_path, monkeypatch, planted):
    victim = tmp_path / "victim"
    victim.mkdir()
    if planted == "foreign_owner":
        real_uid = os.getuid()  # windows-footgun: ok — posix-only test
        monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    squatted = shared_root / f"hermes-ssh-{os.getuid()}"  # windows-footgun: ok — posix-only test
    if planted == "symlink":
        squatted.symlink_to(victim, target_is_directory=True)
    elif planted == "plain_file":
        squatted.write_text("x", encoding="utf-8")
    else:
        squatted.mkdir()
        squatted.chmod(0o777)

    control_dir = _control_dir()

    assert control_dir != squatted and control_dir.parent == shared_root
    assert control_dir.name.startswith(squatted.name + "-")  # fresh mkdtemp name, unpredictable
    assert stat.S_ISDIR(os.lstat(control_dir).st_mode) and _mode(control_dir) == 0o700
    assert not any(victim.iterdir())
    if planted == "foreign_owner":
        assert not any(squatted.iterdir())


@pytest.mark.platforms("posix")
def test_own_dir_left_open_by_an_older_version_is_tightened(shared_root):
    ours = shared_root / f"hermes-ssh-{os.getuid()}"  # windows-footgun: ok — posix-only test
    ours.mkdir()
    ours.chmod(0o777)

    assert _control_dir() == ours
    assert _mode(ours) == 0o700


@pytest.mark.platforms("posix")
def test_private_xdg_runtime_dir_is_preferred_over_the_shared_root(shared_root, tmp_path, monkeypatch):
    xdg = tmp_path / "run-user"
    xdg.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))

    control_dir = _control_dir()

    assert control_dir == xdg / "hermes-ssh"
    assert _mode(control_dir) == 0o700
    assert not any(shared_root.iterdir())
