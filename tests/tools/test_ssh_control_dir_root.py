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
    monkeypatch.setattr(ssh_mod.tempfile, "gettempdir", lambda: str(root))
    monkeypatch.setattr(ssh_mod, "socket_safe_tmpdir", lambda: str(root), raising=False)
    monkeypatch.setattr(ssh_mod, "_SHORT_TMP_ROOT", str(root), raising=False)
    monkeypatch.setattr(ssh_mod, "_CONTROL_DIR_MAX_BYTES", 4096, raising=False)  # tmp_path is deep
    monkeypatch.setattr(ssh_mod, "_ensure_ssh_available", lambda: None)
    monkeypatch.setattr(ssh_mod.SSHEnvironment, "_establish_connection", lambda self: None)
    return root


@pytest.mark.platforms("posix")
def test_every_candidate_leaves_room_for_the_openssh_listener(tmp_path, monkeypatch):
    deep = tmp_path / ("x" * 80)
    deep.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(deep))
    monkeypatch.setattr(ssh_mod.tempfile, "gettempdir", lambda: str(deep))
    monkeypatch.setattr(ssh_mod, "socket_safe_tmpdir", lambda: str(deep), raising=False)
    monkeypatch.setattr(ssh_mod, "ensure_owned_private_dir", os.fspath, raising=False)  # no real /tmp dir
    monkeypatch.setattr(ssh_mod, "_ensure_ssh_available", lambda: None)
    monkeypatch.setattr(ssh_mod.SSHEnvironment, "_establish_connection", lambda self: None)

    env = ssh_mod.SSHEnvironment(host="9373:9b91:4480:558d:708e:e601:24e8:d8d0", user="u", probe_only=True)

    listener = f"{env.control_socket}.{'a' * 16}"  # OpenSSH's temporary ControlMaster listener
    assert len(os.fsencode(listener)) <= 103


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
    # Every name the backend could pick in the shared root, current and pre-per-user.
    squatted = [shared_root / "hermes-ssh", shared_root / f"hermes-ssh-{os.getuid()}"]  # windows-footgun: ok
    for path in squatted:
        if planted == "symlink":
            path.symlink_to(victim, target_is_directory=True)
        elif planted == "plain_file":
            path.write_text("x", encoding="utf-8")
        else:
            path.mkdir()
            path.chmod(0o777)

    control_dir = _control_dir()

    assert control_dir not in squatted and control_dir.parent == shared_root
    assert stat.S_ISDIR(os.lstat(control_dir).st_mode) and _mode(control_dir) == 0o700
    assert not any(victim.iterdir())
    if planted == "foreign_owner":
        assert not any(any(path.iterdir()) for path in squatted)


@pytest.mark.platforms("posix")
def test_cleanup_removes_only_a_fallback_dir_it_created(shared_root):
    for name in ("hermes-ssh", f"hermes-ssh-{os.getuid()}"):  # windows-footgun: ok — posix-only test
        (shared_root / name).write_text("x", encoding="utf-8")  # squat both names: forces mkdtemp
    fallback = ssh_mod.SSHEnvironment(host="h", user="u", probe_only=True)
    fallback.cleanup()
    (shared_root / f"hermes-ssh-{os.getuid()}").unlink()  # windows-footgun: ok
    shared = ssh_mod.SSHEnvironment(host="h", user="u", probe_only=True)
    shared.cleanup()

    assert not fallback.control_dir.exists()
    assert shared.control_dir.is_dir()  # a per-user dir other instances share is kept


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
