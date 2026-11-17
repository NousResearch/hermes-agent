"""The SSH ControlMaster dir sits under a predictable name, in the shared ``/tmp`` whenever the
scratch TMPDIR is too deep for ``sun_path`` (macOS always). It is used only once proven ours (a
real directory, our uid, owner-only); a squatted name never hosts our master socket and never
takes the SSH backend down (#80284, #133436). ``tmp_path`` stands in for the shared root."""

import gc
import os
import stat
import tempfile
from pathlib import Path

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


@pytest.fixture
def make_env():
    """Keep each env alive until teardown: a collected fallback env takes its dir with it."""
    envs = []
    def make():
        envs.append(ssh_mod.SSHEnvironment(host="h", user="u", probe_only=True))
        return envs[-1]
    yield make
    for env in envs:
        env.cleanup()


def _mode(path):
    return stat.S_IMODE(os.lstat(path).st_mode)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("planted", ["symlink", "foreign_owner", "plain_file"])
def test_squatted_name_is_refused_and_backend_still_gets_a_private_dir(
        shared_root, tmp_path, monkeypatch, make_env, planted):
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

    control_dir = make_env().control_dir

    assert control_dir not in squatted and control_dir.parent == shared_root
    assert stat.S_ISDIR(os.lstat(control_dir).st_mode) and _mode(control_dir) == 0o700
    assert not any(victim.iterdir())
    if planted == "foreign_owner":
        assert not any(any(path.iterdir()) for path in squatted)


@pytest.mark.platforms("posix")
def test_fallback_dir_outlives_cleanup_and_goes_with_its_env(shared_root, monkeypatch):
    monkeypatch.setattr(ssh_mod.logger, "disabled", True)  # a captured record's exc traceback pins the env
    for name in ("hermes-ssh", f"hermes-ssh-{os.getuid()}"):  # windows-footgun: ok — posix-only test
        (shared_root / name).write_text("x", encoding="utf-8")  # squat both names: forces mkdtemp
    fallback = ssh_mod.SSHEnvironment(host="h", user="u", probe_only=True)
    fallback_dir = fallback.control_dir
    fallback.cleanup()
    assert fallback_dir.is_dir()  # cleanup_stops_processes=False: a poller's next call reconnects through it
    del fallback
    gc.collect()
    (shared_root / f"hermes-ssh-{os.getuid()}").unlink()  # windows-footgun: ok
    shared = ssh_mod.SSHEnvironment(host="h", user="u", probe_only=True)
    shared_dir = shared.control_dir
    shared.cleanup()
    del shared
    gc.collect()

    assert not fallback_dir.exists()
    assert shared_dir.is_dir()  # a per-user dir other instances share is kept


@pytest.mark.platforms("posix")
def test_last_cleanup_of_a_cyclic_env_still_has_its_fallback_dir(shared_root, monkeypatch):
    """A synced env sits in a cycle (FileSyncManager holds its bound methods), so the cyclic GC
    frees it, and its final cleanup() syncs back through the ControlPath: the dir must outlive it."""
    monkeypatch.setattr(ssh_mod.logger, "disabled", True)  # a captured record's exc traceback pins the env
    for name in ("hermes-ssh", f"hermes-ssh-{os.getuid()}"):  # windows-footgun: ok — posix-only test
        (shared_root / name).write_text("x", encoding="utf-8")  # squat both names: forces mkdtemp
    env = ssh_mod.SSHEnvironment(host="h", user="u", probe_only=True)
    fallback_dir, seen = env.control_dir, []

    class SyncBack:
        def __init__(self, upload_fn):
            self.upload_fn = upload_fn  # a bound method of the env, like FileSyncManager's callbacks

        def sync_back(self):
            seen.append(fallback_dir.is_dir())

    env._sync_manager = SyncBack(env._run_ssh)
    del env
    gc.collect()

    assert seen and all(seen)
    assert not fallback_dir.exists()


@pytest.mark.platforms("posix")
def test_failed_connection_does_not_leave_a_fallback_dir_behind(shared_root, monkeypatch):
    for name in ("hermes-ssh", f"hermes-ssh-{os.getuid()}"):  # windows-footgun: ok — posix-only test
        (shared_root / name).write_text("x", encoding="utf-8")  # squat both names: forces mkdtemp
    def refuse(self):
        raise ssh_mod.EnvironmentConnectionError("unreachable")
    monkeypatch.setattr(ssh_mod.SSHEnvironment, "_establish_connection", refuse)

    with pytest.raises(ssh_mod.EnvironmentConnectionError):
        ssh_mod.SSHEnvironment(host="h", user="u")

    assert sorted(p.name for p in shared_root.iterdir()) == sorted(
        ("hermes-ssh", f"hermes-ssh-{os.getuid()}"))  # windows-footgun: ok


@pytest.mark.platforms("posix")
def test_own_dir_left_open_by_an_older_version_is_tightened(shared_root, make_env):
    ours = shared_root / f"hermes-ssh-{os.getuid()}"  # windows-footgun: ok — posix-only test
    ours.mkdir()
    ours.chmod(0o777)

    assert make_env().control_dir == ours
    assert _mode(ours) == 0o700


@pytest.mark.platforms("posix")
def test_private_xdg_runtime_dir_is_preferred_over_the_shared_root(shared_root, tmp_path, monkeypatch, make_env):
    xdg = tmp_path / "run-user"
    xdg.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))

    control_dir = make_env().control_dir

    assert control_dir == xdg / "hermes-ssh"
    assert _mode(control_dir) == 0o700
    assert not any(shared_root.iterdir())


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("mode", [0o755, 0o1777])
def test_shared_xdg_runtime_dir_is_not_trusted(shared_root, tmp_path, monkeypatch, make_env, mode):
    """A mis-set ``$XDG_RUNTIME_DIR`` naming a group/world-accessible dir (sudo -E, a shared /run
    path) is skipped like the gateway's control socket skips it: other users can see or pre-create
    names in it, so it is no stronger than the shared root it would otherwise outrank."""
    xdg = tmp_path / "shared-run"
    xdg.mkdir()
    xdg.chmod(mode)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))

    control_dir = make_env().control_dir

    assert not (xdg / "hermes-ssh").exists()
    assert control_dir == shared_root / f"hermes-ssh-{os.getuid()}"  # windows-footgun: ok — posix-only test


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("suffix", ["/", "/."])
def test_trailing_component_does_not_trust_symlinked_xdg_runtime_dir(monkeypatch, suffix):
    """POSIX lstat() follows a final symlink with / or /. appended; a mutable
    shared-parent link must not masquerade as a real private runtime directory."""
    # Keep both candidates under the actual 64-byte macOS ControlPath budget.
    with tempfile.TemporaryDirectory(prefix="hxdg-", dir="/tmp") as root_name:
        root = Path(root_name)
        private = root / "private"
        private.mkdir(mode=0o700)
        private.chmod(0o700)
        link = root / "runtime-link"
        link.symlink_to(private, target_is_directory=True)
        monkeypatch.setenv("XDG_RUNTIME_DIR", f"{link}{suffix}")
        monkeypatch.setattr(ssh_mod, "socket_safe_tmpdir", lambda: str(root))
        monkeypatch.setattr(ssh_mod, "_SHORT_TMP_ROOT", str(root))
        monkeypatch.setattr(ssh_mod, "_ensure_ssh_available", lambda: None)
        monkeypatch.setattr(ssh_mod.SSHEnvironment, "_establish_connection", lambda self: None)
        assert len(os.fsencode(link / "hermes-ssh")) <= ssh_mod._CONTROL_DIR_MAX_BYTES

        env = ssh_mod.SSHEnvironment(host="h", user="u", probe_only=True)
        try:
            assert env.control_dir == root / f"hermes-ssh-{os.getuid()}"  # windows-footgun: ok — posix-only test
            assert not (private / "hermes-ssh").exists()
        finally:
            env.cleanup()


@pytest.mark.platforms("posix")
def test_xdg_runtime_dir_owned_by_another_user_is_not_trusted(shared_root, tmp_path, monkeypatch, make_env):
    """``sudo -E`` keeps the invoking user's ``$XDG_RUNTIME_DIR``: 0700, but its owner is not us and can
    swap a ``hermes-ssh`` leaf we create there."""
    xdg = tmp_path / "run-user"
    xdg.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))
    real_uid = os.getuid()  # windows-footgun: ok — posix-only test
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

    control_dir = make_env().control_dir

    assert not (xdg / "hermes-ssh").exists()
    assert control_dir.parent == shared_root
