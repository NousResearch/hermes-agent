"""Regression tests for session-snapshot file permissions.

``init_session()`` runs ``export -p > {snapshot}`` to capture the login shell
environment, and the snapshot lives in a shared temp dir (``/tmp`` on the
default local backend).  ``export -p`` dumps every exported env var — including
secrets that are NOT in the provider blocklist (SUDO_PASSWORD, the general AWS
credential chain, user-set secrets).  With the inherited default umask
(typically 022 -> mode 0644) any other local user on a shared host could read
``/tmp/hermes-snap-*.sh`` and harvest those secrets for the whole session.

The fix forces ``umask 077`` before the snapshot / cwd files are created, so
they land 0600 (owner-only).  These tests pin both the wrapper-string contract
and the real on-disk file mode produced by the local backend.
"""

import os
import stat

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments.local import LocalEnvironment


class _TestableEnv(BaseEnvironment):
    """Concrete subclass for exercising base-class wrapping in isolation."""

    def __init__(self, cwd="/tmp", timeout=10):
        super().__init__(cwd=cwd, timeout=timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError("Use mock")

    def cleanup(self):
        pass


class TestWrapCommandUmask:
    def test_snapshot_redump_is_umask_protected(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hello", "/tmp")

        # The env re-dump must run under a restrictive umask so a snapshot
        # recreated mid-session isn't left world-readable.
        assert f"(umask 077; export -p > {env._snapshot_path})" in wrapped
        # ...and so must the cwd-file write.
        assert f"(umask 077; pwd -P > {env._cwd_file})" in wrapped

    def test_umask_does_not_leak_into_user_command(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("touch out.txt", "/tmp")

        # The user's command runs before the snapshot bookkeeping and must not
        # inherit the 0600 umask — the umask is scoped to its own subshell.
        user_line = wrapped.split("eval ", 1)[1].splitlines()[0]
        assert "umask" not in user_line


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
class TestSnapshotFileMode:
    @pytest.fixture(autouse=True)
    def _isolate_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "logs").mkdir(exist_ok=True)

    def test_local_snapshot_is_owner_only(self, tmp_path, monkeypatch):
        # A secret that is NOT in the provider blocklist still reaches the
        # snapshot via ``export -p`` — exactly the exposure being closed here.
        monkeypatch.setenv("SUDO_PASSWORD", "hunter2-should-not-leak")
        # Force a permissive umask so a missing fix would clearly produce a
        # world-readable file rather than relying on the test runner's umask.
        old_umask = os.umask(0o022)
        try:
            env = LocalEnvironment(cwd=str(tmp_path), timeout=20)
        finally:
            os.umask(old_umask)

        try:
            assert env._snapshot_ready, "snapshot must be created for this test"
            assert os.path.exists(env._snapshot_path)

            snap_mode = stat.S_IMODE(os.stat(env._snapshot_path).st_mode)
            assert snap_mode == 0o600, f"snapshot mode {oct(snap_mode)} is not owner-only"
            assert not (snap_mode & (stat.S_IRWXG | stat.S_IRWXO)), "snapshot is group/other accessible"

            if os.path.exists(env._cwd_file):
                cwd_mode = stat.S_IMODE(os.stat(env._cwd_file).st_mode)
                assert not (cwd_mode & (stat.S_IRWXG | stat.S_IRWXO)), "cwd file is group/other accessible"
        finally:
            env.cleanup()

    def test_snapshot_mode_survives_a_command(self, tmp_path):
        env = LocalEnvironment(cwd=str(tmp_path), timeout=20)
        try:
            env.execute("export NEW_VAR=1")  # triggers the per-command re-dump
            assert os.path.exists(env._snapshot_path)
            snap_mode = stat.S_IMODE(os.stat(env._snapshot_path).st_mode)
            assert not (snap_mode & (stat.S_IRWXG | stat.S_IRWXO)), (
                f"snapshot became group/other accessible after a command: {oct(snap_mode)}"
            )
        finally:
            env.cleanup()
