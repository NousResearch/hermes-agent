"""Regression tests for inaccessible-cwd fallback (#65583).

``/root`` leaking into a non-root gateway/cron process's terminal cwd used to
kill every command with ``PermissionError: [Errno 13] Permission denied:
'/root'`` — ``os.path.isdir('/root')`` is True for a non-root user (stat only
needs search permission on ``/``), so the old existence-only check in
``_resolve_safe_cwd`` happily returned a directory ``subprocess.Popen`` could
not enter. The fix checks X_OK and falls back to a writable ancestor when the
configured path itself is missing or unenterable.
"""

import os
import sys
import tempfile

import pytest

from tools.environments.local import (
    LocalEnvironment,
    _cwd_fallback_usable,
    _cwd_usable,
    _resolve_safe_cwd,
)


@pytest.fixture
def denied_dir(tmp_path):
    """A directory that exists but cannot be entered (simulates /root)."""
    d = tmp_path / "rootlike"
    d.mkdir()
    d.chmod(0)
    yield d
    d.chmod(0o755)  # so pytest can clean up


needs_posix_perms = pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="chmod-based access denial needs POSIX + non-root",
)


@needs_posix_perms
class TestInaccessibleCwdFallback:
    def test_cwd_usable_rejects_unenterable_directory(self, denied_dir):
        assert os.path.isdir(denied_dir)  # the trap: stat succeeds
        assert _cwd_usable(str(denied_dir)) is False


    def test_resolve_safe_cwd_climbs_past_denied_ancestor(self, denied_dir, tmp_path):
        missing_child = str(denied_dir / "sub" / "dir")
        resolved = _resolve_safe_cwd(missing_child)
        assert os.access(resolved, os.X_OK)
        assert resolved == str(tmp_path)

    def test_local_environment_survives_denied_cwd(self, denied_dir):
        """The #65583 shape: env constructed with an unenterable cwd must
        still execute commands instead of raising PermissionError."""
        env = LocalEnvironment(cwd=str(denied_dir), timeout=30)
        try:
            handle = env._run_bash("echo ALIVE")
            out = handle.stdout.read() if handle.stdout else b""
            rc = handle.wait(timeout=15)
            assert rc == 0
            assert b"ALIVE" in (out if isinstance(out, bytes) else out.encode())
        finally:
            try:
                env.cleanup()
            except Exception:
                pass


class TestWritableCwdFallback:
    def test_resolve_safe_cwd_skips_searchable_but_non_writable_directory(
        self, tmp_path, monkeypatch
    ):
        read_only = tmp_path / "read-only"
        read_only.mkdir()
        real_access = os.access

        def access(path, mode):
            if os.fspath(path) == str(read_only) and mode & os.W_OK:
                return False
            return real_access(path, mode)

        monkeypatch.setattr(os, "access", access)

        assert real_access(read_only, os.X_OK)
        assert _cwd_usable(str(read_only)) is True
        assert _cwd_fallback_usable(str(read_only)) is False
        # Read-only but enterable configured CWDs retain their semantics.
        assert _resolve_safe_cwd(str(read_only)) == str(read_only)
        assert _resolve_safe_cwd(str(read_only / "missing")) == str(tmp_path)

    def test_local_environment_uses_resolved_fallback_in_wrapper(self, tmp_path, monkeypatch):
        read_only = tmp_path / "read-only"
        read_only.mkdir()
        missing = read_only / "missing"
        real_access = os.access

        def access(path, mode):
            if os.fspath(path) == str(read_only) and mode & os.W_OK:
                return False
            return real_access(path, mode)

        monkeypatch.setattr(os, "access", access)
        env = LocalEnvironment(cwd=str(missing), timeout=30)
        try:
            result = env.execute("pwd")
            assert result["returncode"] == 0
            assert result["output"].strip() == str(tmp_path)
        finally:
            env.cleanup()


class TestUsableCwdBehaviorUnchanged:
    def test_existing_accessible_cwd_returned_verbatim(self, tmp_path):
        assert _resolve_safe_cwd(str(tmp_path)) == str(tmp_path)


    def test_hopeless_path_falls_back_to_tempdir(self):
        # A path whose every component is missing outside any real tree.
        resolved = _resolve_safe_cwd("")
        assert resolved == tempfile.gettempdir()
