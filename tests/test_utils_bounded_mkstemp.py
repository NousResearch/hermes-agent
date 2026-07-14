"""Tests for utils.bounded_mkstemp — the bounded tempfile.mkstemp replacement.

Regression coverage for the one-shot CLI busy-spin: ``tempfile.mkstemp``
treats PermissionError as a name collision on Windows (bpo-22107) and
retries up to TMP_MAX times — 2**31-1 on Windows — so commands writing
into an ACL-denied ``~/.hermes`` (sandboxed agent shells, hardened
service accounts) burned a full CPU core for hours instead of exiting.
``bounded_mkstemp`` must fail within ``_BOUNDED_MKSTEMP_ATTEMPTS`` opens.
"""

import os
import time

import pytest

import utils
from utils import _BOUNDED_MKSTEMP_ATTEMPTS, bounded_mkstemp


class TestHappyPath:
    def test_creates_file_and_returns_open_fd(self, tmp_path):
        fd, path = bounded_mkstemp(
            dir=str(tmp_path), prefix=".manifest_", suffix=".tmp"
        )
        try:
            assert os.path.dirname(path) == str(tmp_path)
            name = os.path.basename(path)
            assert name.startswith(".manifest_")
            assert name.endswith(".tmp")
            os.write(fd, b"payload")
        finally:
            os.close(fd)
        assert (tmp_path / name).read_bytes() == b"payload"

    def test_distinct_names_across_calls(self, tmp_path):
        paths = set()
        for _ in range(16):
            fd, path = bounded_mkstemp(dir=str(tmp_path))
            os.close(fd)
            paths.add(path)
        assert len(paths) == 16

    def test_owner_only_mode_on_posix(self, tmp_path):
        if os.name == "nt":
            pytest.skip("POSIX permission bits")
        fd, path = bounded_mkstemp(dir=str(tmp_path))
        os.close(fd)
        assert os.stat(path).st_mode & 0o777 == 0o600


class TestCollisionRetry:
    def test_collision_moves_to_fresh_name(self, tmp_path, monkeypatch):
        tokens = iter([b"\x00" * 8, b"\x11" * 8])
        monkeypatch.setattr(os, "urandom", lambda n: next(tokens))
        (tmp_path / ("tmp" + (b"\x00" * 8).hex() + ".tmp")).touch()

        fd, path = bounded_mkstemp(dir=str(tmp_path))
        os.close(fd)
        assert path == str(tmp_path / ("tmp" + (b"\x11" * 8).hex() + ".tmp"))

    def test_persistent_collision_is_bounded(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def constant_urandom(n):
            calls["n"] += 1
            return b"\x00" * n

        monkeypatch.setattr(os, "urandom", constant_urandom)
        (tmp_path / ("tmp" + ("00" * 8) + ".tmp")).touch()

        with pytest.raises(FileExistsError):
            bounded_mkstemp(dir=str(tmp_path))
        assert calls["n"] == _BOUNDED_MKSTEMP_ATTEMPTS


class TestPermissionErrorBounded:
    def test_windows_semantics_fail_fast_not_tmp_max(self, tmp_path, monkeypatch):
        """The incident regression: a denied directory must raise within
        ``_BOUNDED_MKSTEMP_ATTEMPTS`` open attempts — not retry TMP_MAX
        (2**31-1 on Windows) candidate names at 100% CPU.
        """
        monkeypatch.setattr(utils, "_MKSTEMP_RETRY_PERMISSION_ERROR", True)
        calls = {"n": 0}

        def denying_open(path, flags, mode=0o777):
            calls["n"] += 1
            raise PermissionError(13, "Access is denied", path)

        monkeypatch.setattr(os, "open", denying_open)
        with pytest.raises(PermissionError):
            bounded_mkstemp(dir=str(tmp_path))
        assert calls["n"] == _BOUNDED_MKSTEMP_ATTEMPTS

    def test_posix_semantics_raise_on_first_denial(self, tmp_path, monkeypatch):
        """On POSIX PermissionError is unambiguous — no retry can help."""
        monkeypatch.setattr(utils, "_MKSTEMP_RETRY_PERMISSION_ERROR", False)
        calls = {"n": 0}

        def denying_open(path, flags, mode=0o777):
            calls["n"] += 1
            raise PermissionError(13, "Permission denied", path)

        monkeypatch.setattr(os, "open", denying_open)
        with pytest.raises(PermissionError):
            bounded_mkstemp(dir=str(tmp_path))
        assert calls["n"] == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows ACL deny semantics")
class TestWindowsAclDeny:
    def test_acl_denied_dir_raises_promptly(self, tmp_path):
        """End-to-end on a real ACL-denied directory: PermissionError in
        milliseconds, where ``tempfile.mkstemp`` spins for hours.

        The deny ACE is placed by SID, not account name: on machines with
        several similar local principals (sandbox pseudo-users, a machine
        name equal to the user name) icacls can resolve a bare name to the
        wrong principal and silently produce an ineffective deny.  A probe
        write then confirms the deny actually bites before asserting.
        """
        import subprocess

        denied = tmp_path / "denied"
        denied.mkdir()
        whoami = os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "whoami.exe"
        )
        got_sid = subprocess.run(
            [whoami, "/user", "/fo", "csv", "/nh"], capture_output=True, text=True
        )
        if got_sid.returncode != 0:
            pytest.skip(f"cannot resolve own SID: {got_sid.stderr.strip()}")
        sid = got_sid.stdout.strip().rsplit(",", 1)[-1].strip().strip('"')
        set_deny = subprocess.run(
            ["icacls", str(denied), "/deny", f"*{sid}:(OI)(CI)(W,AD,WD,DC)"],
            capture_output=True,
            text=True,
        )
        if set_deny.returncode != 0:
            pytest.skip(f"cannot set a deny ACL here: {set_deny.stderr.strip()}")
        try:
            try:
                probe_fd = os.open(
                    str(denied / "probe.bin"), os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
                )
            except PermissionError:
                pass  # the deny is effective — proceed to the real assertion
            else:
                os.close(probe_fd)
                pytest.skip("deny ACL not effective in this environment")

            start = time.monotonic()
            with pytest.raises(PermissionError):
                bounded_mkstemp(dir=str(denied))
            assert time.monotonic() - start < 5.0
        finally:
            subprocess.run(
                ["icacls", str(denied), "/remove:d", f"*{sid}"],
                capture_output=True,
                text=True,
            )
