"""Tests for write_file post-write content verification (verified flag)."""

import json
from unittest.mock import patch as mock_patch

import pytest

from tools.file_tools import write_file_tool

@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return tmp_path

class TestWriteVerification:
    def test_successful_write_reports_verified(self, workdir):
        f = workdir / "out.txt"
        r = json.loads(write_file_tool(str(f), "hello verified world\n", task_id="t-wv"))
        assert r.get("bytes_written") == len("hello verified world\n")
        assert r.get("verified") is True

    def test_unicode_content_verified(self, workdir):
        f = workdir / "uni.txt"
        content = "línea → uno · ✓\n"
        r = json.loads(write_file_tool(str(f), content, task_id="t-wv"))
        assert r.get("verified") is True

    def test_crlf_preservation_still_verifies(self, workdir):
        # Existing CRLF file: write_file converts LF content to CRLF before
        # writing; verification hashes the shim-adjusted content, so it must
        # still report verified.
        f = workdir / "win.txt"
        f.write_bytes(b"old line\r\n")
        # Establish a full-read baseline: write_file refuses to overwrite
        # existing files the task has never read (stale-write guard).
        from tools.file_tools import read_file_tool
        json.loads(read_file_tool(str(f), task_id="t-wv"))
        r = json.loads(write_file_tool(str(f), "new line\nsecond\n", task_id="t-wv"))
        assert "error" not in r
        assert r.get("verified") is True
        assert b"\r\n" in f.read_bytes()

    def test_hash_mismatch_is_hard_error(self, workdir, monkeypatch):
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")  # the shell sha256sum transport
        f = workdir / "bad.txt"
        import tools.file_operations as fo
        real_sha = fo.hashlib.sha256

        class _WrongHash:
            def __init__(self, *a, **k):
                self._h = real_sha(b"different content entirely")
            def hexdigest(self):
                return self._h.hexdigest()

        with mock_patch.object(fo.hashlib, "sha256", _WrongHash):
            r = json.loads(write_file_tool(str(f), "actual content\n", task_id="t-wv"))
        assert "error" in r

    @pytest.mark.platforms("posix")  # the native write path is POSIX-only (_native_read_enabled)
    def test_native_hash_mismatch_is_hard_error(self, workdir):
        f = workdir / "bad_native.txt"
        with mock_patch("tools.file_operations_native.sha256_of", return_value="0" * 64):
            r = json.loads(write_file_tool(str(f), "actual content\n", task_id="t-wv3"))
        assert "error" in r and "verification failed" in r["error"]

    def test_hash_exception_fails_closed(self, workdir, monkeypatch):
        # sha256sum raising must not report a successful unverified write.
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")  # the shell sha256sum transport
        f = workdir / "ok.txt"
        import tools.file_operations as fo

        real_exec = fo.ShellFileOperations._exec

        def flaky_exec(self, cmd, **kw):
            if "sha256sum" in cmd:
                raise RuntimeError("no hash binary")
            return real_exec(self, cmd, **kw)

        with mock_patch.object(fo.ShellFileOperations, "_exec", flaky_exec):
            r = json.loads(write_file_tool(str(f), "content lands anyway\n", task_id="t-wv2"))
        assert "error" in r
        assert "verification failed" in r["error"].lower()
        assert "could not be hashed" in r["error"].lower()
        assert "verified" not in r

    def test_unreadable_hash_fails_closed(self, workdir, monkeypatch):
        # exit 0 write + sha256sum that cannot read the path is not success.
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        f = workdir / "missing-confirm.txt"
        import tools.file_operations as fo

        real_exec = fo.ShellFileOperations._exec

        def blind_exec(self, cmd, **kw):
            if "sha256sum" in cmd:
                return fo.ExecuteResult(stdout="", exit_code=1)
            return real_exec(self, cmd, **kw)

        with mock_patch.object(fo.ShellFileOperations, "_exec", blind_exec):
            r = json.loads(write_file_tool(str(f), "hello\n", task_id="t-wv4"))
        assert "error" in r
        assert "could not be read" in r["error"].lower()
        assert "no file was confirmed" in r["error"].lower()
