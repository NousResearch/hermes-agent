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
        r = json.loads(write_file_tool(str(f), "new line\nsecond\n", task_id="t-wv"))
        assert "error" not in r
        assert r.get("verified") is True
        assert b"\r\n" in f.read_bytes()

    def test_hash_mismatch_is_hard_error(self, workdir):
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
        assert "did not persist" in r["error"]

    def test_verification_failure_never_breaks_write(self, workdir):
        # sha256sum unavailable/failing -> verified omitted, write still ok.
        f = workdir / "ok.txt"
        import tools.file_operations as fo

        real_exec = fo.ShellFileOperations._exec

        def flaky_exec(self, cmd, **kw):
            if "sha256sum" in cmd:
                raise RuntimeError("no hash binary")
            return real_exec(self, cmd, **kw)

        with mock_patch.object(fo.ShellFileOperations, "_exec", flaky_exec):
            r = json.loads(write_file_tool(str(f), "content lands anyway\n", task_id="t-wv2"))
        assert "error" not in r
        assert f.read_text() == "content lands anyway\n"
        assert "verified" not in r or r.get("verified") is None


# ── Local patch (ebizmarts): digest parse tolerates a noisy prefix line ──


def test_hash_verification_ignores_non_hash_output_prefix(workdir):
    # Some native libraries log after fork and their diagnostic is merged
    # into command stdout before sha256sum output. Verification must find
    # the digest line rather than assuming the first token is the digest.
    f = workdir / "noisy-hash.txt"
    import tools.file_operations as fo

    real_exec = fo.ShellFileOperations._exec

    def noisy_exec(self, cmd, **kw):
        result = real_exec(self, cmd, **kw)
        if "sha256sum" in cmd and result.exit_code == 0:
            return fo.ExecuteResult(
                stdout=(
                    "W0826 11:08:16.577183 ev_poll_posix.cc:593] "
                    "some other diagnostic that is not a digest\n"
                    + result.stdout
                ),
                exit_code=result.exit_code,
            )
        return result

    with mock_patch.object(fo.ShellFileOperations, "_exec", noisy_exec):
        r = json.loads(write_file_tool(str(f), "content survives noisy output\n", task_id="t-wv-noisy"))
    assert "error" not in r
    assert r.get("verified") is True
    assert f.read_text() == "content survives noisy output\n"
