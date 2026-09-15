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


class TestDigestParsing:
    """The executor merges stderr into `output` (process_registry spawns with
    stderr=subprocess.STDOUT), so shell noise can precede sha256sum's line. The
    verifier must find the real digest and must never turn noise into a
    'content mismatch' on a write that persisted. Regression for the
    shell-init/getcwd false mismatch."""

    NOISE = (
        "shell-init: error retrieving current directory: getcwd: cannot access "
        "parent directories: Operation not permitted\n"
        "chdir: error retrieving current directory: getcwd: cannot access "
        "parent directories: Operation not permitted\n"
    )

    @staticmethod
    def _exec_with_noise(noise, digest_for):
        """Patch _exec so sha256sum returns `noise` followed by a digest line.

        `digest_for` maps the on-disk bytes to the digest to report, so a test
        can emit either the correct digest or a wrong-but-valid one.
        """
        import tools.file_operations as fo
        real_exec = fo.ShellFileOperations._exec

        def fake_exec(self, cmd, **kw):
            if "sha256sum" in cmd:
                real = real_exec(self, cmd, **kw)
                digest = digest_for(real.stdout)
                body = f"{digest}  /path\n" if digest else ""
                return fo.ExecuteResult(stdout=noise + body, exit_code=0)
            return real_exec(self, cmd, **kw)

        return mock_patch.object(fo.ShellFileOperations, "_exec", fake_exec)

    def test_warning_before_digest_still_verifies(self, workdir):
        # The actual bug: noise + correct digest was read as a mismatch.
        import tools.file_operations as fo
        f = workdir / "noisy.md"
        content = "# heading\n\nbody text\n"
        expected = fo.hashlib.sha256(content.encode()).hexdigest()

        with self._exec_with_noise(self.NOISE, lambda _out: expected):
            r = json.loads(write_file_tool(str(f), content, task_id="t-wv3"))

        assert "error" not in r, r.get("error")
        assert r.get("verified") is True
        assert r.get("bytes_written") == len(content.encode())
        assert f.read_text() == content

    def test_warning_plus_wrong_digest_is_still_a_hard_error(self, workdir):
        # Noise must not become a blanket excuse: a genuinely wrong digest is
        # still a mismatch.
        import tools.file_operations as fo
        f = workdir / "wrong.md"
        wrong = fo.hashlib.sha256(b"something else entirely").hexdigest()

        with self._exec_with_noise(self.NOISE, lambda _out: wrong):
            r = json.loads(write_file_tool(str(f), "real content\n", task_id="t-wv4"))

        assert "error" in r
        assert "did not persist" in r["error"]

    def test_noise_without_any_digest_is_unverified_not_mismatch(self, workdir):
        # No parseable digest -> verification unavailable. The write stands and
        # no mismatch is fabricated.
        f = workdir / "nodigest.md"
        content = "content lands\n"

        with self._exec_with_noise(self.NOISE, lambda _out: None):
            r = json.loads(write_file_tool(str(f), content, task_id="t-wv5"))

        assert "error" not in r, r.get("error")
        assert r.get("verified") is None
        assert f.read_text() == content


class TestParseSha256Digest:
    """Unit coverage for the parser itself."""

    def test_plain_output(self):
        from tools.file_operations import _parse_sha256_digest
        d = "a" * 64
        assert _parse_sha256_digest(f"{d}  /tmp/f") == d

    def test_prepended_noise(self):
        from tools.file_operations import _parse_sha256_digest
        d = "3a1363c0" + "b" * 56
        out = "shell-init: error retrieving current directory\n" + f"{d}  /tmp/f\n"
        assert _parse_sha256_digest(out) == d

    def test_uppercase_digest_normalized(self):
        from tools.file_operations import _parse_sha256_digest
        assert _parse_sha256_digest(("A" * 64) + "  /tmp/f") == "a" * 64

    def test_no_digest_returns_none(self):
        from tools.file_operations import _parse_sha256_digest
        assert _parse_sha256_digest("shell-init: error\nchdir: error\n") is None
        assert _parse_sha256_digest("") is None

    def test_too_short_or_too_long_hex_rejected(self):
        from tools.file_operations import _parse_sha256_digest
        assert _parse_sha256_digest(("a" * 63) + "  /tmp/f") is None
        assert _parse_sha256_digest(("a" * 65) + "  /tmp/f") is None

    def test_hex_not_at_line_start_ignored(self):
        from tools.file_operations import _parse_sha256_digest
        assert _parse_sha256_digest("warning: token " + ("a" * 64) + "\n") is None
