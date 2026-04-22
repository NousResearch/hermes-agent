"""Regression tests for byte-exact stdin piping (no LF -> CRLF translation).

On Windows, ``subprocess.Popen(text=True)`` enables universal-newlines on the
stdin pipe, which silently rewrites every ``"\n"`` to ``"\r\n"`` before the
child reads it. For ``cat > file`` writers this corrupts on-disk content (LF
input lands as CRLF; downstream pipes can layer a second ``\r`` to produce
CRCRLF).

These tests pin the contract: data written via ``_pipe_stdin`` must reach the
child's stdin byte-for-byte identical to the input string (after UTF-8
encoding), regardless of host OS.
"""

import subprocess
import sys

import pytest

from tools.environments.base import _pipe_stdin, _popen_bash


def _read_child_stdin(data: str) -> bytes:
    """Pipe *data* into a passthrough child and capture the raw bytes it saw on stdin."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # Binary stdout so we don't lose bytes to universal-newlines on the
        # capture side. _pipe_stdin must still cope with text-mode stdin
        # (which is what the production code path uses).
        text=True,
    )
    _pipe_stdin(proc, data)
    proc.wait(timeout=10)
    # Read stdout as raw bytes via the underlying buffer to preserve the
    # exact byte sequence the child wrote.
    return proc.stdout.buffer.read() if hasattr(proc.stdout, "buffer") else proc.stdout.read().encode("utf-8")


class TestPipeStdinByteExact:
    """``_pipe_stdin`` must deliver UTF-8 bytes without newline translation."""

    def test_lf_only_preserved(self):
        # Three LFs in must arrive as three LFs -- never CRLF, never CRCRLF.
        captured = _read_child_stdin("a\nb\nc\n")
        # The Python helper reads via ``sys.stdin.buffer`` (no decoding) and
        # writes the raw bytes back to stdout.buffer. We then read that back
        # in text mode (text=True), which on Windows applies universal-
        # newlines to *stdout*, but not stdin -- so any CRLF that arrived
        # would be visible as a stripped \r in stdout. Use binary length
        # checks instead of equality to be robust to that wrinkle.
        assert b"\r\n" not in captured
        assert b"\r\r\n" not in captured
        # Three logical lines plus terminator
        assert captured.count(b"\n") == 3

    def test_existing_crlf_not_doubled(self):
        # If user writes Windows-style ``\r\n``, it must not become ``\r\r\n``.
        captured = _read_child_stdin("a\r\nb\r\n")
        assert b"\r\r\n" not in captured

    def test_unicode_payload_utf8_encoded(self):
        captured = _read_child_stdin("日本語\n")
        assert captured == "日本語\n".encode("utf-8")

    def test_empty_payload(self):
        captured = _read_child_stdin("")
        assert captured == b""

    def test_no_trailing_newline_added(self):
        captured = _read_child_stdin("nofinal")
        assert captured == b"nofinal"


class TestPopenBashStdinByteExact:
    """End-to-end: full ``_popen_bash`` + ``cat > file`` pipeline."""

    def test_cat_to_file_preserves_lf(self, tmp_path):
        target = tmp_path / "out.txt"
        # Use POSIX-style path that works under both Linux bash and git-bash.
        # tmp_path on Windows looks like C:\Users\... -- bash will accept the
        # forward-slashed form via /c/Users/...
        target_str = str(target).replace("\\", "/")
        # On Windows git-bash, drive letter prefix needs translation.
        if sys.platform == "win32" and len(target_str) >= 2 and target_str[1] == ":":
            target_str = "/" + target_str[0].lower() + target_str[2:]
        proc = _popen_bash(
            ["bash", "-lc", f"cat > {target_str}"],
            stdin_data="alpha\nbeta\ngamma\n",
        )
        proc.wait(timeout=10)
        assert proc.returncode == 0
        raw = target.read_bytes()
        assert raw == b"alpha\nbeta\ngamma\n", (
            f"Expected pure LF, got {raw!r}. This indicates the stdin pipe "
            f"applied newline translation -- the very bug this test guards."
        )

    def test_cat_to_file_no_crcrlf(self, tmp_path):
        """Specifically guard the CRCRLF (``\\r\\r\\n``) corruption pattern."""
        target = tmp_path / "out.txt"
        target_str = str(target).replace("\\", "/")
        if sys.platform == "win32" and len(target_str) >= 2 and target_str[1] == ":":
            target_str = "/" + target_str[0].lower() + target_str[2:]
        proc = _popen_bash(
            ["bash", "-lc", f"cat > {target_str}"],
            stdin_data="x\ny\nz\n",
        )
        proc.wait(timeout=10)
        raw = target.read_bytes()
        assert b"\r\r\n" not in raw, (
            f"CRCRLF detected on disk: {raw!r}. The Windows pipe layer "
            f"double-translated newlines."
        )
