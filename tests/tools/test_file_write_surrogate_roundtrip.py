"""Surrogate-safe stdin piping for the local execution environment (#79178).

These tests exercise the REAL `_pipe_stdin` writer thread against a real
subprocess — no mocks. They pin the round-trip byte contract (utf-8 +
surrogateescape is the inverse of the decode that produced the content) and
the always-close / error-capture guarantees of the writer thread. Later
tasks in this plan append propagation and write_file tests to this file.
"""
import shlex
import subprocess

import pytest

from tools.environments.base import _pipe_stdin


def _cat_to_file_proc(out_path):
    """A real child that copies its stdin to a file, byte for byte."""
    return subprocess.Popen(
        ["bash", "-c", f"cat > {shlex.quote(str(out_path))}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _wait_or_kill(proc, timeout=5):
    """wait() with a bounded timeout; kill on timeout so a hung child never
    leaks into the next test."""
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise


class TestPipeStdinSurrogates:
    def test_roundtrips_surrogateescape_bytes(self, tmp_path):
        out = tmp_path / "out.bin"
        proc = _cat_to_file_proc(out)
        content = b"\xff\x00\xfe".decode("utf-8", "surrogateescape")
        try:
            _pipe_stdin(proc, content)
            _wait_or_kill(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode == 0
        assert out.read_bytes() == b"\xff\x00\xfe"
        assert proc._hermes_stdin_errors == []

    def test_unencodable_surrogate_captures_error_and_closes_stdin(self, tmp_path):
        out = tmp_path / "out.bin"
        proc = _cat_to_file_proc(out)
        try:
            _pipe_stdin(proc, "\ud800")  # outside the surrogateescape round-trip range
            _wait_or_kill(proc)  # child MUST exit promptly — stdin closed in finally
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode == 0  # child saw EOF and exited cleanly
        assert proc._hermes_stdin_errors  # the encode failure was captured
        assert isinstance(proc._hermes_stdin_errors[0], UnicodeEncodeError)

    def test_normal_content_unchanged(self, tmp_path):
        out = tmp_path / "out.bin"
        proc = _cat_to_file_proc(out)
        try:
            _pipe_stdin(proc, "hello\nworld\n")
            _wait_or_kill(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode == 0
        assert out.read_bytes() == b"hello\nworld\n"
        assert proc._hermes_stdin_errors == []
