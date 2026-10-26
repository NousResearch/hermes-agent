"""Surrogate-safe stdin piping for the local execution environment (#79178).

These tests exercise the REAL `_pipe_stdin` writer thread against a real
subprocess — no mocks. They pin the round-trip byte contract (utf-8 +
surrogateescape is the inverse of the decode that produced the content) and
the always-close / error-capture guarantees of the writer thread.
"""
import shlex
import subprocess
import time
from unittest.mock import MagicMock

import pytest

from tools.environments.base_output import _pipe_stdin
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


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


@pytest.fixture
def env(tmp_path):
    """A real LocalEnvironment rooted in a temp directory."""
    return LocalEnvironment(cwd=str(tmp_path), timeout=15)


@pytest.fixture
def ops(env, tmp_path):
    """ShellFileOperations wired to the real local environment."""
    return ShellFileOperations(env, cwd=str(tmp_path))


class TestWriteFileSurrogates:
    def test_roundtrip_preserves_bytes_count_and_hash(self, ops, tmp_path):
        p = tmp_path / "surrogate.bin"
        res = ops.write_file(str(p), b"\xff\x00\xfe".decode("utf-8", "surrogateescape"))
        assert res.error is None
        assert res.bytes_written == 3
        assert res.verified is True
        assert p.read_bytes() == b"\xff\x00\xfe"
        assert not list(tmp_path.glob(".hermes-tmp*"))

    def test_roundtrip_mixed_normal_and_surrogate(self, ops, tmp_path):
        content = "head\n" + b"\xff".decode("utf-8", "surrogateescape") + "\ntail\n"
        p = tmp_path / "mixed.bin"
        res = ops.write_file(str(p), content)
        assert res.error is None
        assert res.verified is True
        assert p.read_bytes() == b"head\n\xff\ntail\n"

    @pytest.mark.parametrize("bad", ["\ud800", "\udc7f", "\udd00"])
    def test_unencodable_surrogate_rejected_before_write(self, ops, tmp_path, bad):
        p = tmp_path / "reject.bin"
        res = ops.write_file(str(p), bad)
        assert res.error and "surrogate" in res.error
        assert "NOT created or modified" in res.error
        assert "timed out" not in res.error
        # Pins the EARLY rejection (char repr in the message) rather than the
        # post-BOM backstop (whose message contains the codec traceback) —
        # the early rejection is what guarantees no child ever spawns.
        assert "codec can't encode" not in res.error
        assert not p.exists()

    def test_rejected_write_leaves_existing_target_unchanged(self, ops, tmp_path):
        p = tmp_path / "keep.bin"
        p.write_bytes(b"precious original bytes")
        res = ops.write_file(str(p), "\ud800")
        assert res.error and "NOT created or modified" in res.error
        assert p.read_bytes() == b"precious original bytes"

    def test_patch_replace_funnel_rejects_surrogate_new_string(self, ops, tmp_path):
        p = tmp_path / "patchme.txt"
        p.write_text("old\n")
        # \udc7f is OUTSIDE the surrogateescape round-trip range (U+DC80–U+DCFF)
        # — unencodable even with surrogateescape — so write_file's early
        # rejection must catch it through the patch funnel. (An in-range
        # surrogate like \udcff legitimately round-trips, per the spec.)
        res = ops.patch_replace(str(p), "old", "new" + "\udc7f")
        assert res.error and "surrogate" in res.error
        assert p.read_text() == "old\n"

    def test_normal_content_verified(self, ops, tmp_path):
        p = tmp_path / "normal.txt"
        res = ops.write_file(str(p), "hello\nworld\n")
        assert res.error is None
        assert res.verified is True
        assert p.read_bytes() == b"hello\nworld\n"


class TestStdinErrorPropagation:
    def test_execute_surfaces_stdin_error_without_hanging(self, env):
        t0 = time.monotonic()
        result = env.execute("cat > /dev/null", stdin_data="\ud800")
        elapsed = time.monotonic() - t0

        assert result["returncode"] == 0  # child saw EOF, exited cleanly
        assert result.get("stdin_error")  # the write failure was surfaced
        assert "stdin write failed" in result["output"]
        assert elapsed < 5.0, f"stdin failure path hung for {elapsed:.1f}s"

    def test_normal_path_result_has_no_stdin_error_key(self, env):
        result = env.execute("echo hi")
        assert "stdin_error" not in result
        assert result["returncode"] == 0
        assert "hi" in result["output"]


class TestExecStdinErrorMapping:
    def test_exec_maps_stdin_error_to_failure(self):
        """Defense-in-depth path (unreachable from write_file after Task 3 —
        tested with a mock env for exactly that reason)."""
        env = MagicMock()
        env.execute.return_value = {
            "output": "child output\n[stdin write failed: boom]",
            "returncode": 0,
            "stdin_error": "boom",
        }
        ops = ShellFileOperations(env, cwd="/tmp")
        result = ops._exec("echo hi", cwd="/tmp", stdin_data="\ud800")
        assert result.exit_code == 1
        assert "boom" in result.stdout


class TestPipeStdinRemainingBranches:
    """Review-requested coverage: bytes passthrough + proc.stdin None."""

    def test_bytes_input_passes_through_untouched(self, tmp_path):
        out = tmp_path / "out.bin"
        proc = subprocess.Popen(
            ["bash", "-c", f"cat > {shlex.quote(str(out))}"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
        )
        try:
            _pipe_stdin(proc, b"\x00\x01\xfe")
            _wait_or_kill(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode == 0
        assert out.read_bytes() == b"\x00\x01\xfe"
        assert proc._hermes_stdin_errors == []

    def test_stdin_none_records_runtime_error(self, tmp_path):
        proc = subprocess.Popen(
            ["bash", "-c", "exit 0"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
        )
        _pipe_stdin(proc, "data")
        _wait_or_kill(proc)
        assert proc.returncode == 0
        assert proc._hermes_stdin_errors
        assert isinstance(proc._hermes_stdin_errors[0], RuntimeError)
