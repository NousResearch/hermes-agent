"""Regression tests for issue #8340.

When a user command backgrounds a child process (``cmd &``, ``setsid cmd &
disown``, etc.), the backgrounded grandchild inherits the write-end of our
stdout pipe via fork().  Before the fix, the drain thread's blocking
``for line in proc.stdout`` would never see EOF until that grandchild
closed the pipe — causing the terminal tool to hang for the full lifetime
of the backgrounded service (indefinitely for a uvicorn server).

The fix switches ``_drain()`` to select()-based non-blocking reads and
stops draining shortly after bash exits even if the pipe hasn't EOF'd.
"""
import shlex
import subprocess
import sys
import time

import pytest

from tools.environments.local import LocalEnvironment

# The interpreter that is actually running these tests. Hardcoding ``python3``
# names a binary that does not exist on Windows (it is ``python`` there), so
# the shell child exited 127 and the drain behaviour under test never ran.
PY = shlex.quote(sys.executable)


def _pkill(pattern: str) -> None:
    subprocess.run(f"pkill -9 -f {pattern!r} 2>/dev/null", shell=True)


@pytest.fixture
def local_env():
    env = LocalEnvironment(cwd="/tmp")
    try:
        yield env
    finally:
        env.cleanup()


class TestBackgroundChildDoesNotHang:
    """Regression guard for issue #8340."""

    def test_plain_background_returns_promptly(self, local_env):
        """``cmd &`` with no output redirection must not hang on pipe inherit."""
        marker = "hermes_8340_plain_bg"
        cmd = f'python3 -c "import time; time.sleep(60)" & echo {marker}'
        try:
            t0 = time.monotonic()
            result = local_env.execute(cmd, timeout=15)
            elapsed = time.monotonic() - t0

            assert elapsed < 10.0, (  # hang under guard is 15s+; loose bound rides out runner stalls
                f"terminal_tool hung for {elapsed:.1f}s — drain thread "
                f"is still blocking on backgrounded child's inherited pipe fd"
            )
            assert result["returncode"] == 0
            assert marker in result["output"]
        finally:
            _pkill("time.sleep(60)")

    def test_setsid_disown_pattern_returns_promptly(self, local_env):
        """The exact pattern from the issue: setsid ... & disown."""
        cmd = (
            'setsid python3 -c "import time; time.sleep(60)" '
            '> /dev/null 2>&1 < /dev/null & disown; echo started'
        )
        try:
            t0 = time.monotonic()
            result = local_env.execute(cmd, timeout=15)
            elapsed = time.monotonic() - t0

            assert elapsed < 10.0, f"setsid+disown path hung for {elapsed:.1f}s"
            assert result["returncode"] == 0
            assert "started" in result["output"]
        finally:
            _pkill("time.sleep(60)")


    def test_default_capture_is_full_fidelity_for_internal_consumers(
        self, local_env
    ):
        """Default execute() (no bounded_capture) must return complete output.

        Internal consumers — file-operation ``cat`` reads that feed the patch
        engine, code-execution RPC reads, log reads — rely on full-fidelity
        capture. Bounding them at tool_output.max_bytes would CORRUPT files
        on read-modify-write (#64435 review finding), so only the foreground
        terminal tool opts in via bounded_capture=True.
        """
        # ~200 KB — four times the default 50 KB cap.
        command = (
            f"{PY} -c \"import sys; "
            "sys.stdout.write('START-MARK\\n' + ('y' * 200000) + '\\nEND-MARK')\""
        )

        result = local_env.execute(command, timeout=10)

        assert result["returncode"] == 0
        assert "[OUTPUT TRUNCATED" not in result["output"]
        assert result["output"].startswith("START-MARK")
        assert result["output"].endswith("END-MARK")
        assert len(result["output"]) > 200000


    def test_utf8_multibyte_across_read_boundary(self, local_env):
        """Multibyte UTF-8 characters straddling a 4096-byte ``os.read()`` boundary
        must be decoded correctly via the incremental decoder — not lost to a
        ``UnicodeDecodeError`` fallback.  Regression for a bug in the first draft
        of the fix where a strict ``bytes.decode('utf-8')`` on each raw chunk
        wiped the entire buffer as soon as any chunk split a multi-byte char.
        """
        # 10000 "日" chars = 30000 bytes — guaranteed to cross multiple 4096
        # read boundaries, and most boundaries will land in the middle of the
        # 3-byte UTF-8 encoding of U+65E5.
        cmd = (
            f'{PY} -c \'import sys; '
            'sys.stdout.buffer.write(chr(0x65e5).encode("utf-8") * 10000); '
            'sys.stdout.buffer.write(b"\\n")\''
        )
        result = local_env.execute(cmd, timeout=10)
        assert result["returncode"] == 0
        # All 10000 characters must survive the round-trip
        assert result["output"].count("\u65e5") == 10000, (
            f"lost multibyte chars across read boundaries: got "
            f"{result['output'].count(chr(0x65e5))} / 10000"
        )
        # And the "[binary output detected ...]" fallback must NOT fire
        assert "binary output detected" not in result["output"]

    def test_invalid_utf8_uses_replacement_not_fallback(self, local_env):
        """Truly invalid byte sequences must be substituted with U+FFFD (matching
        the pre-fix ``errors='replace'`` behaviour of the old ``TextIOWrapper``
        drain), not clobber the entire buffer with a fallback placeholder.
        """
        # Write a deliberate invalid UTF-8 lead byte sandwiched between valid ASCII
        cmd = (
            f'{PY} -c \'import sys; '
            'sys.stdout.buffer.write(b"before "); '
            'sys.stdout.buffer.write(b"\\xff\\xfe"); '
            'sys.stdout.buffer.write(b" after\\n")\''
        )
        result = local_env.execute(cmd, timeout=15)
        assert result["returncode"] == 0
        assert "before" in result["output"]
        assert "after" in result["output"]
        assert "binary output detected" not in result["output"]
