"""Regression tests for Windows drive-letter path detection in approval.py.

Teknium1 review (#38979): drive-letter rm detection must live in the
_RM_FLAG_PREFIX / _hardline_rm_path() mechanism, use real single-backslash
separators (matching what the shell delivers after normalization), and cover
forward-slash form without gating safe temp writes.
"""
import pytest


def _detect(cmd):
    from tools.approval import detect_dangerous_command
    return detect_dangerous_command(cmd)


class TestWindowsDrivePaths:
    """rm targeting Windows drive roots must be hardline-blocked."""

    def test_rm_rf_windows_forward_slash(self):
        """rm -rf C:/ must be hardline-blocked (forward-slash form)."""
        dangerous, key, desc = _detect("rm -rf C:/")
        assert dangerous is True
        assert key is not None
        assert "delete" in (desc or "").lower() or "drive" in (desc or "").lower()

    def test_rm_rf_windows_forward_slash_with_path(self):
        """rm -rf X:/important must be hardline-blocked."""
        dangerous, key, desc = _detect("rm -rf X:/important/data")
        assert dangerous is True

    def test_rm_rf_windows_lowercase_drive(self):
        """rm -rf c:/ (lowercase) must be hardline-blocked."""
        dangerous, key, desc = _detect("rm -rf c:/")
        assert dangerous is True

    def test_rm_rf_windows_single_backslash(self):
        """rm -rf C:\\ — single backslash as delivered by normalization."""
        # The shell delivers a single backslash; normalization in approval.py
        # strips one level of escaping, so we pass the normalized form here.
        dangerous, key, desc = _detect("rm -rf C:\\")
        assert dangerous is True

    def test_rm_no_flag_windows_drive_root(self):
        """rm C:/ (drive root, no flag) must be caught."""
        dangerous, key, desc = _detect("rm C:/")
        assert dangerous is True

    def test_safe_write_to_temp_not_blocked(self):
        """echo data > C:/Temp/scratch.txt must NOT be hardline-blocked.
        Windows analogue of the safe POSIX /tmp write."""
        dangerous, key, desc = _detect("echo data > C:/Temp/scratch.txt")
        # Must not be in HARDLINE_PATTERNS (could be in DANGEROUS_PATTERNS
        # with a warning, but must not be an unconditional block for temp files)
        from tools.approval import HARDLINE_PATTERNS
        import re
        cmd = "echo data > C:/Temp/scratch.txt"
        for pattern, _ in HARDLINE_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                pytest.fail(
                    f"Safe temp write '{cmd}' matched hardline pattern {pattern!r}. "
                    "Windows drive-root deletion rule must not catch all drive-path writes."
                )
