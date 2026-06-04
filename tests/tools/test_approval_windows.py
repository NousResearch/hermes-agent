"""
Regression tests for Windows drive-letter path detection in dangerous commands.
Issue #38964: paths like X:/foo or X:\\foo bypassed approval on Windows/git-bash.
"""
import pytest
from tools.approval import detect_dangerous_command, detect_hardline_command


class TestWindowsDriveLetterApproval:
    """Windows drive-letter paths must trigger the same checks as Unix '/' paths."""

    # --- rm dangerous patterns ---
    def test_rm_windows_forward_slash(self):
        dangerous, key, desc = detect_dangerous_command("rm X:/test.txt")
        assert dangerous is True
        assert key is not None
        assert "delete" in desc.lower()

    def test_rm_windows_backslash(self):
        dangerous, key, desc = detect_dangerous_command(r"rm X:\test.txt")
        assert dangerous is True
        assert key is not None

    def test_rm_windows_lowercase_drive(self):
        dangerous, key, desc = detect_dangerous_command("rm c:/foo/bar")
        assert dangerous is True

    def test_rm_windows_uppercase_drive(self):
        dangerous, key, desc = detect_dangerous_command("rm C:/foo/bar")
        assert dangerous is True

    def test_rm_with_flags_windows_path(self):
        dangerous, key, desc = detect_dangerous_command("rm -rf X:/PROJECTs/test")
        assert dangerous is True

    def test_rm_unix_root_still_triggers(self):
        dangerous, key, desc = detect_dangerous_command("rm /x/test.txt")
        assert dangerous is True
        assert "delete" in desc.lower()

    def test_rm_relative_path_safe(self):
        dangerous, key, desc = detect_dangerous_command("rm ./foo.txt")
        assert dangerous is False

    def test_rm_plain_filename_safe(self):
        dangerous, key, desc = detect_dangerous_command("rm readme.txt")
        assert dangerous is False

    # --- redirection dangerous patterns ---
    def test_redirect_to_windows_path(self):
        dangerous, key, desc = detect_dangerous_command("> C:/config.txt")
        assert dangerous is True

    def test_append_to_windows_path(self):
        dangerous, key, desc = detect_dangerous_command("echo foo >> C:/config.txt")
        assert dangerous is True

    def test_redirect_to_windows_backslash(self):
        dangerous, key, desc = detect_dangerous_command(r"> C:\config.txt")
        assert dangerous is True

    # --- tee dangerous patterns ---
    def test_tee_to_windows_path(self):
        dangerous, key, desc = detect_dangerous_command(r"tee C:\windows\system32\file")
        assert dangerous is True

    def test_tee_to_windows_forward_slash(self):
        dangerous, key, desc = detect_dangerous_command("echo foo | tee C:/important.txt")
        assert dangerous is True

    # --- hardline patterns ---
    def test_hardline_wipe_windows_drive_forward_slash(self):
        is_hardline, desc = detect_hardline_command("rm -rf X:/")
        assert is_hardline is True
        assert desc is not None

    def test_hardline_wipe_windows_drive_backslash(self):
        is_hardline, desc = detect_hardline_command(r"rm -rf X:\\")
        assert is_hardline is True

    def test_hardline_unix_root_still_blocks(self):
        is_hardline, desc = detect_hardline_command("rm -rf /")
        assert is_hardline is True
