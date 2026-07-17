"""Tests for the Windows Git Bash unquoted-backslash-path lint.

On Windows the local terminal backend runs commands through Git Bash,
which strips single backslashes from unquoted words — ``C:\\Users\\x``
reaches the command as ``C:Usersx``. The lint rejects such commands
loudly before execution instead of letting the path mangle silently
(this broke stored cron-job prompts for days before being noticed).
"""

import json
import platform

import pytest

from tools.terminal_tool import (
    _find_unquoted_backslash_win_path,
    cleanup_all_environments,
    terminal_tool,
)


# ---------------------------------------------------------------------------
# Unit tests for the detector (platform-independent)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command,expected", [
    # The original bug: unquoted backslash path in a python invocation.
    (
        r"python C:\Users\Paul\.hermes\scripts\x.py",
        r"C:\Users\Paul\.hermes\scripts\x.py",
    ),
    # Bare drive-letter path as the command itself.
    (r"C:\tools\run.exe --flag", r"C:\tools\run.exe"),
    # After a shell operator.
    (r"cd /tmp && cat C:\logs\out.txt", r"C:\logs\out.txt"),
    # First path quoted, second unquoted — the unquoted one is flagged.
    (r"cp 'C:\a\b' C:\c\d", r"C:\c\d"),
])
def test_flags_unquoted_backslash_paths(command, expected):
    assert _find_unquoted_backslash_win_path(command) == expected


@pytest.mark.parametrize("command", [
    # Forward slashes — the recommended form.
    "python C:/Users/Paul/.hermes/scripts/x.py",
    # Single-quoted — bash preserves the backslashes literally.
    r"python 'C:\Users\Paul\x.py'",
    # Double-quoted — bash keeps backslashes before ordinary characters.
    r'python "C:\Users\Paul\x.py"',
    # Doubled backslashes — bash collapses them back to single ones.
    r"python C:\\Users\\Paul\\x.py",
    # Inside a here-doc body backslashes before ordinary chars survive.
    "cat <<EOF\n" + r"path is C:\Users\Paul" + "\nEOF",
    # No Windows path at all.
    "ls -la /tmp && echo done",
    # Drive-letter-like text without a backslash (URL-ish, scp-ish).
    "scp host:/tmp/f . && echo C:/ok",
])
def test_passes_safe_commands(command):
    assert _find_unquoted_backslash_win_path(command) is None


def test_unterminated_single_quote_does_not_crash():
    assert _find_unquoted_backslash_win_path(r"echo 'C:\Users\Paul") is None


# ---------------------------------------------------------------------------
# terminal_tool gate (platform monkeypatched so it runs everywhere)
# ---------------------------------------------------------------------------

def test_terminal_tool_rejects_backslash_path_loudly(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("HERMES_TERMINAL_SKIP_WIN_PATH_LINT", raising=False)
    monkeypatch.setattr(
        "tools.terminal_tool.platform.system", lambda: "Windows"
    )

    result = json.loads(
        terminal_tool(r"python C:\Users\Paul\.hermes\scripts\x.py")
    )

    assert result["exit_code"] == -1
    assert result["status"] == "error"
    assert "Git Bash" in result["error"]
    assert r"C:\Users\Paul\.hermes\scripts\x.py" in result["error"]
    # The error must teach the fix, not just reject.
    assert "forward slashes" in result["error"]


def test_terminal_tool_lint_skipped_on_non_windows(monkeypatch):
    """The lint is Windows-only: on POSIX hosts backslash commands are
    legitimate shell escapes and must not be intercepted."""
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(
        "tools.terminal_tool.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "tools.terminal_tool._find_unquoted_backslash_win_path",
        lambda _c: pytest.fail("lint must not run on non-Windows hosts"),
    )
    # Force the command down an early error path *after* the lint gate so
    # no real environment is created: foreground timeout above the cap.
    result = json.loads(terminal_tool(r"echo C:\x", timeout=100000))
    assert "exceeds the maximum" in result["error"]


# ---------------------------------------------------------------------------
# Real execution on a Windows host (integration; skipped elsewhere)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only")
def test_windows_real_run_backslash_errors_and_forward_slash_works(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("HERMES_TERMINAL_SKIP_WIN_PATH_LINT", raising=False)

    # Backslash path: rejected loudly before any bash process runs.
    rejected = json.loads(terminal_tool(r"echo C:\Users\nobody\x.txt"))
    assert rejected["status"] == "error"
    assert "Git Bash" in rejected["error"]

    # Forward-slash form of the same command: executes and survives intact.
    try:
        ran = json.loads(terminal_tool("echo C:/Users/nobody/x.txt"))
        assert ran.get("exit_code") == 0
        assert "C:/Users/nobody/x.txt" in ran.get("output", "")
    finally:
        cleanup_all_environments()
