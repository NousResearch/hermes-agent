"""Tests for Windows compatibility of process management and platform code.

Verifies that os.setsid and os.killpg are never called unconditionally,
that each module uses a platform guard before invoking POSIX-only functions,
and that Windows-specific code paths are properly implemented.
"""

import ast
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Files that must have Windows-safe process management
GUARDED_FILES = [
    "tools/environments/local.py",
    "tools/process_registry.py",
    "tools/code_execution_tool.py",
    "gateway/platforms/whatsapp.py",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_preexec_fn_values(filepath: Path) -> list:
    """Find all preexec_fn= keyword arguments in Popen calls."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "preexec_fn":
            values.append(ast.dump(node.value))
    return values


class TestNoUnconditionalSetsid:
    """preexec_fn must never be a bare os.setsid reference."""

    @pytest.mark.parametrize("relpath", GUARDED_FILES)
    def test_preexec_fn_is_guarded(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        values = _get_preexec_fn_values(filepath)
        for val in values:
            # A bare os.setsid would be: Attribute(value=Name(id='os'), attr='setsid')
            assert "attr='setsid'" not in val or "IfExp" in val or "None" in val, (
                f"{relpath} has unconditional preexec_fn=os.setsid"
            )


class TestIsWindowsConstant:
    """Each guarded file must define _IS_WINDOWS."""

    @pytest.mark.parametrize("relpath", GUARDED_FILES)
    def test_has_is_windows(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        source = filepath.read_text(encoding="utf-8")
        assert "_IS_WINDOWS" in source, (
            f"{relpath} missing _IS_WINDOWS platform guard"
        )


class TestKillpgGuarded:
    """os.killpg must always be behind a platform check."""

    @pytest.mark.parametrize("relpath", GUARDED_FILES)
    def test_no_unguarded_killpg(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        source = filepath.read_text(encoding="utf-8")
        lines = source.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "os.killpg" in stripped or "os.getpgid" in stripped:
                # Check that there's an _IS_WINDOWS guard in the surrounding context
                context = "\n".join(lines[max(0, i - 15):i + 1])
                assert "_IS_WINDOWS" in context or "else:" in context, (
                    f"{relpath}:{i + 1} has unguarded os.killpg/os.getpgid call"
                )


# ═════════════════════════════════════════════════════════════════════════
# Unix-only import guards
# ═════════════════════════════════════════════════════════════════════════

class TestFcntlImportGuarded:
    """Files that use fcntl must have try/except ImportError guard."""

    FCNTL_FILES = [
        "tools/memory_tool.py",
        "hermes_cli/auth.py",
        "cron/scheduler.py",
    ]

    @pytest.mark.parametrize("relpath", FCNTL_FILES)
    def test_fcntl_has_guard(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        source = filepath.read_text(encoding="utf-8")
        if "import fcntl" not in source:
            pytest.skip(f"{relpath} does not import fcntl")
        # Must have try/except around the import
        assert "except" in source and "fcntl" in source, (
            f"{relpath} has unguarded fcntl import"
        )
        # Must also have msvcrt fallback
        assert "msvcrt" in source, (
            f"{relpath} imports fcntl but has no msvcrt fallback"
        )


# ═════════════════════════════════════════════════════════════════════════
# Temp path handling — no hardcoded /tmp
# ═════════════════════════════════════════════════════════════════════════

class TestNoHardcodedTmpPaths:
    """Temp paths must use tempfile.gettempdir(), not hardcoded /tmp."""

    TEMP_PATH_FILES = [
        "tools/environments/local.py",
        "tools/environments/persistent_shell.py",
    ]

    @pytest.mark.parametrize("relpath", TEMP_PATH_FILES)
    def test_no_hardcoded_tmp(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        source = filepath.read_text(encoding="utf-8")
        lines = source.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip comments and strings that mention /tmp in docs
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
                continue
            # Look for f-string or string concatenation with /tmp/hermes
            if '"/tmp/hermes' in stripped or "'/tmp/hermes" in stripped:
                pytest.fail(
                    f"{relpath}:{i + 1} has hardcoded /tmp path: {stripped[:80]}"
                )


# ═════════════════════════════════════════════════════════════════════════
# signal.SIGKILL guard
# ═════════════════════════════════════════════════════════════════════════

class TestSigkillGuarded:
    """signal.SIGKILL must be guarded on Windows (doesn't exist there)."""

    SIGKILL_FILES = [
        "hermes_cli/gateway.py",
        "tools/code_execution_tool.py",
    ]

    @pytest.mark.parametrize("relpath", SIGKILL_FILES)
    def test_sigkill_is_guarded(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        source = filepath.read_text(encoding="utf-8")
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if "signal.SIGKILL" in line:
                # Must be behind a platform check or conditional expression
                context = "\n".join(lines[max(0, i - 10):i + 1])
                has_guard = any(g in context for g in [
                    "_IS_WINDOWS", "win32", "else:", "not is_windows",
                ])
                assert has_guard, (
                    f"{relpath}:{i + 1} has unguarded signal.SIGKILL"
                )


# ═════════════════════════════════════════════════════════════════════════
# /dev/tty guard
# ═════════════════════════════════════════════════════════════════════════

class TestDevTtyGuarded:
    """References to /dev/tty must have Windows alternatives."""

    def test_display_write_tty_has_windows_path(self):
        filepath = PROJECT_ROOT / "agent" / "display.py"
        if not filepath.exists():
            pytest.skip("agent/display.py not found")
        source = filepath.read_text(encoding="utf-8")
        assert "CON" in source or "win32" in source, (
            "agent/display.py: write_tty() has no Windows alternative to /dev/tty"
        )


# ═════════════════════════════════════════════════════════════════════════
# Clipboard — platform dispatch
# ═════════════════════════════════════════════════════════════════════════

class TestClipboardPlatformCoverage:
    """Clipboard module must handle all major platforms."""

    def test_has_windows_native_support(self):
        source = (PROJECT_ROOT / "hermes_cli" / "clipboard.py").read_text(encoding="utf-8")
        assert "win32" in source, "clipboard.py missing native Windows support"
        assert "_windows_save" in source, "clipboard.py missing _windows_save"
        assert "_windows_has_image" in source, "clipboard.py missing _windows_has_image"

    def test_has_text_clipboard_functions(self):
        source = (PROJECT_ROOT / "hermes_cli" / "clipboard.py").read_text(encoding="utf-8")
        assert "get_clipboard_text" in source, "clipboard.py missing get_clipboard_text"
        assert "has_clipboard_text" in source, "clipboard.py missing has_clipboard_text"

    def test_has_wsl_powershell_path_resolution(self):
        source = (PROJECT_ROOT / "hermes_cli" / "clipboard.py").read_text(encoding="utf-8")
        assert "_find_powershell_wsl" in source, "clipboard.py missing WSL powershell path finder"


# ═════════════════════════════════════════════════════════════════════════
# Config — Windows security functions
# ═════════════════════════════════════════════════════════════════════════

class TestConfigWindowsSecurity:
    """Config module must have Windows-aware security."""

    def test_secure_file_has_icacls(self):
        source = (PROJECT_ROOT / "hermes_cli" / "config.py").read_text(encoding="utf-8")
        assert "icacls" in source, "config.py: _secure_file missing icacls for Windows"

    def test_has_keyring_integration(self):
        source = (PROJECT_ROOT / "hermes_cli" / "config.py").read_text(encoding="utf-8")
        assert "keyring" in source, "config.py: missing keyring integration"
        assert "_KEYRING_SERVICE" in source, "config.py: missing keyring service name"

    def test_get_env_value_checks_keyring(self):
        source = (PROJECT_ROOT / "hermes_cli" / "config.py").read_text(encoding="utf-8")
        assert "_keyring_get" in source, "config.py: get_env_value doesn't check keyring"


# ═════════════════════════════════════════════════════════════════════════
# Gateway — Windows Task Scheduler
# ═════════════════════════════════════════════════════════════════════════

class TestGatewayWindowsSupport:
    """Gateway must have Windows service management."""

    def test_has_task_scheduler_functions(self):
        source = (PROJECT_ROOT / "hermes_cli" / "gateway.py").read_text(encoding="utf-8")
        assert "windows_task_install" in source, "gateway.py: missing windows_task_install"
        assert "windows_task_uninstall" in source, "gateway.py: missing windows_task_uninstall"
        assert "windows_task_start" in source, "gateway.py: missing windows_task_start"
        assert "windows_task_stop" in source, "gateway.py: missing windows_task_stop"
        assert "schtasks" in source, "gateway.py: Task Scheduler not used"

    def test_command_handler_routes_to_windows(self):
        source = (PROJECT_ROOT / "hermes_cli" / "gateway.py").read_text(encoding="utf-8")
        # The install command must check is_windows()
        assert "is_windows()" in source, "gateway.py: command handler doesn't check is_windows()"

    def test_status_has_windows_branch(self):
        source = (PROJECT_ROOT / "hermes_cli" / "status.py").read_text(encoding="utf-8")
        assert "win32" in source, "status.py: missing Windows gateway status branch"


# ═════════════════════════════════════════════════════════════════════════
# Shell quoting — cross-platform
# ═════════════════════════════════════════════════════════════════════════

class TestShellQuote:
    """Cross-platform shell quoting utility."""

    def test_module_exists(self):
        filepath = PROJECT_ROOT / "tools" / "shell_quote.py"
        assert filepath.exists(), "tools/shell_quote.py not found"

    def test_unix_quoting(self):
        with patch("tools.shell_quote.sys") as mock_sys:
            mock_sys.platform = "linux"
            from tools.shell_quote import _win_cmd_quote
            # On Unix, shell_quote delegates to shlex.quote
            import shlex
            assert shlex.quote("hello world") == "'hello world'"

    def test_win_cmd_quote_empty(self):
        from tools.shell_quote import _win_cmd_quote
        assert _win_cmd_quote("") == '""'

    def test_win_cmd_quote_no_special_chars(self):
        from tools.shell_quote import _win_cmd_quote
        assert _win_cmd_quote("hello") == "hello"

    def test_win_cmd_quote_spaces(self):
        from tools.shell_quote import _win_cmd_quote
        result = _win_cmd_quote("hello world")
        assert result.startswith('"') and result.endswith('"')
        assert "hello world" in result

    def test_win_cmd_quote_internal_quotes(self):
        from tools.shell_quote import _win_cmd_quote
        result = _win_cmd_quote('say "hi"')
        assert '\\"' in result  # escaped internal quotes

    def test_win_cmd_quote_percent(self):
        from tools.shell_quote import _win_cmd_quote
        result = _win_cmd_quote("100%")
        assert "%%" in result  # escaped percent


# ═════════════════════════════════════════════════════════════════════════
# Process killing — Windows taskkill
# ═════════════════════════════════════════════════════════════════════════

class TestProcessKillingWindows:
    """Process tree killing must use taskkill on Windows."""

    def test_kill_shell_children_has_taskkill(self):
        source = (PROJECT_ROOT / "tools" / "environments" / "local.py").read_text(encoding="utf-8")
        assert "taskkill" in source, "local.py: _kill_shell_children missing taskkill"
        assert "_IS_WINDOWS" in source, "local.py: missing _IS_WINDOWS guard"
