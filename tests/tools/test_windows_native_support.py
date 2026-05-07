"""Behavioral tests for Windows-specific compatibility fixes.

Complements ``tests/tools/test_windows_compat.py`` (which does source-level
pattern linting) with cross-platform-mocked tests that exercise the actual
code paths Hermes takes on native Windows.

Runs on Linux CI — every test mocks ``sys.platform``, ``subprocess.run``,
and ``os.kill`` as needed to simulate Windows behavior without requiring a
Windows runner.
"""

from __future__ import annotations

import importlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# configure_windows_stdio
# ---------------------------------------------------------------------------


class TestConfigureWindowsStdio:
    """``hermes_cli.stdio.configure_windows_stdio`` wiring.

    The function must:
    - be a no-op on non-Windows
    - only configure once per process (idempotent)
    - set PYTHONIOENCODING / PYTHONUTF8 without overriding explicit user settings
    - reconfigure sys.stdout/stderr/stdin to UTF-8 on Windows
    - flip the console code page to CP_UTF8 (65001) via ctypes
    - respect HERMES_DISABLE_WINDOWS_UTF8 opt-out
    """

    @pytest.fixture(autouse=True)
    def _reset_configured(self, monkeypatch):
        """Reload the module before each test so the _CONFIGURED flag resets."""
        # Remove from sys.modules so import triggers a fresh load
        sys.modules.pop("hermes_cli.stdio", None)
        # Fresh import now; tests import from hermes_cli.stdio themselves,
        # but this guarantees the module they get is a brand-new copy.
        import hermes_cli.stdio as _s
        _s._CONFIGURED = False
        yield
        sys.modules.pop("hermes_cli.stdio", None)

    def test_no_op_on_posix(self):
        from hermes_cli import stdio

        assert stdio.is_windows() is False
        result = stdio.configure_windows_stdio()
        assert result is False

    def test_idempotent(self):
        from hermes_cli import stdio

        stdio.configure_windows_stdio()
        # Second call returns False because _CONFIGURED is set
        assert stdio.configure_windows_stdio() is False

    def test_windows_path_sets_env_and_reconfigures_streams(self, monkeypatch):
        from hermes_cli import stdio

        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        # Pretend the user has no prior setting
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
        monkeypatch.delenv("PYTHONUTF8", raising=False)
        monkeypatch.delenv("HERMES_DISABLE_WINDOWS_UTF8", raising=False)
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.delenv("VISUAL", raising=False)

        reconfigure_calls = []

        def fake_reconfigure(stream, *, encoding="utf-8", errors="replace"):
            reconfigure_calls.append((stream, encoding, errors))

        cp_calls = []

        def fake_flip():
            cp_calls.append(True)

        monkeypatch.setattr(stdio, "_reconfigure_stream", fake_reconfigure)
        monkeypatch.setattr(stdio, "_flip_console_code_page_to_utf8", fake_flip)
        # Pretend notepad.exe is on PATH (it always is on real Windows hosts,
        # but not on the Linux CI runner — mock it so the editor default
        # survives).
        monkeypatch.setattr(stdio, "_default_windows_editor", lambda: "notepad")

        result = stdio.configure_windows_stdio()
        assert result is True
        assert os.environ.get("PYTHONIOENCODING") == "utf-8"
        assert os.environ.get("PYTHONUTF8") == "1"
        # EDITOR must be set so prompt_toolkit's open_in_editor finds
        # a working program on Windows (it defaults to /usr/bin/nano).
        assert os.environ.get("EDITOR") == "notepad"
        assert len(cp_calls) == 1  # SetConsoleOutputCP path hit
        assert len(reconfigure_calls) == 3  # stdout, stderr, stdin

    def test_respects_existing_editor_var(self, monkeypatch):
        """User's explicit EDITOR wins over our default."""
        from hermes_cli import stdio

        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        monkeypatch.setenv("EDITOR", "code --wait")
        monkeypatch.setattr(stdio, "_reconfigure_stream", lambda *a, **kw: None)
        monkeypatch.setattr(stdio, "_flip_console_code_page_to_utf8", lambda: None)
        monkeypatch.setattr(stdio, "_default_windows_editor", lambda: "notepad")

        stdio.configure_windows_stdio()
        assert os.environ["EDITOR"] == "code --wait"

    def test_respects_existing_visual_var(self, monkeypatch):
        """VISUAL takes precedence over our EDITOR default too."""
        from hermes_cli import stdio

        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.setenv("VISUAL", "nvim")
        monkeypatch.setattr(stdio, "_reconfigure_stream", lambda *a, **kw: None)
        monkeypatch.setattr(stdio, "_flip_console_code_page_to_utf8", lambda: None)
        monkeypatch.setattr(stdio, "_default_windows_editor", lambda: "notepad")

        stdio.configure_windows_stdio()
        # EDITOR should NOT be set when VISUAL already is (prompt_toolkit
        # checks VISUAL first anyway, but we also shouldn't override it).
        assert os.environ.get("EDITOR", "") != "notepad"
        assert os.environ["VISUAL"] == "nvim"

    def test_respects_existing_env_var(self, monkeypatch):
        """User's explicit PYTHONIOENCODING wins over our default."""
        from hermes_cli import stdio

        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        monkeypatch.setenv("PYTHONIOENCODING", "latin-1")
        monkeypatch.setattr(stdio, "_reconfigure_stream", lambda *a, **kw: None)
        monkeypatch.setattr(stdio, "_flip_console_code_page_to_utf8", lambda: None)

        stdio.configure_windows_stdio()
        assert os.environ["PYTHONIOENCODING"] == "latin-1"

    @pytest.mark.parametrize("optout", ["1", "true", "True", "yes"])
    def test_disable_flag_short_circuits(self, monkeypatch, optout):
        from hermes_cli import stdio

        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        monkeypatch.setenv("HERMES_DISABLE_WINDOWS_UTF8", optout)

        reconfigure_hit = []
        monkeypatch.setattr(
            stdio,
            "_reconfigure_stream",
            lambda *a, **kw: reconfigure_hit.append(True),
        )

        result = stdio.configure_windows_stdio()
        assert result is False
        assert reconfigure_hit == [], "opt-out must skip stream reconfiguration"

    def test_reconfigure_stream_handles_missing_method(self, monkeypatch):
        """StringIO-like objects without .reconfigure() must not blow up."""
        from hermes_cli import stdio
        import io

        buf = io.StringIO()
        # Must not raise
        stdio._reconfigure_stream(buf)


# ---------------------------------------------------------------------------
# terminate_pid — the centralized kill primitive
# ---------------------------------------------------------------------------


class TestTerminatePidRoutingOnWindows:
    """``gateway.status.terminate_pid`` must use taskkill /T /F on Windows.

    On Linux we can't reload gateway/status with sys.platform=win32 because
    the module unconditionally imports ``msvcrt`` in that branch.  Instead
    we patch the module-level ``_IS_WINDOWS`` flag and ``subprocess.run``
    on the already-loaded module, which exercises the same branching code.
    """

    def test_force_uses_taskkill_on_windows(self, monkeypatch):
        from gateway import status

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            result.stdout = ""
            return result

        monkeypatch.setattr(status, "_IS_WINDOWS", True)
        monkeypatch.setattr(status.subprocess, "run", fake_run)
        status.terminate_pid(12345, force=True)

        assert captured["args"][0] == "taskkill"
        assert "/PID" in captured["args"]
        assert "12345" in captured["args"]
        assert "/T" in captured["args"]
        assert "/F" in captured["args"]

    def test_force_taskkill_failure_raises_oserror(self, monkeypatch):
        from gateway import status

        def fake_run(args, **kwargs):
            result = MagicMock()
            result.returncode = 128
            result.stderr = "ERROR: The process cannot be terminated."
            result.stdout = ""
            return result

        monkeypatch.setattr(status, "_IS_WINDOWS", True)
        monkeypatch.setattr(status.subprocess, "run", fake_run)
        with pytest.raises(OSError, match="cannot be terminated"):
            status.terminate_pid(12345, force=True)

    def test_graceful_on_windows_uses_os_kill_sigterm(self, monkeypatch):
        """Non-force path calls os.kill with SIGTERM (Windows has no SIGKILL).

        ``terminate_pid(pid)`` with force=False bypasses the taskkill branch
        and uses ``os.kill`` directly — so platform doesn't actually matter
        for the signal choice.  Verifies the getattr fallback works.
        """
        from gateway import status

        captured = {}

        def fake_kill(pid, sig):
            captured["pid"] = pid
            captured["sig"] = sig

        monkeypatch.setattr(status.os, "kill", fake_kill)
        status.terminate_pid(99, force=False)

        assert captured["pid"] == 99
        assert captured["sig"] == signal.SIGTERM

    def test_taskkill_not_found_falls_back_to_os_kill(self, monkeypatch):
        """On Windows without taskkill (WinPE, containers), fall back gracefully."""
        from gateway import status

        captured = {}

        def fake_run(args, **kwargs):
            raise FileNotFoundError(2, "taskkill not found")

        def fake_kill(pid, sig):
            captured["pid"] = pid
            captured["sig"] = sig

        monkeypatch.setattr(status, "_IS_WINDOWS", True)
        monkeypatch.setattr(status.subprocess, "run", fake_run)
        monkeypatch.setattr(status.os, "kill", fake_kill)
        status.terminate_pid(42, force=True)

        assert captured["pid"] == 42
        assert captured["sig"] == signal.SIGTERM


# ---------------------------------------------------------------------------
# SIGKILL fallback pattern
# ---------------------------------------------------------------------------


class TestSigkillFallback:
    """Modules that want SIGKILL must fall back to SIGTERM when absent."""

    def test_getattr_fallback_works_when_sigkill_missing(self, monkeypatch):
        """The `getattr(signal, "SIGKILL", signal.SIGTERM)` pattern."""
        # Build a stand-in signal module with no SIGKILL attribute
        fake_signal = MagicMock()
        del fake_signal.SIGKILL  # ensure it's absent
        fake_signal.SIGTERM = 15

        result = getattr(fake_signal, "SIGKILL", fake_signal.SIGTERM)
        assert result == 15

    def test_getattr_fallback_prefers_sigkill_when_present(self):
        """On POSIX the fallback is a no-op: real SIGKILL wins."""
        result = getattr(signal, "SIGKILL", signal.SIGTERM)
        assert result == signal.SIGKILL

    @pytest.mark.parametrize(
        "module_path, line_pattern",
        [
            ("hermes_cli.kanban_db", 'getattr(signal, "SIGKILL", signal.SIGTERM)'),
        ],
    )
    def test_module_uses_getattr_fallback(self, module_path, line_pattern):
        """Source-level check that our modules use the safe fallback."""
        rel = module_path.replace(".", "/") + ".py"
        root = Path(__file__).resolve().parents[2]
        source = (root / rel).read_text(encoding="utf-8")
        assert line_pattern in source, (
            f"{rel} must use the getattr fallback pattern on its SIGKILL site"
        )


# ---------------------------------------------------------------------------
# OSError widening on os.kill(pid, 0) probes
# ---------------------------------------------------------------------------


class TestProcessRegistryOSErrorWidening:
    """_is_host_pid_alive must treat Windows' OSError as 'not alive'."""

    def test_oserror_treated_as_not_alive(self, monkeypatch):
        from tools.process_registry import ProcessRegistry

        def fake_kill(pid, sig):
            # Simulate Windows' WinError 87 for an unknown PID
            raise OSError(22, "Invalid argument")

        monkeypatch.setattr("tools.process_registry.os.kill", fake_kill)
        assert ProcessRegistry._is_host_pid_alive(12345) is False

    def test_permission_error_treated_as_not_alive(self, monkeypatch):
        """Conservative: PermissionError also means 'not alive' (matches existing behavior)."""
        from tools.process_registry import ProcessRegistry

        def fake_kill(pid, sig):
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr("tools.process_registry.os.kill", fake_kill)
        assert ProcessRegistry._is_host_pid_alive(12345) is False

    def test_zero_or_none_pid_returns_false_without_calling_kill(self, monkeypatch):
        """No wasted syscall on falsy pids."""
        from tools.process_registry import ProcessRegistry

        kill_calls = []
        monkeypatch.setattr(
            "tools.process_registry.os.kill",
            lambda pid, sig: kill_calls.append(pid),
        )
        assert ProcessRegistry._is_host_pid_alive(None) is False
        assert ProcessRegistry._is_host_pid_alive(0) is False
        assert kill_calls == []

    def test_alive_pid_returns_true(self, monkeypatch):
        from tools.process_registry import ProcessRegistry

        # os.kill returning None (default) means "probe succeeded → pid alive"
        monkeypatch.setattr("tools.process_registry.os.kill", lambda pid, sig: None)
        assert ProcessRegistry._is_host_pid_alive(os.getpid()) is True


# ---------------------------------------------------------------------------
# tzdata dependency
# ---------------------------------------------------------------------------


class TestTzdataDependencyDeclared:
    """Windows installs must pull tzdata for zoneinfo to work."""

    def test_pyproject_declares_tzdata_for_win32(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "pyproject.toml").read_text(encoding="utf-8")
        # The dependency line should be conditional on sys_platform == 'win32'
        # and should NOT be in the core dependencies for Linux/macOS.
        assert (
            'tzdata>=2023.3; sys_platform == \'win32\'' in source
            or "tzdata>=2023.3; sys_platform == 'win32'" in source
            or 'tzdata>=2023.3; sys_platform == "win32"' in source
        ), "tzdata must be a Windows-only dep in pyproject.toml dependencies"


# ---------------------------------------------------------------------------
# README / docs consistency
# ---------------------------------------------------------------------------


class TestReadmeNoLongerSaysWindowsUnsupported:
    """The README shouldn't claim native Windows isn't supported."""

    def test_readme_does_not_say_not_supported(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "README.md").read_text(encoding="utf-8")
        # Previous string (removed in this PR): "Native Windows is not supported"
        assert "Native Windows is not supported" not in source, (
            "README.md still says native Windows is not supported — update the "
            "install copy to reflect the PowerShell installer."
        )

    def test_readme_mentions_powershell_installer(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "README.md").read_text(encoding="utf-8")
        assert "install.ps1" in source, (
            "README.md must point at scripts/install.ps1 for Windows users"
        )


# ---------------------------------------------------------------------------
# pty_bridge graceful import on Windows
# ---------------------------------------------------------------------------


class TestWebServerPtyBridgeGuard:
    """The web server must not crash if pty_bridge can't import (Windows)."""

    def test_import_guard_present_in_source(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "hermes_cli" / "web_server.py").read_text(encoding="utf-8")
        assert "_PTY_BRIDGE_AVAILABLE" in source
        assert "except ImportError" in source, (
            "web_server.py must wrap the pty_bridge import in try/except ImportError"
        )

    def test_pty_handler_checks_availability_flag(self):
        """The /api/pty handler must short-circuit when the bridge is unavailable."""
        root = Path(__file__).resolve().parents[2]
        source = (root / "hermes_cli" / "web_server.py").read_text(encoding="utf-8")
        assert "if not _PTY_BRIDGE_AVAILABLE" in source, (
            "/api/pty handler must return a friendly error when PTY is unavailable"
        )


# ---------------------------------------------------------------------------
# Entry points wire configure_windows_stdio
# ---------------------------------------------------------------------------


class TestEntryPointsConfigureStdio:
    """cli.py, hermes_cli/main.py, gateway/run.py must call configure_windows_stdio."""

    @pytest.mark.parametrize(
        "relpath",
        ["cli.py", "hermes_cli/main.py", "gateway/run.py"],
    )
    def test_entry_point_calls_configure_stdio(self, relpath):
        root = Path(__file__).resolve().parents[2]
        source = (root / relpath).read_text(encoding="utf-8")
        assert "configure_windows_stdio" in source, (
            f"{relpath} must call hermes_cli.stdio.configure_windows_stdio() "
            "early in startup so Windows consoles render Unicode without crashing"
        )
