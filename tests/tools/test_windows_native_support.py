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
# OSError widening on liveness probes
#
# Post-#21561, ``ProcessRegistry._is_host_pid_alive`` delegates to
# ``gateway.status._pid_exists``, which is the cross-platform liveness
# primitive (psutil-first, ctypes/os.kill fallback). The tests below assert
# (a) the delegation is correct and (b) ``_pid_exists`` correctly widens
# Windows' ``OSError(WinError 87)`` / ``PermissionError`` behavior on the
# POSIX fallback branch.
# ---------------------------------------------------------------------------


class TestProcessRegistryOSErrorWidening:
    """_is_host_pid_alive delegates to gateway.status._pid_exists."""

    def test_oserror_treated_as_not_alive(self, monkeypatch):
        """_pid_exists → False propagates as _is_host_pid_alive → False."""
        from tools.process_registry import ProcessRegistry

        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
        assert ProcessRegistry._is_host_pid_alive(12345) is False

    def test_permission_error_treated_as_alive(self, monkeypatch):
        """PermissionError is encoded by _pid_exists as alive=True; propagates as-is.

        This is a meaningful semantic change from the pre-#21561 version of
        this test (which asserted PermissionError → not-alive). The old
        ``os.kill(pid, 0)``-based probe couldn't distinguish "gone" from
        "owned by another user" on some platforms, so it conservatively
        returned False. The new psutil-based probe CAN distinguish them via
        ``OpenProcess + ERROR_ACCESS_DENIED`` on Windows / ``except
        PermissionError`` on POSIX, so alive=True is correct.
        """
        from tools.process_registry import ProcessRegistry

        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
        assert ProcessRegistry._is_host_pid_alive(12345) is True

    def test_zero_or_none_pid_returns_false_without_probing(self, monkeypatch):
        """No wasted syscall on falsy pids."""
        from tools.process_registry import ProcessRegistry

        probes = []
        monkeypatch.setattr(
            "gateway.status._pid_exists",
            lambda pid: probes.append(pid) or True,
        )
        assert ProcessRegistry._is_host_pid_alive(None) is False
        assert ProcessRegistry._is_host_pid_alive(0) is False
        assert probes == []

    def test_alive_pid_returns_true(self, monkeypatch):
        from tools.process_registry import ProcessRegistry

        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
        assert ProcessRegistry._is_host_pid_alive(os.getpid()) is True


class TestPidExistsOSErrorWidening:
    """gateway.status._pid_exists itself must widen Windows errors correctly.

    The POSIX fallback branch (reached when psutil isn't importable) is the
    only path where Python raises ``OSError(WinError 87)`` on Windows for a
    gone PID instead of ``ProcessLookupError``. The function must catch the
    wider ``OSError`` to match POSIX semantics.
    """

    def test_oserror_gone_pid_returns_false(self, monkeypatch):
        """Simulate Windows' OSError(WinError 87) for a gone PID via the POSIX fallback."""
        from gateway import status

        # Force the psutil-first branch to miss so we exercise the fallback.
        monkeypatch.setitem(
            __import__("sys").modules, "psutil",
            type("P", (), {"pid_exists": staticmethod(lambda pid: (_ for _ in ()).throw(ImportError()))})()
        )
        monkeypatch.setattr(status, "_IS_WINDOWS", False)

        def fake_kill(pid, sig):
            raise OSError(22, "Invalid argument")

        monkeypatch.setattr(status.os, "kill", fake_kill)
        assert status._pid_exists(12345) is False

    def test_permission_error_returns_true(self, monkeypatch):
        """POSIX fallback: PermissionError means alive (owned by another user)."""
        from gateway import status

        monkeypatch.setitem(
            __import__("sys").modules, "psutil",
            type("P", (), {"pid_exists": staticmethod(lambda pid: (_ for _ in ()).throw(ImportError()))})()
        )
        monkeypatch.setattr(status, "_IS_WINDOWS", False)

        def fake_kill(pid, sig):
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(status.os, "kill", fake_kill)
        assert status._pid_exists(12345) is True


# ---------------------------------------------------------------------------
# tzdata dependency
# ---------------------------------------------------------------------------


class TestTzdataDependencyDeclared:
    """Windows installs must pull tzdata for zoneinfo to work."""

    def test_pyproject_declares_tzdata_for_win32(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "pyproject.toml").read_text(encoding="utf-8")
        # The dependency line should be conditional on sys_platform == 'win32'
        # and should NOT be in the core dependencies for Linux/macOS. We do
        # not care about the exact pinned version (which is bumped over time)
        # — only that tzdata is declared with a win32 marker. This is an
        # invariant check, not a snapshot test.
        import re
        # Match `"tzdata` … `; sys_platform == 'win32'"` allowing any version
        # specifier in between (==X.Y.Z, >=X.Y.Z,<W, etc.) and either quote
        # style on the marker.
        pattern = re.compile(
            r'"tzdata[^"]*;\s*sys_platform\s*==\s*[\'"]win32[\'"]\s*"'
        )
        assert pattern.search(source), (
            "tzdata must be a Windows-only dep in pyproject.toml dependencies "
            "(declared with a `; sys_platform == 'win32'` marker)"
        )


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


# ---------------------------------------------------------------------------
# _subprocess_compat shared helpers
# ---------------------------------------------------------------------------


class TestSubprocessCompatHelpers:
    """hermes_cli/_subprocess_compat.py POSIX + Windows behaviour."""

    def test_is_windows_matches_sys_platform(self):
        from hermes_cli import _subprocess_compat as sc
        assert sc.IS_WINDOWS == (sys.platform == "win32")

    def test_resolve_node_command_returns_absolute_on_posix(self):
        """On Linux, resolve_node_command('sh', ['-c','echo hi']) picks up /bin/sh."""
        from hermes_cli._subprocess_compat import resolve_node_command
        # We can't assert "npm is on PATH" portably; use `sh` which is
        # guaranteed on POSIX.  On Windows the test only confirms the
        # no-crash fallback path.
        argv = resolve_node_command("sh", ["-c", "echo hi"])
        assert argv[1:] == ["-c", "echo hi"]
        # First element is either an absolute path (sh found) or the bare
        # name (fallback) — both are acceptable behaviours.

    def test_resolve_node_command_fallback_when_absent(self):
        from hermes_cli._subprocess_compat import resolve_node_command
        argv = resolve_node_command(
            "zzz-definitely-not-on-path-xyzzy", ["--help"]
        )
        # Must fall back to the bare name — NOT return None, NOT crash.
        assert argv[0] == "zzz-definitely-not-on-path-xyzzy"
        assert argv[1:] == ["--help"]

    def test_windows_flags_zero_on_posix(self):
        from hermes_cli._subprocess_compat import (
            windows_detach_flags,
            windows_hide_flags,
        )
        if sys.platform != "win32":
            assert windows_detach_flags() == 0
            assert windows_hide_flags() == 0

    def test_windows_detach_popen_kwargs_is_posix_equivalent_on_posix(self):
        from hermes_cli._subprocess_compat import windows_detach_popen_kwargs
        kwargs = windows_detach_popen_kwargs()
        if sys.platform != "win32":
            # POSIX path MUST produce start_new_session=True, which maps to
            # os.setsid() in the child — identical to the unchanged main
            # branch behaviour.  Do NOT break Linux/macOS here.
            assert kwargs == {"start_new_session": True}
        else:
            # Windows path must include creationflags with all 3 bits set.
            assert "creationflags" in kwargs
            assert kwargs["creationflags"] != 0
            # No start_new_session on Windows (silently no-op there).
            assert "start_new_session" not in kwargs

    def test_windows_detach_flags_has_expected_win32_bits(self, monkeypatch):
        """Simulate Windows to verify flag bundle."""
        from hermes_cli import _subprocess_compat as sc
        monkeypatch.setattr(sc, "IS_WINDOWS", True)
        flags = sc.windows_detach_flags()
        # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_NO_WINDOW
        assert flags & 0x00000200, "missing CREATE_NEW_PROCESS_GROUP"
        assert flags & 0x00000008, "missing DETACHED_PROCESS"
        assert flags & 0x08000000, "missing CREATE_NO_WINDOW"


# ---------------------------------------------------------------------------
# tui_gateway/entry.py signal installation survives absent POSIX signals
# ---------------------------------------------------------------------------


class TestTuiGatewayEntrySignalGuards:
    """Importing tui_gateway.entry must not crash when SIGPIPE/SIGHUP absent.

    Linux has both signals, so this is mostly a source-level invariant check
    (no bare ``signal.SIGPIPE`` at module level without a ``hasattr`` guard).
    On Windows the import would have raised AttributeError before this fix.
    """

    def test_source_guards_each_signal_installation(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "tui_gateway" / "entry.py").read_text(encoding="utf-8")
        # Every signal.signal(...) at module scope must be preceded by a
        # hasattr check.  We look at the text: no bare "signal.signal("
        # call should appear outside a function body without a guard.
        # Simpler heuristic: all SIGPIPE / SIGHUP references outside the
        # dict-building loop must be wrapped in hasattr.
        assert 'hasattr(signal, "SIGPIPE")' in source
        assert 'hasattr(signal, "SIGHUP")' in source
        assert 'hasattr(signal, "SIGTERM")' in source
        assert 'hasattr(signal, "SIGINT")' in source

    def test_module_imports_cleanly(self):
        """Importing the module must not raise — verifies the guards work."""
        # Drop any cached import so the module re-initialises
        for mod in list(sys.modules):
            if mod.startswith("tui_gateway"):
                del sys.modules[mod]
        import tui_gateway.entry  # noqa: F401  # must not raise


# ---------------------------------------------------------------------------
# hermes_cli/kanban_db.py waitpid guard
# ---------------------------------------------------------------------------


class TestKanbanWaitpidWindowsGuard:
    """os.WNOHANG doesn't exist on Windows — the dispatcher tick reap loop
    must be gated behind ``os.name != "nt"``."""

    def test_source_gates_waitpid_loop(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "hermes_cli" / "kanban_db.py").read_text(encoding="utf-8")
        # Find the waitpid call and confirm it's inside a POSIX gate.
        idx = source.find("os.waitpid(-1, os.WNOHANG)")
        assert idx > 0, "waitpid call must exist"
        # Look backwards up to 400 chars for the gate.
        preamble = source[max(0, idx - 400):idx]
        assert 'os.name != "nt"' in preamble or "os.name != 'nt'" in preamble, (
            "os.waitpid(-1, os.WNOHANG) must sit behind an os.name != 'nt' guard"
        )


# ---------------------------------------------------------------------------
# code_execution_tool TCP loopback on Windows
# ---------------------------------------------------------------------------


class TestCodeExecutionTransportTcpFallback:
    """The RPC transport must fall back to TCP on Windows.

    We can't easily execute the sandbox on Linux CI in Windows mode, but we
    CAN assert that the generated client module supports both AF_UNIX and
    AF_INET endpoints based on the HERMES_RPC_SOCKET format.
    """

    def test_generated_client_handles_tcp_endpoint(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "tools" / "code_execution_tool.py").read_text(encoding="utf-8")
        # _UDS_TRANSPORT_HEADER body must parse both transports.
        assert 'endpoint.startswith("tcp://")' in source, (
            "generated sandbox client must accept tcp:// endpoints for Windows"
        )
        assert "socket.AF_INET" in source, (
            "generated sandbox client must be able to open AF_INET sockets"
        )

    def test_server_side_branches_on_use_tcp_rpc(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "tools" / "code_execution_tool.py").read_text(encoding="utf-8")
        assert "_use_tcp_rpc = _IS_WINDOWS" in source
        assert 'rpc_endpoint = f"tcp://{_host}:{_port}"' in source


# ---------------------------------------------------------------------------
# cron/scheduler.py /bin/bash dynamic resolution
# ---------------------------------------------------------------------------


class TestCronSchedulerBashResolution:
    """cron.scheduler must NOT hardcode /bin/bash — .sh scripts need a
    dynamically-resolved bash so Windows (Git Bash) works."""

    def test_source_uses_shutil_which_for_bash(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "cron" / "scheduler.py").read_text(encoding="utf-8")
        # The old hardcoded path should be gone as the sole bash source.
        # It may still appear as a POSIX fallback after shutil.which(), so
        # we check for the shutil.which call near the .sh/.bash branch.
        assert 'shutil.which("bash")' in source, (
            "cron.scheduler must resolve bash dynamically via shutil.which"
        )

    def test_error_message_when_bash_missing(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "cron" / "scheduler.py").read_text(encoding="utf-8")
        # The graceful-failure message must mention "bash not found" so
        # Windows users without Git Bash see an actionable error instead
        # of a WinError 2 traceback.
        assert "bash not found" in source.lower()


# ---------------------------------------------------------------------------
# Node-ecosystem launcher resolution (npm / npx / node)
# ---------------------------------------------------------------------------


class TestNpmBareSpawnsResolved:
    """Every spawn site that launches ``npm``/``npx`` must resolve via
    shutil.which / hermes_cli._subprocess_compat.resolve_node_command
    so Windows can execute the .cmd batch shims."""

    @pytest.mark.parametrize(
        "relpath",
        [
            "hermes_cli/tools_config.py",
            "hermes_cli/doctor.py",
            "gateway/platforms/whatsapp.py",
            "tools/browser_tool.py",
        ],
    )
    def test_no_bare_npm_or_npx_in_popen_argv(self, relpath):
        """Reject ``subprocess.run(["npm", ...])`` / ``["npx", ...]`` patterns.

        Those fail on Windows with WinError 193.  Callers must resolve
        via shutil.which(...) and pass the absolute path (or fall back
        to the bare name only as a last resort behind a variable).
        """
        root = Path(__file__).resolve().parents[2]
        source = (root / relpath).read_text(encoding="utf-8")
        # The forbidden literal: a subprocess invocation that names npm
        # or npx as a bare string inside an argv list.
        forbidden_patterns = [
            '["npm",',
            '["npx",',
            "['npm',",
            "['npx',",
        ]
        for pat in forbidden_patterns:
            # Exception: strings inside error-message text or comments are fine.
            # We only fail if the literal appears in an argv position, which
            # we approximate by checking it isn't inside a print/log/comment.
            # Find all occurrences and verify they're behind shutil.which.
            idx = 0
            while True:
                idx = source.find(pat, idx)
                if idx < 0:
                    break
                # Look at the preceding 120 chars — if "shutil.which" appears
                # there, or the pattern is inside a comment/string, it's fine.
                context = source[max(0, idx - 120):idx]
                if "#" in context.split("\n")[-1]:
                    idx += len(pat)
                    continue
                # Argv forms that START with a bare npm/npx are the bug.
                raise AssertionError(
                    f"{relpath}: bare {pat!r} still present at offset {idx} — "
                    f"resolve via shutil.which(...) so Windows can execute .cmd shims"
                )


# ---------------------------------------------------------------------------
# tools/environments/local.py Windows temp dir & PATH injection
# ---------------------------------------------------------------------------


class TestLocalEnvironmentWindowsTempDir:
    """LocalEnvironment.get_temp_dir must return a native Windows path on
    Windows, NOT the POSIX ``/tmp`` literal (which Python can't open)."""

    def test_posix_path_preserved_on_linux(self):
        """Linux/macOS behaviour MUST be unchanged — return / tmp or
        tempfile.gettempdir()-derived POSIX path.  This is the 'do no harm'
        test — regressions here break every Unix user's terminal tool."""
        from tools.environments.local import LocalEnvironment

        env = LocalEnvironment(cwd="/tmp", timeout=10, env={})
        tmp_dir = env.get_temp_dir()
        if sys.platform != "win32":
            assert tmp_dir.startswith("/"), (
                f"POSIX temp dir must start with '/'; got {tmp_dir!r}"
            )

    def test_source_has_windows_branch_using_hermes_home(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "tools" / "environments" / "local.py").read_text(encoding="utf-8")
        assert "if _IS_WINDOWS:" in source
        assert "get_hermes_home" in source
        assert 'cache_dir = get_hermes_home() / "cache" / "terminal"' in source


class TestLocalEnvironmentPathInjectionGated:
    """The /usr/bin PATH injection in _make_run_env must be POSIX-only."""

    def test_source_gates_path_injection(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "tools" / "environments" / "local.py").read_text(encoding="utf-8")
        # The fix wraps the injection in `if not _IS_WINDOWS`.
        assert 'not _IS_WINDOWS and "/usr/bin" not in existing_path.split(":")' in source


# ---------------------------------------------------------------------------
# cli.py git path normalization
# ---------------------------------------------------------------------------


class TestGitBashPathNormalization:
    """_normalize_git_bash_path should turn /c/Users/... into C:\\Users\\...
    on Windows and leave paths unchanged on POSIX."""

    def test_posix_noop(self):
        """Must NOT mutate paths on Linux/macOS."""
        from cli import _normalize_git_bash_path
        if sys.platform != "win32":
            assert _normalize_git_bash_path("/home/teknium/foo") == "/home/teknium/foo"
            assert _normalize_git_bash_path("/c/Users/foo") == "/c/Users/foo"
            assert _normalize_git_bash_path("C:/Users/foo") == "C:/Users/foo"
            assert _normalize_git_bash_path(None) is None

    def test_empty_string_preserved(self):
        from cli import _normalize_git_bash_path
        assert _normalize_git_bash_path("") == ""

    def test_windows_translation(self, monkeypatch):
        """Simulate Windows and verify /c/Users/... becomes C:\\Users\\..."""
        import cli as cli_mod
        monkeypatch.setattr(cli_mod.sys, "platform", "win32")
        assert cli_mod._normalize_git_bash_path("/c/Users/foo") == r"C:\Users\foo"
        assert cli_mod._normalize_git_bash_path("/C/Users/foo") == r"C:\Users\foo"
        assert cli_mod._normalize_git_bash_path("/cygdrive/d/data") == r"D:\data"
        assert cli_mod._normalize_git_bash_path("/mnt/c/Users") == r"C:\Users"
        # Already-native path is preserved
        assert cli_mod._normalize_git_bash_path(r"C:\Users\foo") == r"C:\Users\foo"
        # Forward-slash Windows path is preserved (git on Windows often
        # returns this form; it's valid for both bash and Python, so we
        # don't need to translate).
        assert cli_mod._normalize_git_bash_path("C:/Users/foo") == "C:/Users/foo"


class TestWorktreeSymlinkFallback:
    """.worktreeinclude directory symlinks must fall back to copytree on
    Windows (where symlink creation requires admin / Dev Mode)."""

    def test_source_has_symlink_fallback(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "cli.py").read_text(encoding="utf-8")
        # Look for the try/except that handles OSError around os.symlink
        # with a shutil.copytree fallback.
        assert "os.symlink(str(src_resolved), str(dst))" in source
        assert "except (OSError, NotImplementedError)" in source
        assert "shutil.copytree" in source
        assert 'sys.platform == "win32"' in source


# ---------------------------------------------------------------------------
# Gateway detached watcher — Windows creationflags
# ---------------------------------------------------------------------------


class TestGatewayDetachedWatcherWindowsFlags:
    """launch_detached_profile_gateway_restart and the in-gateway update
    launcher must use CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS on
    Windows, not silent start_new_session=True."""

    def test_hermes_cli_gateway_uses_compat_kwargs(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "hermes_cli" / "gateway.py").read_text(encoding="utf-8")
        assert "windows_detach_popen_kwargs" in source, (
            "hermes_cli/gateway.py must use the platform-aware detach helper"
        )
        # The legacy start_new_session=True on the outer Popen should be
        # replaced by **windows_detach_popen_kwargs(). Inside the watcher
        # STRING the old pattern is replaced by explicit creationflags.
        assert "**windows_detach_popen_kwargs()" in source

    def test_gateway_run_update_has_windows_branch(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "gateway" / "run.py").read_text(encoding="utf-8")
        # Both the /restart and /update paths must have sys.platform=='win32' branches.
        assert 'if sys.platform == "win32":' in source
        # Windows branch uses windows_detach_popen_kwargs
        assert "windows_detach_popen_kwargs" in source
