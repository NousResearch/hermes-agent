"""Behavioral tests for Windows-specific compatibility fixes.

Complements ``tests/tools/test_windows_compat.py`` (which does source-level
pattern linting) with cross-platform-mocked tests that exercise the actual
code paths Hermes takes on native Windows.

Runs on Linux CI — every test mocks ``sys.platform``, ``subprocess.run``,
and ``os.kill`` as needed to simulate Windows behavior without requiring a
Windows runner.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

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

    def test_no_op_on_posix(self, monkeypatch):
        from hermes_cli import stdio

        monkeypatch.setattr(stdio, "is_windows", lambda: False)
        result = stdio.configure_windows_stdio()
        assert result is False

    def test_idempotent(self):
        from hermes_cli import stdio

        stdio.configure_windows_stdio()
        # Second call returns False because _CONFIGURED is set
        assert stdio.configure_windows_stdio() is False


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


@pytest.mark.windows_only
class TestTerminatePidRoutingOnWindows:
    """``gateway.status.terminate_pid`` must use taskkill /T /F on Windows.

    ``windows_only``: this used to patch the module-level ``_IS_WINDOWS``
    flag on Linux, which selected the taskkill branch on a host where
    ``taskkill`` does not exist and ``gateway/status`` cannot even import its
    ``msvcrt`` branch. On the Windows runner the flag is genuinely True, so
    only ``subprocess.run`` is mocked — the dependency, not the host.
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

        monkeypatch.setattr(status.subprocess, "run", fake_run)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123456)
        status.terminate_pid(12345, force=True, expected_start_time=123456)

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

        monkeypatch.setattr(status.subprocess, "run", fake_run)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123456)
        with pytest.raises(OSError, match="cannot be terminated"):
            status.terminate_pid(12345, force=True, expected_start_time=123456)

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

        monkeypatch.setattr(status.subprocess, "run", fake_run)
        monkeypatch.setattr(status.os, "kill", fake_kill)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123456)
        status.terminate_pid(42, force=True, expected_start_time=123456)

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


    def test_alive_pid_returns_true(self, monkeypatch):
        from tools.process_registry import ProcessRegistry

        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
        assert ProcessRegistry._is_host_pid_alive(os.getpid()) is True


@pytest.mark.linux_only
class TestPidExistsOSErrorWidening:
    """gateway.status._pid_exists itself must widen Windows errors correctly.

    The POSIX fallback branch (reached when psutil isn't importable) is the
    only path where Python raises ``OSError(WinError 87)`` on Windows for a
    gone PID instead of ``ProcessLookupError``. The function must catch the
    wider ``OSError`` to match POSIX semantics.

    ``linux_only``: the subject is the POSIX fallback branch and its
    ``os.kill`` error handling, exercised with the errno values Windows
    produces. Gating to Linux is what makes ``_IS_WINDOWS`` genuinely False
    here instead of forced false by a patch.
    """

    def test_oserror_gone_pid_returns_false(self, monkeypatch):
        """Simulate Windows' OSError(WinError 87) for a gone PID via the POSIX fallback."""
        from gateway import status

        # Force the psutil-first branch to miss so we exercise the fallback.
        monkeypatch.setitem(
            __import__("sys").modules, "psutil",
            type("P", (), {"pid_exists": staticmethod(lambda pid: (_ for _ in ()).throw(ImportError()))})()
        )

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


    @pytest.mark.windows_only
    def test_windows_detach_flags_exclude_detached_process(self):
        """DETACHED_PROCESS must stay OUT of every detach bundle.

        ``windows_only`` (with ``IS_WINDOWS`` no longer patched): the helpers
        return 0 off Windows, so on Linux the old flag patch was the only
        thing making the bit assertions reachable at all.

        Two reasons (the #54220/#56747 console-flash class):
        1. MSDN: CREATE_NO_WINDOW is IGNORED when combined with
           DETACHED_PROCESS — the hide bit would be dead.
        2. A console-less daemon forces every console-subsystem descendant
           (git, gh, cmd, node, …) to allocate its own visible console — a
           flash per spawn that no per-call-site hide sweep can fully cover.
           CREATE_NO_WINDOW instead gives the daemon one hidden console that
           all descendants inherit (parent-console root cause isolated by
           the desktop backend fix, commit aa2ae36c3f).
        """
        from hermes_cli import _subprocess_compat as sc
        assert not sc.windows_detach_flags() & 0x00000008, (
            "DETACHED_PROCESS must not be in windows_detach_flags(): it makes "
            "CREATE_NO_WINDOW a no-op and re-creates the per-descendant "
            "console flash (#54220/#56747)."
        )
        assert not sc.windows_detach_flags_without_breakaway() & 0x00000008, (
            "DETACHED_PROCESS must not be in the no-breakaway fallback either."
        )

    @pytest.mark.windows_only
    def test_windows_detach_flags_includes_breakaway_from_job(self):
        """CREATE_BREAKAWAY_FROM_JOB is load-bearing for the GUI-driven update path.

        Without it, the gateway-respawn watcher spawned by ``hermes update``
        (which runs under hermes-setup.exe, itself a grandchild of the
        Electron Desktop app) gets reaped when Electron exits and its
        Win32 job object is torn down by the OS.  Result: gateway dies
        during update and never comes back.

        Regression guard against accidentally dropping the breakaway bit
        from the default detach bundle.  This was fixed in
        ``fix/windows-gateway-reliability`` (PR #40909) and the bit must
        stay in the default bundle going forward.
        """
        from hermes_cli import _subprocess_compat as sc
        assert sc.windows_detach_flags() & 0x01000000, (
            "CREATE_BREAKAWAY_FROM_JOB (0x01000000) must remain in the "
            "default detach flag bundle so the Desktop GUI update flow "
            "can respawn the gateway after Electron exits."
        )

    @pytest.mark.windows_only
    def test_windows_detach_flags_without_breakaway_drops_only_that_bit(self):
        """Fallback retry payload for restrictive job objects.

        Some Windows Terminal / container / kiosk configurations refuse
        CREATE_BREAKAWAY_FROM_JOB with ERROR_ACCESS_DENIED.  Callers
        catch ``OSError`` and retry with this payload (see
        ``gateway_windows._spawn_detached`` for the canonical pattern).
        It must drop ONLY the breakaway bit — DETACHED_PROCESS et al.
        are still required for the child to survive the parent's exit.
        """
        from hermes_cli import _subprocess_compat as sc
        full = sc.windows_detach_flags()
        fallback = sc.windows_detach_flags_without_breakaway()
        # Fallback equals full minus the breakaway bit, nothing else changed.
        assert fallback == full & ~0x01000000
        # And the detach bits we still need are present (hidden console, own
        # process group — NOT console-less DETACHED_PROCESS, see
        # test_windows_detach_flags_exclude_detached_process).
        assert fallback & 0x00000200, "fallback missing CREATE_NEW_PROCESS_GROUP"
        assert fallback & 0x08000000, "fallback missing CREATE_NO_WINDOW"


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
        # Every signal installation at module scope must be guarded against
        # missing signals (Windows: SIGPIPE/SIGHUP absent).  Originally this
        # was ``hasattr(signal, "SIGPIPE")`` inline; PR #72677 refactored to
        # ``_install_signal("SIGPIPE", ...)`` which does the same guard via
        # ``getattr(signal, signame, None)`` internally.  Either form is
        # acceptable — what matters is no bare ``signal.signal(SIGPIPE)``
        # at module scope without a guard.
        for sig_name in ("SIGPIPE", "SIGHUP", "SIGTERM", "SIGINT"):
            assert (
                f'hasattr(signal, "{sig_name}")' in source
                or f'_install_signal("{sig_name}"' in source
            ), (
                f"signal {sig_name} must be installed via a guarded path "
                f"(hasattr or _install_signal), not bare signal.signal()"
            )

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
        # Look backwards up to 400 chars for the gate. Accept either form:
        #   `if os.name != "nt":` (run iff POSIX), or
        #   `if os.name == "nt": return []` (early-return guard).
        # Both correctly keep the waitpid loop off Windows; the early-return
        # form is stronger because the rest of the function never runs.
        preamble = source[max(0, idx - 400):idx]
        guard_patterns = (
            'os.name != "nt"',
            "os.name != 'nt'",
            'os.name == "nt"',  # early-return guard
            "os.name == 'nt'",
        )
        assert any(p in preamble for p in guard_patterns), (
            "os.waitpid(-1, os.WNOHANG) must sit behind an os.name guard "
            f"(checked patterns: {guard_patterns})"
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
            "plugins/platforms/whatsapp/adapter.py",
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
    """Sane PATH completion must stay POSIX-only."""

    @pytest.mark.windows_only
    def test_windows_path_is_left_unchanged(self):
        """``windows_only``: the assertion is that a real Windows ``PATH``
        (``;``-separated, drive-lettered) comes back untouched. On Linux the
        old ``_IS_WINDOWS`` patch made the function return early without ever
        meeting a genuine Windows PATH."""
        from tools.environments.local import _append_missing_sane_path_entries

        path = r"C:\Windows\System32;C:\Program Files\Git\bin"
        assert _append_missing_sane_path_entries(path) == path


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


    @pytest.mark.windows_only
    def test_windows_translation(self):
        """On native Windows, /c/Users/... becomes C:\\Users\\...

        ``windows_only``: the function's whole job is producing native
        Windows paths, which is only meaningful where ``os.sep`` is ``\\``.
        """
        import cli as cli_mod
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


    def test_launch_detached_profile_gateway_restart_inlined_watcher_uses_breakaway(self):
        """The inlined respawn script (stringified Python passed to ``python -c``)
        must include CREATE_BREAKAWAY_FROM_JOB so the *respawned gateway* also
        breaks away from any job-object the watcher itself inherits.

        Static check — the watcher source is built at import time and embedded
        verbatim in the module text.  The literal Win32 bits live in
        hermes_cli._subprocess_compat; the watcher must call that helper from
        inside the inlined payload so runtime behavior keeps the breakaway bit.

        The bit was added to the inlined payload by PR #40909.  This test
        ensures a future refactor of the dedent block doesn't silently drop it.
        """
        root = Path(__file__).resolve().parents[2]
        text = (root / "hermes_cli" / "gateway.py").read_text(encoding="utf-8")
        marker = "watcher = textwrap.dedent("
        idx = text.find(marker)
        assert idx != -1, "watcher block not found in gateway.py"
        end = text.find(").strip()", idx)
        assert end != -1, "watcher block end not found"
        block = text[idx:end]
        assert "from hermes_cli._subprocess_compat import" in block
        assert "windows_detach_flags" in block
        assert "windows_detach_flags()" in block, (
            "Inlined respawn watcher must call windows_detach_flags() for the "
            "respawned gateway; that helper carries CREATE_BREAKAWAY_FROM_JOB "
            "so the new gateway is not reaped when the parent job tears down."
        )
        assert "See _subprocess_compat.windows_detach_flags()" in block, (
            "Inlined respawn watcher should keep the breakaway intent greppable "
            "near the helper call."
        )

    def test_launch_detached_profile_gateway_restart_outer_popen_has_access_denied_fallback(
        self,
    ):
        """When the outer watcher Popen raises OSError (breakaway denied by
        the parent job object), the watcher launch must retry without the
        breakaway bit instead of giving up.

        This mirrors the canonical pattern in
        ``gateway_windows._spawn_detached`` and brings the post-update
        watcher path into parity with the gateway-start path: a
        breakaway-denied job object on the parent process (rare but
        possible on Windows Terminal with restrictive job settings,
        containers, kiosk-mode shells) shouldn't take out the entire
        gateway-respawn chain.

        Static check — without standing up a real Windows job object
        with breakaway forbidden, we can't trigger the OSError in a unit
        test.  The textual presence of the fallback helper import +
        ``windows_detach_flags_without_breakaway`` in the fallback path
        is the regression guard.
        """
        root = Path(__file__).resolve().parents[2]
        text = (root / "hermes_cli" / "gateway.py").read_text(encoding="utf-8")
        assert "windows_detach_flags_without_breakaway" in text, (
            "launch_detached_profile_gateway_restart must import "
            "windows_detach_flags_without_breakaway so it can retry a "
            "breakaway-denied Popen without giving up on the watcher."
        )
        # And the inlined watcher's respawn must also handle the denial —
        # check the symbol is referenced INSIDE the watcher block (not
        # just at module scope).
        marker = "watcher = textwrap.dedent("
        idx = text.find(marker)
        end = text.find(").strip()", idx)
        block = text[idx:end]
        assert "except OSError" in block
        assert "windows_detach_flags_without_breakaway()" in block, (
            "Inlined respawn must catch OSError on the breakaway-denied "
            "CreateProcess and retry with windows_detach_flags_without_breakaway(), "
            "matching gateway_windows._spawn_detached's fallback pattern."
        )

    def test_watcher_threads_hidden_console_spec_into_respawn(self):
        """The post-update respawn must route through
        ``gateway_windows.windowless_gateway_restart_spec``.

        The spec supplies the stable cwd + env overlay (HERMES_HOME,
        VIRTUAL_ENV, PYTHONPATH) so the respawned gateway doesn't depend on
        the watcher's transient working directory. (The interpreter itself
        stays the venv's console ``python.exe``, launched hidden via
        CREATE_NO_WINDOW — see the hidden-console rationale in
        ``_subprocess_compat``.)

        Static check: the watcher build (in ``_spawn_gateway_restart_watcher``)
        must invoke the spec helper and thread the cwd / env overlay into
        the inlined respawn ``Popen``.
        """
        root = Path(__file__).resolve().parents[2]
        text = (root / "hermes_cli" / "gateway.py").read_text(encoding="utf-8")
        assert "windowless_gateway_restart_spec" in text, (
            "_spawn_gateway_restart_watcher must build the respawn via "
            "gateway_windows.windowless_gateway_restart_spec so the gateway "
            "comes back with the stable cwd + env overlay."
        )
        marker = "watcher = textwrap.dedent("
        idx = text.find(marker)
        end = text.find(".strip()", idx)
        block = text[idx:end]
        # The inlined respawn must apply the cwd + env overlay so the
        # respawned gateway starts in the stable gateway working dir with
        # the right venv context.
        assert '_popen_kwargs["cwd"]' in block, (
            "Inlined respawn must set cwd from the restart spec so the "
            "gateway starts in the stable gateway working dir."
        )
        assert '_popen_kwargs["env"]' in block, (
            "Inlined respawn must overlay env (VIRTUAL_ENV / PYTHONPATH / "
            "HERMES_HOME) from the restart spec."
        )


class TestWindowlessGatewayRestartSpec:
    """gateway_windows.windowless_gateway_restart_spec — supplies the
    hidden-console respawn spec (normalized interpreter + stable cwd + env
    overlay)."""

    def test_noop_on_non_windows(self):
        import hermes_cli.gateway_windows as gw

        argv = ["/path/venv/bin/python", "-m", "hermes_cli.main", "gateway", "run"]
        new_argv, cwd, env = gw.windowless_gateway_restart_spec(list(argv))
        assert new_argv == argv
        assert cwd == ""
        assert env == {}

    def test_empty_argv_is_safe(self):
        import hermes_cli.gateway_windows as gw

        new_argv, cwd, env = gw.windowless_gateway_restart_spec([])
        assert new_argv == []
        assert cwd == ""
        assert env == {}

    @pytest.mark.windows_only
    def test_windows_keeps_console_python_and_preserves_tail(self):
        """On Windows the console interpreter is kept (hidden-console launch,
        NOT a pythonw swap — #54220/#56747) while every subsequent argument
        is preserved verbatim.

        ``windows_only``: faking this on Linux needed two more fakes to hold
        it up — a pre-import so the lazy ``hermes_cli.gateway`` import didn't
        re-run ``gateway/status``'s ``import msvcrt`` branch, and a mock of
        ``get_hermes_home`` because the real one's ``Path.resolve()`` consults
        sysconfig and blew up under the platform patch. Both workarounds were
        symptoms of testing Windows on a host that isn't Windows; on the
        Windows runner neither is needed.
        """
        import hermes_cli.gateway_windows as gw

        argv = [
            "C:/venv/Scripts/python.exe",
            "-m",
            "hermes_cli.main",
            "--profile",
            "work",
            "gateway",
            "run",
            "--replace",
        ]

        # Only the environment-dependent lookups are stubbed — the host is
        # genuinely Windows here.
        with mock.patch.object(
            gw, "_stable_gateway_working_dir", return_value="C:/hermes"
        ), mock.patch(
            "hermes_cli.config.get_hermes_home", return_value="C:/hermes"
        ):
            new_argv, cwd, env = gw.windowless_gateway_restart_spec(list(argv))

        # Interpreter is kept as the console python — hidden-console launch,
        # no pythonw swap.
        assert new_argv[0] == "C:/venv/Scripts/python.exe"
        # Everything after the interpreter is byte-for-byte preserved.
        assert new_argv[1:] == argv[1:]
        assert cwd == "C:/hermes"
        assert env["VIRTUAL_ENV"] == str(Path("C:/venv"))
        assert "PYTHONPATH" in env


# ---------------------------------------------------------------------------
# gateway/run.py :: GatewayRunner._launch_detached_restart_command
# outer watcher Popen breakaway-denied fallback (PR #42993)
# ---------------------------------------------------------------------------


@pytest.mark.windows_only
class TestGatewayRunRestartWatcherOuterPopenFallback:
    """The Windows ``/restart`` watcher in ``gateway.run`` spawns an outer
    detached ``python -c <watcher>`` process with
    ``windows_detach_popen_kwargs()`` (which carries
    ``CREATE_BREAKAWAY_FROM_JOB``).  A restrictive parent job object rejects
    the breakaway bit with ``ERROR_ACCESS_DENIED`` (surfaced as ``OSError``);
    the launcher must retry once without breakaway, preserving argv and the
    scrubbed environment, and only warn — never crash, never leak secrets —
    if the retry also fails.

    Behavioral: drives the real coroutine with a mocked ``subprocess.Popen``
    rather than asserting on source text.

    ``windows_only``: this used to run on Linux behind a ``sys.platform``
    patch, and the breakaway-bit assertions had to be skipped there anyway
    (``_subprocess_compat`` caches ``IS_WINDOWS`` at import, so the flags
    were all 0) — i.e. the most important assertions in the class never
    executed. On the Windows runner they do.
    """

    @staticmethod
    def _fake_self():
        from types import SimpleNamespace

        return SimpleNamespace(
            _detached_restart_helper_started=False,
            _restart_drain_timeout=0.0,
        )

    @classmethod
    def _drive(cls, gr):
        asyncio.run(
            gr.GatewayRunner._launch_detached_restart_command(cls._fake_self())
        )

    def test_outer_watcher_retries_without_breakaway_on_oserror(self, monkeypatch):
        import gateway.run as gr
        from hermes_cli._subprocess_compat import (
            windows_detach_flags_without_breakaway,
            windows_detach_popen_kwargs,
        )

        monkeypatch.setattr(gr, "_resolve_hermes_bin", lambda: ["hermes"])

        calls = []

        def fake_popen(argv, **kwargs):
            calls.append((argv, kwargs))
            if len(calls) == 1:
                raise OSError(5, "Access is denied")  # ERROR_ACCESS_DENIED
            return MagicMock()

        monkeypatch.setattr("subprocess.Popen", fake_popen)

        self._drive(gr)

        assert len(calls) == 2, "outer watcher must retry exactly once on OSError"
        (argv1, kw1), (argv2, kw2) = calls

        # argv is identical across primary and fallback, and every current
        # watcher parameter survives:
        #   [watcher_python, "-c", <script>, str(pid), str(restart_after_s), *cmd_argv]
        assert argv1 == argv2
        assert argv1[1] == "-c"
        assert argv1[3] == str(os.getpid())
        assert float(argv1[4]) >= 5.0  # restart deadline preserved
        assert argv1[-2:] == ["gateway", "restart"]

        # Scrubbed env preserved and identical on both calls.
        assert kw1["env"] is kw2["env"]
        assert "_HERMES_GATEWAY" not in kw1["env"]

        # Stable, non-flag spawn configuration preserved across both attempts.
        assert kw1["stdout"] is subprocess.DEVNULL
        assert kw1["stderr"] is subprocess.DEVNULL
        assert kw2["stdout"] is subprocess.DEVNULL
        assert kw2["stderr"] is subprocess.DEVNULL

        # Primary spreads the full detach helper.  Assert every returned helper
        # kwarg is present on the call.  The fallback uses the explicit
        # no-breakaway creationflags.
        expected_primary = windows_detach_popen_kwargs()
        for key, value in expected_primary.items():
            assert kw1[key] == value
        assert kw2["creationflags"] == windows_detach_flags_without_breakaway()
        assert "start_new_session" not in kw2

        # The point of the whole fallback: primary asks for breakaway, the
        # retry drops exactly that bit. Reachable now that the flags are real.
        _BREAKAWAY = 0x01000000
        assert kw1["creationflags"] & _BREAKAWAY, (
            "primary spawn must request CREATE_BREAKAWAY_FROM_JOB"
        )
        assert not (kw2["creationflags"] & _BREAKAWAY), (
            "fallback spawn must drop CREATE_BREAKAWAY_FROM_JOB"
        )


    def test_outer_watcher_happy_path_spawns_once(self, monkeypatch):
        import gateway.run as gr

        monkeypatch.setattr(gr, "_resolve_hermes_bin", lambda: ["hermes"])

        calls = []
        monkeypatch.setattr(
            "subprocess.Popen",
            lambda argv, **kwargs: calls.append((argv, kwargs)) or MagicMock(),
        )
        warn = MagicMock()
        monkeypatch.setattr(gr.logger, "warning", warn)

        self._drive(gr)

        assert len(calls) == 1, "no retry when the primary spawn succeeds"
        warn.assert_not_called()

    def test_outer_watcher_dual_failure_warns_without_leaking_secrets(
        self, monkeypatch
    ):
        import gateway.run as gr

        monkeypatch.setattr(gr, "_resolve_hermes_bin", lambda: ["hermes"])

        calls = []

        def always_fail(argv, **kwargs):
            calls.append((argv, kwargs))
            raise OSError(5, "Access is denied")

        monkeypatch.setattr("subprocess.Popen", always_fail)
        warn = MagicMock()
        monkeypatch.setattr(gr.logger, "warning", warn)

        # Deterministic sentinel in the environment the watcher inherits
        # (watcher_env = os.environ.copy()); the warning must never echo it.
        secret = "maxwell-do-not-log-this-secret-42993"
        monkeypatch.setenv("HERMES_TEST_SECRET", secret)

        # Dual failure must NOT propagate — the user's CLI still exits cleanly.
        self._drive(gr)

        assert len(calls) == 2, "both primary and fallback attempted"
        warn.assert_called_once()

        # Secret-safe logging: only (interpreter basename, error field, error
        # code) are logged — never the exception object (str(exc) can carry a
        # path), the argv (watcher source + interpreter path), or env contents.
        argv_used, kwargs_used = calls[0]
        fmt, *log_args = warn.call_args.args
        assert len(log_args) == 3, "warning should log only (basename, field, code)"
        basename_arg, error_field, error_code = log_args
        assert basename_arg == os.path.basename(argv_used[0])
        assert error_field in ("winerror", "errno")
        assert isinstance(error_code, int)
        for arg in log_args:
            assert not isinstance(arg, (OSError, list, dict))

        # The watcher's env carried the sentinel; the rendered warning must not.
        assert secret in (kwargs_used.get("env") or {}).get("HERMES_TEST_SECRET", "")
        rendered = fmt % tuple(log_args)
        assert secret not in rendered
        assert argv_used[2] not in rendered  # watcher script body
        assert "argv" not in fmt.lower()
        assert "env=" not in fmt.lower()
