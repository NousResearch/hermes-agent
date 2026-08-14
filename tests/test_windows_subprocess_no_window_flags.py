from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


_CREATE_NO_WINDOW = 0x08000000


class _Completed:
    def __init__(self, stdout: str | bytes = "ok\n", returncode: int = 0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def _spawns(captured, *needles):
    """Captured ``subprocess.run`` calls whose argv contains every needle.

    These tests patch ``<module>.subprocess.run``, which is the shared
    ``subprocess`` module singleton — so the patch is process-wide. Importing
    ``tui_gateway.server`` kicks off ``prefetch_update_check`` (a daemon thread
    that shells out to ``git ... origin`` with ``text=True, timeout=5``), and
    that call can land in ``captured`` mid-test. Matching the distinctive argv
    tokens of the call under test (e.g. ``--show-toplevel``, ``ls-files``) keeps
    each assertion scoped to its own contract and immune to that cross-talk —
    otherwise a stray ``git`` spawn trips a bare ``KeyError: 'creationflags'``
    or a call-count / full-list mismatch.
    """
    return [
        (cmd, kwargs)
        for cmd, kwargs in captured
        if cmd and all(n in cmd for n in needles)
    ]


def _is_git_spawn(cmd) -> bool:
    """True only for a ``git -C <cwd> ...`` spawn.

    ``bounded_git_probe`` lives in ``hermes_cli._subprocess_compat`` and both
    probe call sites delegate to it, so these tests patch
    ``_subprocess_compat.subprocess.Popen`` — which is the shared ``subprocess``
    module singleton, i.e. a process-wide patch. Any unrelated daemon spawn
    (e.g. an import-time update-check thread) must stay benign and out of the
    recorded spawns, mirroring the ``_spawns`` scoping the other tests use.
    """
    return bool(cmd) and cmd[:2] == ["git", "-C"]


def _make_fake_popen(spawns, *, stdout="ok\n", returncode=0):
    """Fast-path Popen stand-in: git returns within the budget."""

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            if _is_git_spawn(cmd):
                spawns.append((cmd, kwargs))
            self.returncode = returncode

        def communicate(self, timeout=None):
            return (stdout, "")

        def kill(self):  # pragma: no cover - never reached on the fast path
            raise AssertionError("kill() must not run when git returns in time")

    return _FakePopen


@pytest.mark.windows_only
def test_bounded_git_probe_fast_path_spawn_contract_windows(monkeypatch):
    """The normal-path spawn contract survives the run()->Popen rewrite:
    PIPE/PIPE/DEVNULL, text + utf-8/replace, hidden-window flags on Windows.

    ``windows_only``: the ``creationflags`` assertion is the point, and
    ``bounded_git_probe`` only sets that key when ``IS_WINDOWS`` — which the
    helper caches from the real platform at import. ``windows_hide_flags`` is
    still stubbed so the expected value is a fixed constant rather than
    whatever bundle the helper currently returns.
    """
    from hermes_cli import _subprocess_compat

    spawns = []
    monkeypatch.setattr(_subprocess_compat, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(_subprocess_compat.subprocess, "Popen", _make_fake_popen(spawns, stdout="main\n"))

    out = _subprocess_compat.bounded_git_probe(
        ["git", "-C", "C:/repo", "branch", "--show-current"], timeout=1.5
    )
    assert out == "main"
    assert len(spawns) == 1, spawns
    cmd, kwargs = spawns[0]
    assert cmd == ["git", "-C", "C:/repo", "branch", "--show-current"]
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW




def test_bounded_git_probe_nonzero_returncode_returns_empty(monkeypatch):
    from hermes_cli import _subprocess_compat

    spawns = []
    monkeypatch.setattr(
        _subprocess_compat.subprocess,
        "Popen",
        _make_fake_popen(spawns, stdout="garbage-should-not-leak\n", returncode=1),
    )

    assert _subprocess_compat.bounded_git_probe(["git", "-C", "/repo", "status"], timeout=1.5) == ""












def test_bounded_git_probe_spawn_failure_returns_empty(monkeypatch):
    """A spawn failure (git not on PATH) fails open to ""."""
    from hermes_cli import _subprocess_compat

    def boom(cmd, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(_subprocess_compat.subprocess, "Popen", boom)

    assert _subprocess_compat.bounded_git_probe(["git", "-C", "/repo", "status"], timeout=1.5) == ""




















@pytest.mark.windows_only
def test_shell_hooks_hide_hook_command_windows(monkeypatch):
    """``windows_only``: ``shell_hooks._spawn`` only adds ``creationflags``
    under its module-level ``IS_WINDOWS``, so on Linux the flag patch was
    what created the thing being asserted."""
    from agent import shell_hooks

    captured = []

    class FakeProc:
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return "{}", ""

    def fake_popen(cmd, **kwargs):
        captured.append((cmd, kwargs))
        return FakeProc()

    monkeypatch.setattr(shell_hooks, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(shell_hooks.subprocess, "Popen", fake_popen)

    result = shell_hooks._spawn(
        shell_hooks.ShellHookSpec(event="post_tool_call", command="hook-bin --flag"),
        "{}",
    )

    assert result["returncode"] == 0
    assert captured[0][1]["creationflags"] == _CREATE_NO_WINDOW
    # The POSIX-only process_group kwarg must NOT reach a Windows spawn.
    assert "process_group" not in captured[0][1]


def test_agent_browser_npx_warmup_hides_npx_window(monkeypatch):
    """warm_agent_browser_npx_cache spawns via subprocess.Popen (not .run,
    since the T3 security-hardening rewrite added process-tree containment
    via Popen + communicate()) — the console-hiding flag must still survive
    that rewrite. On Windows the real implementation now ORs in
    CREATE_NEW_PROCESS_GROUP alongside windows_hide_flags()'s bits (for
    _kill_process_tree's taskkill /T to have a coherent tree to kill), so
    this checks the CREATE_NO_WINDOW bit is present rather than exact
    equality with the whole creationflags value."""
    from tools import browser_tool

    captured = []

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured.append((cmd, kwargs))
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("1.2.3\n", "")

    monkeypatch.setattr(
        browser_tool.shutil, "which",
        lambda name, path=None: "/usr/bin/npx",
    )
    monkeypatch.setattr(browser_tool, "node_tool_runnable", lambda p: True)
    monkeypatch.setattr(browser_tool, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(browser_tool.subprocess, "Popen", _FakePopen)

    assert browser_tool.warm_agent_browser_npx_cache() is True
    assert captured[0][0][0] == "/usr/bin/npx"
    assert captured[0][1]["creationflags"] & _CREATE_NO_WINDOW == _CREATE_NO_WINDOW




# ── #56747 GUI-reachable exec paths + provider transports (PR #56877) ──────
#
# These six sites are the desktop-GUI-reachable spawns that still flashed a
# console on Windows after the #54220 sweep: the TUI gateway's cli.exec /
# shell.exec / quick-command exec RPCs, the interactive CLI's quick-command
# exec handler, and the Copilot ACP + Codex app-server stdio transports.
# All are hide-only (creationflags) — PIPE stdio must stay intact.


def _patch_hide_flags(monkeypatch):
    """Pin ``windows_hide_flags()`` to a known constant.

    The spawn sites these tests cover call ``windows_hide_flags()``
    unconditionally and pass the result straight through, so what is under
    test is the WIRING — that the site threads the helper's value into
    ``creationflags`` — not the platform. Stubbing only the helper keeps that
    coverage on the Linux lane; no ``IS_WINDOWS`` fake is needed or wanted.
    """
    import hermes_cli._subprocess_compat as subprocess_compat

    monkeypatch.setattr(subprocess_compat, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)




def test_tui_shell_exec_rpc_hides_console_window(monkeypatch):
    from tui_gateway import server

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((cmd, kwargs))
        return _Completed(stdout="ok\n")

    _patch_hide_flags(monkeypatch)
    monkeypatch.setattr(server.subprocess, "run", fake_run)

    resp = server.handle_request(
        {"id": "2", "method": "shell.exec", "params": {"command": "echo shellexec-56747"}}
    )
    assert resp["result"]["code"] == 0

    spawns = _spawns(captured, "shellexec-56747")
    assert len(spawns) == 1, captured
    assert spawns[0][1]["creationflags"] == _CREATE_NO_WINDOW










# ── #47971 LSP spawn + installer paths (salvage) ────────────────────────────
#
# The LSP language-server spawn (agent/lsp/client.py::_spawn) and the
# npm/go LSP auto-installers (agent/lsp/install.py) are reachable from
# console-less parents — a VS Code/Zed extension host running the ACP
# adapter — where a .cmd-wrapped server (pyright-langserver.CMD via
# cmd.exe /c) or an npm/go console app flashes a window on Windows.
# All are hide-only (creationflags); PIPE stdio must stay intact and the
# POSIX start_new_session detach must be preserved on the client spawn.


def test_lsp_client_spawn_hides_console_window(monkeypatch):
    import asyncio

    from agent.lsp import client as lsp_client

    captured = []

    class _FakeProc:
        stdin = None
        stdout = None
        stderr = None

    async def fake_exec(*cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        return _FakeProc()

    monkeypatch.setattr(lsp_client, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(
        lsp_client.asyncio, "create_subprocess_exec", fake_exec
    )

    client = lsp_client.LSPClient(
        server_id="test-server",
        workspace_root="/tmp/ws",
        command=["fake-langserver", "--stdio"],
    )
    asyncio.run(client._spawn())

    assert len(captured) == 1, captured
    cmd, kwargs = captured[0]
    assert cmd == ["fake-langserver", "--stdio"]
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW
    # Hide-only: the LSP wire still needs its pipes, and the POSIX
    # process-group detach (mcp orphan-sweep guard) must survive.
    assert kwargs["stdin"] == asyncio.subprocess.PIPE
    assert kwargs["stdout"] == asyncio.subprocess.PIPE
    assert kwargs["start_new_session"] is True






# ── #67690 env probes, lazy installs, platform.win32_ver() (@m4r13y) ───────
#
# Windowless processes (pythonw gateway + kanban workers) flashed consoles
# from three more spawn families: tools/env_probe._run's interpreter/pip
# probes, tools/lazy_deps' uv→pip→ensurepip install ladder, and CPython
# 3.11/3.12's platform.win32_ver() which shells out `cmd /c ver` with
# shell=True and no CREATE_NO_WINDOW. All are hide-only (creationflags);
# win32_ver is neutralized by stubbing platform._syscmd_ver so the
# documented ValueError fallback reads sys.getwindowsversion() instead.


def test_env_probe_run_hides_console_window(monkeypatch):
    from tools import env_probe

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((cmd, kwargs))
        return _Completed(stdout="", returncode=0)

    monkeypatch.setattr(env_probe, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(env_probe.subprocess, "run", fake_run)

    rc, out, err = env_probe._run(["python3", "--version"], timeout=1.0)

    assert rc == 0
    assert len(captured) == 1, captured
    cmd, kwargs = captured[0]
    assert cmd == ["python3", "--version"]
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW
    # The temp-file capture contract (#67964) must survive: stdout/stderr are
    # file objects (not PIPE) so a lingering grandchild can't wedge the probe.
    assert kwargs["stdout"] is not None and kwargs["stdout"] != subprocess.PIPE
    assert kwargs["stderr"] is not None and kwargs["stderr"] != subprocess.PIPE
    assert kwargs["stdin"] == subprocess.DEVNULL


def test_lazy_deps_uv_install_hides_console_window(monkeypatch):
    from tools import lazy_deps

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((cmd, kwargs))
        return _Completed(stdout="installed", returncode=0)

    monkeypatch.delenv(lazy_deps._LAZY_TARGET_ENV, raising=False)
    monkeypatch.setattr(lazy_deps, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(lazy_deps.subprocess, "run", fake_run)
    monkeypatch.setattr(lazy_deps.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    res = lazy_deps._venv_pip_install(("left-pad",))

    assert res.success
    spawns = _spawns(captured, "pip", "install", "left-pad")
    assert len(spawns) == 1, captured
    cmd, kwargs = spawns[0]
    assert cmd[:3] == ["/usr/bin/uv", "pip", "install"]
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW
    assert kwargs["stdin"] == subprocess.DEVNULL








@pytest.mark.windows_only
def test_suppress_platform_ver_console_stubs_syscmd_ver(monkeypatch):
    """``_syscmd_ver`` is replaced by an in-process echo stub so win32_ver()
    takes its ValueError fallback instead of shelling out to `cmd /c ver`.

    ``windows_only``: ``suppress_platform_ver_console()`` is a no-op unless
    ``IS_WINDOWS``, and the console flash it prevents (``cmd /c ver``) only
    exists on Windows — the old flag patch installed the stub on a host where
    ``win32_ver`` is never consulted at all.
    """
    import platform

    from hermes_cli import _subprocess_compat

    # Register the original with monkeypatch so it gets restored after.
    monkeypatch.setattr(platform, "_syscmd_ver", platform._syscmd_ver)

    _subprocess_compat.suppress_platform_ver_console()

    # The stub echoes its inputs — win32_ver() treats the unparseable value
    # as the documented ValueError path and falls back to
    # sys.getwindowsversion().platform_version (no subprocess, no window).
    assert platform._syscmd_ver("s", "r", "v") == ("s", "r", "v")
    # Idempotent + never raises on repeat calls.
    _subprocess_compat.suppress_platform_ver_console()
    assert platform._syscmd_ver() == ("", "", "")
