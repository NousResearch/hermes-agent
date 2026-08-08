from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


_CREATE_NO_WINDOW = 0x08000000


def _run_with_retained_stdin(cmd, *, env, cwd, timeout=10):
    """Run *cmd* while keeping the stdin writer open (ACP JSON-RPC-like)."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
    )
    try:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            stderr = proc.stderr.read() if proc.stderr else ""
            raise AssertionError(
                f"helper hung under retained stdin writer (stderr={stderr!r})"
            )
        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    finally:
        if proc.stdin:
            proc.stdin.close()


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


def test_bounded_git_probe_fast_path_spawn_contract_windows(monkeypatch):
    """The normal-path spawn contract survives the run()->Popen rewrite:
    PIPE/PIPE/DEVNULL, text + utf-8/replace, hidden-window flags on Windows."""
    from hermes_cli import _subprocess_compat

    spawns = []
    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
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
    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", False)
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

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", False)
    monkeypatch.setattr(_subprocess_compat.subprocess, "Popen", boom)

    assert _subprocess_compat.bounded_git_probe(["git", "-C", "/repo", "status"], timeout=1.5) == ""


def test_bounded_captured_run_fast_path_spawn_contract_windows(monkeypatch):
    """``bounded_captured_run`` keeps PIPE/PIPE/DEVNULL + hide flags on Windows."""
    from hermes_cli import _subprocess_compat

    spawns = []

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            spawns.append((cmd, kwargs))
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("ok\n", "warn\n")

        def kill(self):  # pragma: no cover
            raise AssertionError("kill() must not run when the child returns in time")

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(_subprocess_compat, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(_subprocess_compat.subprocess, "Popen", _FakePopen)

    result = _subprocess_compat.bounded_captured_run(
        ["bash", "--noprofile", "--norc", "-c", "true"], timeout=15
    )
    assert result.returncode == 0
    assert result.stdout == "ok\n"
    assert result.stderr == "warn\n"
    assert len(spawns) == 1
    _cmd, kwargs = spawns[0]
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW


def test_bounded_captured_run_timeout_tree_kills_on_windows(monkeypatch):
    """Timeout path: taskkill tree first, then kill fallback, then bounded drain."""
    from hermes_cli import _subprocess_compat

    events = []

    class _HangingPopen:
        def __init__(self, cmd, **kwargs):
            self.returncode = None
            self.pid = 5150

        def communicate(self, timeout=None):
            events.append(f"comm:{timeout}")
            if timeout != 1:
                raise subprocess.TimeoutExpired(cmd="bash", timeout=timeout)
            return ("", "")

        def kill(self):
            events.append("kill")

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "taskkill":
            events.append(("taskkill", list(cmd)))
        return _Completed()

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(_subprocess_compat, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(_subprocess_compat.subprocess, "Popen", _HangingPopen)
    monkeypatch.setattr(_subprocess_compat.subprocess, "run", fake_run)

    result = _subprocess_compat.bounded_captured_run(["bash", "-c", "true"], timeout=15)
    assert result.returncode == -1
    assert result.stdout == ""
    assert result.stderr == "timed out after 15s"
    # Windows: walk the live tree before killing the launcher, then drain.
    assert events == [
        "comm:15",
        ("taskkill", ["taskkill", "/T", "/F", "/PID", "5150"]),
        "kill",
        "comm:1",
    ]


def test_bounded_git_probe_timeout_returns_empty(monkeypatch):
    """Wrapper must fail open to "" when the shared helper times out."""
    from hermes_cli import _subprocess_compat

    def _timeout_run(argv, *, timeout, env=None):
        return subprocess.CompletedProcess(list(argv), -1, "", f"timed out after {timeout}s")

    monkeypatch.setattr(_subprocess_compat, "bounded_captured_run", _timeout_run)
    assert _subprocess_compat.bounded_git_probe(["git", "status"], timeout=1.5) == ""


def test_bounded_captured_run_returns_under_retained_stdin():
    """ACP/TUI hosts keep stdin open; probes must use DEVNULL or hang forever."""
    blocker = (
        "import sys\n"
        "# Block if stdin is an open pipe with no EOF (inherited ACP stdin).\n"
        "sys.stdin.buffer.read(1)\n"
        "raise SystemExit(0)\n"
    )
    helper = (
        "from hermes_cli._subprocess_compat import bounded_captured_run\n"
        "import sys\n"
        f"r = bounded_captured_run([sys.executable, '-c', {blocker!r}], timeout=5)\n"
        "print(r.returncode)\n"
    )
    repo_root = str(Path(__file__).resolve().parents[1])
    proc = _run_with_retained_stdin(
        [sys.executable, "-c", helper],
        env={**os.environ, "PYTHONPATH": repo_root},
        cwd=repo_root,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0"


def test_shell_hooks_hide_hook_command_windows(monkeypatch):
    from agent import shell_hooks

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(shell_hooks, "IS_WINDOWS", True)
    monkeypatch.setattr(shell_hooks, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(shell_hooks.subprocess, "run", fake_run)

    result = shell_hooks._spawn(
        shell_hooks.ShellHookSpec(event="post_tool_call", command="hook-bin --flag"),
        "{}",
    )

    assert result["returncode"] == 0
    assert captured[0][1]["creationflags"] == _CREATE_NO_WINDOW













# ── #56747 GUI-reachable exec paths + provider transports (PR #56877) ──────
#
# These six sites are the desktop-GUI-reachable spawns that still flashed a
# console on Windows after the #54220 sweep: the TUI gateway's cli.exec /
# shell.exec / quick-command exec RPCs, the interactive CLI's quick-command
# exec handler, and the Copilot ACP + Codex app-server stdio transports.
# All are hide-only (creationflags) — PIPE stdio must stay intact.


def _patch_hide_flags(monkeypatch):
    import hermes_cli._subprocess_compat as subprocess_compat

    monkeypatch.setattr(subprocess_compat, "IS_WINDOWS", True)
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








def test_suppress_platform_ver_console_stubs_syscmd_ver(monkeypatch):
    """Simulated Windows: _syscmd_ver is replaced by an in-process echo stub
    so win32_ver() takes its ValueError fallback instead of `cmd /c ver`."""
    import platform

    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
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
