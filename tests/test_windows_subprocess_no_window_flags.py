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
        if cmd
        and all(
            any(needle in str(part) for part in (cmd if isinstance(cmd, (list, tuple)) else [cmd]))
            for needle in needles
        )
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


@pytest.mark.platforms("windows")
def test_bounded_git_probe_fast_path_spawn_contract_windows(monkeypatch):
    """The normal-path spawn contract survives the run()->Popen rewrite:
    PIPE/PIPE/DEVNULL, text + utf-8/replace, hidden-window flags on Windows.

    ``platforms("windows")``: the ``creationflags`` assertion is the point, and
    ``bounded_git_probe`` only sets that key when ``IS_WINDOWS`` — which the
    helper caches from the real platform at import. ``windows_hide_flags`` is
    still stubbed so the expected value is a fixed constant rather than
    whatever bundle the helper currently returns.

    The seam is the Job-Object container (``local_runtime.processes.spawn_server``),
    which is what the probe hands its spawn contract to on Windows; the container
    itself adds CREATE_SUSPENDED and assigns the real process handle, which a fake
    Popen cannot provide.
    """
    from hermes_cli import _subprocess_compat
    from hermes_cli.local_runtime import processes

    spawns = []
    fake_popen = _make_fake_popen(spawns, stdout="main\n")
    monkeypatch.setattr(_subprocess_compat, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(processes, "spawn_server", lambda cmd, **kw: (fake_popen(cmd, **kw), None))

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




















@pytest.mark.platforms("windows")
def test_shell_hooks_hide_hook_command_windows(monkeypatch):
    """``platforms("windows")``: ``shell_hooks._spawn`` only adds ``creationflags``
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
# probes and CPython
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
    spawns = _spawns(captured, "python3", "--version")
    assert len(spawns) == 1, captured
    cmd, kwargs = spawns[0]
    assert cmd == ["python3", "--version"]
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW
    # The temp-file capture contract (#67964) must survive: stdout/stderr are
    # file objects (not PIPE) so a lingering grandchild can't wedge the probe.
    assert kwargs["stdout"] is not None and kwargs["stdout"] != subprocess.PIPE
    assert kwargs["stderr"] is not None and kwargs["stderr"] != subprocess.PIPE
    assert kwargs["stdin"] == subprocess.DEVNULL


@pytest.mark.platforms("windows")
def test_suppress_platform_ver_console_stubs_syscmd_ver(monkeypatch):
    """``_syscmd_ver`` is replaced by an in-process echo stub so win32_ver()
    takes its ValueError fallback instead of shelling out to `cmd /c ver`.

    ``platforms("windows")``: ``suppress_platform_ver_console()`` is a no-op unless
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


# ── #117781 desktop-startup helper spawns (console-less pythonw backend) ────
#
# The desktop backend runs under pythonw.exe (no console). Every helper it
# spawns (git.exe, tasklist.exe, powershell.exe) without CREATE_NO_WINDOW gets
# a brand-new VISIBLE console — one flash per spawn at startup, during the
# update pipeline, and on the scheduled-task probe. These tests assert the
# WIRING (the site threads windows_hide_flags() into creationflags), so the
# helper is stubbed to a known constant and the tests run on every platform.


def test_gitlock_git_spawns_hide_console_window(monkeypatch, tmp_path):
    """gitlock read-only git probes never flash: stdout-lines queries, the
    cat-file batch pair, the rev-list repair probe and the tag fetch."""
    from hermes_cli import gitlock

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        cmd_str = " ".join(cmd)
        if "rev-list" in cmd_str:
            return _Completed(stdout="", returncode=1)
        if "cat-file" in cmd_str and "--batch-check" in cmd_str:
            return _Completed(stdout="", returncode=0)
        if "cat-file" in cmd_str:
            return _Completed(stdout="", returncode=1)
        return _Completed(stdout="deadbeef\n", returncode=0)

    monkeypatch.setattr(gitlock, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(gitlock.subprocess, "run", fake_run)

    assert gitlock._git_stdout_lines(tmp_path, ["rev-parse", "--git-path", "shallow"]) == ["deadbeef"]
    assert gitlock._batch_missing_parents(tmp_path, ["abc123"]) == set()
    gitlock.repair_broken_shallow_boundaries(tmp_path)

    spawns = _spawns(captured, "git")
    assert spawns, "expected git spawns"
    for cmd, kwargs in spawns:
        assert kwargs.get("creationflags") == _CREATE_NO_WINDOW, cmd


def test_gitlock_tasklist_probe_hides_console_window(monkeypatch):
    """The stale-lock tasklist guard flashes on its own under pythonw."""
    from hermes_cli import gitlock

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        return _Completed(stdout="\"git.exe\",\"1234\",\"Console\",\"1\",\"10,000 K\"\n", returncode=0)

    monkeypatch.setattr(gitlock, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(gitlock.subprocess, "run", fake_run)
    monkeypatch.setattr(gitlock.os, "name", "nt")

    assert gitlock._git_proc_running() is True
    spawns = _spawns(captured, "tasklist")
    assert len(spawns) == 1, captured
    assert spawns[0][1]["creationflags"] == _CREATE_NO_WINDOW


def test_gitlock_tag_fetch_hides_console_window(monkeypatch, tmp_path):
    """The full-commit-graph fetch (update pipeline) hides its window."""
    from hermes_cli import gitlock

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        if "cat-file" in " ".join(cmd):
            return _Completed(stdout="", returncode=1)
        return _Completed(stdout="8192, 4096, deadbeef\n", returncode=0)

    monkeypatch.setattr(gitlock, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(gitlock, "_shallow_file_path", lambda root: None)
    monkeypatch.setattr(gitlock, "_partial_clone_filter", lambda root, **kw: None)
    monkeypatch.setattr(gitlock.subprocess, "run", fake_run)

    assert gitlock.fetch_full_commit_graph(tmp_path) is False  # not shallow
    spawns = _spawns(captured, "fetch")
    assert len(spawns) == 1, captured
    assert spawns[0][1]["creationflags"] == _CREATE_NO_WINDOW


def test_gateway_scheduled_task_state_hides_powershell_window(monkeypatch):
    """Task Scheduler COM probe spawns powershell.exe during backend startup."""
    from hermes_cli import gateway

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        return _Completed(stdout="Ready\n", returncode=0)

    _patch_hide_flags(monkeypatch)
    monkeypatch.setattr(gateway, "is_windows", lambda: True)
    monkeypatch.setattr(gateway.subprocess, "run", fake_run)
    monkeypatch.setattr(gateway.shutil, "which", lambda name: r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.EXE")

    assert gateway._windows_scheduled_task_state("hermes-gateway") == "Ready"
    assert len(captured) == 1, captured
    cmd, kwargs = captured[0]
    assert cmd[0].lower().endswith("powershell.exe"), cmd
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW


def test_update_git_run_hides_console_window(monkeypatch, tmp_path):
    """The central update git runner (26 call sites) hides its window."""
    from hermes_cli import update_cmd

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        return _Completed(stdout="ok\n", returncode=0)

    _patch_hide_flags(monkeypatch)
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=tmp_path))
    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    update_cmd._git_run(["git"], ["rev-parse", "HEAD"], tmp_path)
    update_cmd._git_run(["git"], ["fetch", "origin", "main"], tmp_path, network=True)
    assert len(captured) == 2, captured
    for cmd, kwargs in captured:
        assert kwargs.get("creationflags") == _CREATE_NO_WINDOW, cmd


def test_no_prompt_git_kwargs_hide_console_window(monkeypatch):
    """Network git kwargs (fetch/pull behind them) carry the hide flags too,
    so the upstream-sync fetch/pull spawns never flash."""
    from hermes_cli import update_cmd

    _patch_hide_flags(monkeypatch)
    kwargs = update_cmd._no_prompt_git_kwargs()
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW
    assert kwargs["stdin"] == subprocess.DEVNULL


def test_update_git_plumbing_spawns_hide_console_window(monkeypatch, tmp_path):
    """rev-parse label, fork-bomb probe and EOL-normalization git spawns."""
    from hermes_cli import update_cmd_git

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        return _Completed(stdout="main\nabc123\n", returncode=0)

    monkeypatch.setattr(update_cmd_git, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(update_cmd_git, "_GIT_TEXT_KW",
                        dict(capture_output=True, text=True, encoding="utf-8", errors="replace",
                             creationflags=_CREATE_NO_WINDOW))
    monkeypatch.setattr(update_cmd_git.subprocess, "run", fake_run)

    assert update_cmd_git._probe_fork_bomb(["git"]) is False
    label = update_cmd_git._branch_head_label(["git"], cwd=tmp_path)
    assert label is not None and "abc" in label

    assert captured, "expected git spawns"
    for cmd, kwargs in captured:
        assert kwargs.get("creationflags") == _CREATE_NO_WINDOW, cmd


@pytest.mark.real_safe_directory
def test_safe_directory_git_config_probes_hide_console_window(monkeypatch):
    """safe.directory replay (two git config children per internal git call)
    never flashes."""
    from hermes_cli import _subprocess_compat

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        return _Completed(stdout="", returncode=1)

    _subprocess_compat._safe_directory_cache.clear()
    monkeypatch.setattr(_subprocess_compat, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(_subprocess_compat.subprocess, "run", fake_run)

    assert _subprocess_compat._user_safe_directories({}) == []
    spawns = _spawns(captured, "git", "config")
    assert len(spawns) == 2, captured
    for cmd, kwargs in spawns:
        assert kwargs.get("creationflags") == _CREATE_NO_WINDOW, cmd


# ── #101895 nvidia-smi statusbar / hardware probes ──────────────────────────
#
# The desktop GPU statusbar polls /api/local-models/hardware every 5s. Before
# the consolidation that endpoint spawned console-subsystem nvidia-smi twice
# per poll (VRAM budget in hardware._nvidia_vram, then name/util in the
# router) from a windowless backend — two console flashes per poll. Everything
# now funnels through one cached, hidden spawn.


def test_cached_nvidia_gpu_query_hides_console_window_and_caches(monkeypatch):
    """One hidden nvidia-smi spawn serves the budget probe, the endpoint and
    the vendor detector within the TTL; the second caller must not spawn."""
    from hermes_cli.local_runtime import bootstrap, hardware
    from hermes_cli.web_routers import local_models

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        return _Completed(stdout="0, GPU-ab-01, 8192, 4096, NVIDIA GeForce RTX 4060 Ti, 0x250410DE, 4096, 12\n", returncode=0)

    monkeypatch.setattr(hardware, "_gpu_query_cache", None)
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: r"C:\Windows\System32\nvidia-smi.exe")
    _patch_hide_flags(monkeypatch)
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    vram = hardware._nvidia_vram()
    assert vram == (8192 << 20, 4096 << 20, "NVIDIA GeForce RTX 4060 Ti", 0x250410DE)
    facts = local_models._nvidia_smi_facts()
    assert facts == dict(gpu_name="NVIDIA GeForce RTX 4060 Ti", gpu_util_percent=12, vram_used_bytes=4096 << 20)
    vendor = bootstrap._detect_gpu_vendor()
    assert vendor == "nvidia NVIDIA GeForce RTX 4060 Ti"

    spawns = _spawns(captured, "nvidia-smi")
    assert len(spawns) == 1, captured  # one poll window, one spawn
    assert spawns[0][1]["creationflags"] == _CREATE_NO_WINDOW
    assert spawns[0][1]["capture_output"] is True


def test_cached_nvidia_gpu_query_failure_is_cached(monkeypatch):
    """A missing/failed smi must not re-spawn per poll either."""
    from hermes_cli.local_runtime import hardware

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        return _Completed(stdout="", returncode=1)

    monkeypatch.setattr(hardware, "_gpu_query_cache", None)
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: r"C:\Windows\System32\nvidia-smi.exe")
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware._nvidia_vram() is None
    assert hardware._cached_nvidia_gpu_query() is None  # TTL hit, no second spawn
    assert len(captured) == 1, captured
