"""Tests for tools.browser_tool.warm_agent_browser_npx_cache (#43564, security
hardening follow-up on PR #44772 review).

warm_agent_browser_npx_cache() is the fire-and-forget helper `hermes update` /
`hermes doctor --fix` call to pre-fetch agent-browser via npx so the first real
browser-tool invocation in a session doesn't pay npx's registry-lookup cost.
It must never raise, must accurately report success/failure via its return
value, must use a credential-scrubbed and PATH-propagated environment (it
runs registry-fetched, potentially install-scripted npm code on every
`hermes update` — not only when a browser tool is actually used), must pass
--ignore-scripts (AGENT_BROWSER_NPX_SPEC is a floating ^0.26.0 range, not an
exact pin), and must kill the whole process tree — not just the top-level
npx PID — on timeout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import tools.browser_tool as bt
from tools.browser_tool import (
    AGENT_BROWSER_NPX_SPEC,
    _legacy_kill_process_tree,
    warm_agent_browser_npx_cache,
)


def _mock_proc(returncode=0, communicate_side_effect=None, pid=4242):
    proc = MagicMock()
    proc.pid = pid
    if communicate_side_effect is not None:
        proc.communicate.side_effect = communicate_side_effect
    else:
        proc.communicate.return_value = ("", "")
    proc.returncode = returncode
    return proc


def test_lock_path_is_shared_by_profiles_using_the_same_npm_cache(tmp_path):
    cache = tmp_path / "shared-npm-cache"
    first = bt._agent_browser_npx_lock_path(
        {"NPM_CONFIG_CACHE": str(cache), "HERMES_HOME": str(tmp_path / "profile-a")}
    )
    second = bt._agent_browser_npx_lock_path(
        {"NPM_CONFIG_CACHE": str(cache), "HERMES_HOME": str(tmp_path / "profile-b")}
    )

    assert first == second
    assert first.parent == cache


def test_warmup_holds_shared_cache_lock_until_npx_exits():
    state = {"held": False}

    @contextmanager
    def fake_lock(_env):
        state["held"] = True
        try:
            yield True
        finally:
            state["held"] = False

    proc = _mock_proc()

    def communicate(*, timeout):
        assert timeout == 60.0
        assert state["held"] is True
        return "", ""

    proc.communicate.side_effect = communicate
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), \
         patch("tools.browser_tool._agent_browser_npx_cache_lock", side_effect=fake_lock), \
         patch("subprocess.Popen", return_value=proc):
        assert warm_agent_browser_npx_cache() is True

    assert state["held"] is False


def test_lock_failure_fails_closed_without_spawning():
    @contextmanager
    def unavailable_lock(_env):
        yield False

    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), \
         patch("tools.browser_tool._agent_browser_npx_cache_lock", side_effect=unavailable_lock), \
         patch("subprocess.Popen") as mock_popen:
        assert warm_agent_browser_npx_cache() is False

    mock_popen.assert_not_called()


def test_posix_default_lock_path_follows_npm_home_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("os.name", "posix")
    path = bt._agent_browser_npx_lock_path({"HOME": str(tmp_path)})

    assert path == Path(tmp_path) / ".npm" / ".hermes-agent-browser-warmup.lock"


def test_windows_cache_lock_retries_transient_contention(tmp_path, monkeypatch):
    """msvcrt.LK_LOCK gives up after about ten seconds, while a valid cold
    warm-up can take longer.  The lock layer must keep serializing instead of
    letting the caller fall through to a concurrent npx invocation."""
    fake_msvcrt = MagicMock()
    fake_msvcrt.LK_NBLCK = 1
    fake_msvcrt.LK_UNLCK = 2
    fake_msvcrt.locking.side_effect = [
        OSError(13, "lock held by another process"),
        None,
        None,
    ]
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(
        bt,
        "os",
        SimpleNamespace(
            name="nt",
            path=os.path,
            environ=os.environ,
            SEEK_END=os.SEEK_END,
        ),
    )

    with patch("tools.browser_tool.time.sleep") as sleep:
        with bt._agent_browser_npx_cache_lock(
            {"NPM_CONFIG_CACHE": str(tmp_path / "shared-cache")}
        ) as acquired:
            assert acquired is True

    assert [call.args[1] for call in fake_msvcrt.locking.call_args_list] == [1, 1, 2]
    sleep.assert_called_once_with(0.1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX flock regression")
def test_cache_lock_serializes_two_processes(tmp_path):
    trace = tmp_path / "trace.log"
    cache = tmp_path / "shared-cache"
    script = (
        "import sys,time\n"
        "from tools.browser_tool import _agent_browser_npx_cache_lock\n"
        "env={'NPM_CONFIG_CACHE': sys.argv[1]}\n"
        "with _agent_browser_npx_cache_lock(env) as acquired:\n"
        " assert acquired\n"
        " with open(sys.argv[2], 'a', encoding='utf-8', buffering=1) as f:\n"
        "  f.write('enter ' + sys.argv[3] + '\\n')\n"
        "  time.sleep(0.2)\n"
        "  f.write('exit ' + sys.argv[3] + '\\n')\n"
    )
    env = dict(os.environ)
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(cache), str(trace), label],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for label in ("a", "b")
    ]
    results = [process.communicate(timeout=10) for process in processes]

    assert [process.returncode for process in processes] == [0, 0], results
    events = trace.read_text(encoding="utf-8").splitlines()
    assert events in (
        ["enter a", "exit a", "enter b", "exit b"],
        ["enter b", "exit b", "enter a", "exit a"],
    )


def test_returns_false_without_spawning_when_npx_unresolvable():
    with patch("tools.browser_tool._resolve_npx_bin", return_value=None), patch(
        "subprocess.Popen"
    ) as mock_popen:
        assert warm_agent_browser_npx_cache() is False
    mock_popen.assert_not_called()


def test_invokes_npx_with_ignore_scripts_prefer_offline_and_pinned_spec():
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), patch(
        "subprocess.Popen", return_value=_mock_proc()
    ) as mock_popen:
        assert warm_agent_browser_npx_cache() is True

    mock_popen.assert_called_once()
    args, _kwargs = mock_popen.call_args
    assert args[0] == [
        "/usr/bin/npx", "--ignore-scripts", "--prefer-offline", "-y",
        AGENT_BROWSER_NPX_SPEC, "--version",
    ]


def test_stdin_is_explicitly_devnull_not_inherited():
    """Every subprocess call in tools/ must set stdin= explicitly
    (scripts/check_subprocess_stdin.py) — in the TUI gateway, an inherited
    stdin fd can be consumed by a child and cause the gateway's own
    JSON-RPC stdin read to see a premature EOF (issue #14036). This call
    has no reason to read from stdin at all, so it must be DEVNULL, not
    merely "present in kwargs somewhere" (the checker is a literal-argument
    textual scan, so stdin= folded into a shared kwargs dict wouldn't
    satisfy it either — it must appear as a literal keyword on the call)."""
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), \
         patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        warm_agent_browser_npx_cache()

    _args, kwargs = mock_popen.call_args
    assert kwargs.get("stdin") == subprocess.DEVNULL


def test_captures_stdout_and_stderr_instead_of_inheriting_parent_fds():
    """The npx registry fetch runs on every `hermes update` — its stdout/
    stderr must not bleed into the caller's own output (and, on POSIX, an
    inherited fd is one more handle a runaway grandchild could hold open)."""
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), \
         patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        warm_agent_browser_npx_cache()

    _args, kwargs = mock_popen.call_args
    assert kwargs.get("stdout") == subprocess.PIPE
    assert kwargs.get("stderr") == subprocess.PIPE


def test_uses_credential_scrubbed_environment():
    """Must not inherit the full parent environment — matching every other
    agent-browser subprocess spawn (_build_browser_env), not the ambient
    os.environ with every provider/gateway credential Hermes holds."""
    scrubbed_env = {"PATH": "/scrubbed/bin", "SCRUBBED": "1"}
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), \
         patch("tools.browser_tool._build_browser_env", return_value=dict(scrubbed_env)), \
         patch("tools.browser_tool._merge_browser_path", side_effect=lambda p: p), \
         patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        warm_agent_browser_npx_cache()

    _args, kwargs = mock_popen.call_args
    assert kwargs["env"]["SCRUBBED"] == "1"
    assert "OPENAI_API_KEY" not in kwargs["env"]


def test_merges_extended_path_so_managed_only_npx_can_find_sibling_node():
    """If npx was resolved via the Hermes-managed/extended search (not the
    ambient PATH), the child's own PATH must include that same directory —
    npx's #!/usr/bin/env node shebang resolves `node` via the child's PATH
    at exec time, not the resolving process's PATH."""
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/opt/hermes/node/bin/npx"), \
         patch("tools.browser_tool._build_browser_env", return_value={"PATH": "/usr/bin"}), \
         patch(
             "tools.browser_tool._merge_browser_path",
             return_value="/opt/hermes/node/bin:/usr/bin",
         ) as mock_merge, \
         patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        warm_agent_browser_npx_cache()

    mock_merge.assert_called_once_with("/usr/bin")
    _args, kwargs = mock_popen.call_args
    assert kwargs["env"]["PATH"] == "/opt/hermes/node/bin:/usr/bin"


def test_runs_in_its_own_process_group_on_posix(monkeypatch):
    monkeypatch.setattr("os.name", "posix")
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), \
         patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        warm_agent_browser_npx_cache()

    _args, kwargs = mock_popen.call_args
    assert kwargs.get("start_new_session") is True


def test_uses_new_process_group_creationflag_on_windows_instead_of_start_new_session():
    """start_new_session is a POSIX-only Popen kwarg (raises on Windows).
    The Windows equivalent for _kill_process_tree's taskkill /T to have a
    coherent tree to kill is CREATE_NEW_PROCESS_GROUP via creationflags."""
    with patch("os.name", "nt"), \
         patch("tools.browser_tool._resolve_npx_bin", return_value="C:\\npx.cmd"), \
         patch("tools.browser_tool._build_browser_env", return_value={"PATH": "C:\\Windows"}), \
         patch("tools.browser_tool._merge_browser_path", side_effect=lambda p: p), \
         patch("tools.browser_tool._agent_browser_npx_cache_lock", return_value=nullcontext(True)), \
         patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        warm_agent_browser_npx_cache()

    _args, kwargs = mock_popen.call_args
    assert "start_new_session" not in kwargs
    create_new_pgroup = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    assert kwargs["creationflags"] & create_new_pgroup == create_new_pgroup


def test_timeout_kills_the_whole_process_tree_not_just_the_pid():
    """subprocess.Popen.kill() only signals the direct child; npm/npx can
    fork descendants that survive it and hold a capture pipe open past the
    nominal timeout. On timeout, the whole process group/tree must be
    killed, not just the top-level PID."""
    proc = _mock_proc(
        communicate_side_effect=[
            subprocess.TimeoutExpired(cmd=["npx"], timeout=60.0), ("", ""),
        ]
    )
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), \
         patch("subprocess.Popen", return_value=proc), \
         patch("tools.browser_tool._kill_process_tree") as mock_kill:
        assert warm_agent_browser_npx_cache(timeout=60.0) is False

    mock_kill.assert_called_once_with(proc)
    assert proc.communicate.call_count == 2, (
        "must attempt a second, bounded communicate() after the kill to reap "
        "the now-dead process and drain its pipes, not just abandon it"
    )


def test_timeout_cleanup_communicate_itself_raising_does_not_propagate():
    """The post-kill drain call is itself best-effort — if the process is
    stuck badly enough that even the 5s cleanup communicate() times out (or
    raises for any other reason), that must not escape and crash the
    fire-and-forget caller (hermes_cli/doctor.py calls this bare)."""
    proc = _mock_proc(
        communicate_side_effect=[
            subprocess.TimeoutExpired(cmd=["npx"], timeout=60.0),
            subprocess.TimeoutExpired(cmd=["npx"], timeout=5),
        ]
    )
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), \
         patch("subprocess.Popen", return_value=proc), \
         patch("tools.browser_tool._kill_process_tree") as mock_kill:
        assert warm_agent_browser_npx_cache(timeout=60.0) is False

    mock_kill.assert_called_once_with(proc)


def test_returns_false_on_nonzero_exit():
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), patch(
        "subprocess.Popen", return_value=_mock_proc(returncode=1)
    ):
        assert warm_agent_browser_npx_cache() is False


def test_returns_false_instead_of_raising_on_popen_failure():
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), patch(
        "subprocess.Popen", side_effect=OSError("fork failed")
    ):
        assert warm_agent_browser_npx_cache() is False


def test_returns_false_instead_of_raising_on_unexpected_communicate_exception():
    """Fire-and-forget contract: hermes_cli/doctor.py calls this bare (no
    try/except of its own), so any exception must be swallowed here."""
    proc = _mock_proc(communicate_side_effect=OSError("broken pipe"))
    with patch("tools.browser_tool._resolve_npx_bin", return_value="/usr/bin/npx"), \
         patch("subprocess.Popen", return_value=proc), \
         patch("tools.browser_tool._kill_process_tree") as mock_kill:
        assert warm_agent_browser_npx_cache() is False
    mock_kill.assert_called_once_with(proc)


class TestLegacyKillProcessTree:
    """Contract of the pre-#85125 local fallback (used when agent.deadline
    delegation fails); the delegating wrapper is covered in
    tests/agent/test_treekill_consolidation.py."""

    def test_posix_kills_process_group_term_then_kill(self, monkeypatch):
        import signal

        proc = MagicMock()
        proc.pid = 999
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.setattr("os.getpgid", lambda pid: 999)
        killpg_calls = []
        monkeypatch.setattr(
            "os.killpg", lambda pgid, sig: killpg_calls.append((pgid, sig))
        )

        _legacy_kill_process_tree(proc)

        assert killpg_calls == [(999, signal.SIGTERM), (999, signal.SIGKILL)]

    def test_posix_missing_process_returns_silently(self, monkeypatch):
        proc = MagicMock()
        proc.pid = 999
        monkeypatch.setattr("os.name", "posix")

        def _raise(pid):
            raise ProcessLookupError()

        monkeypatch.setattr("os.getpgid", _raise)

        _legacy_kill_process_tree(proc)  # must not raise

    def test_posix_missing_killpg_attribute_falls_back_to_proc_kill(self, monkeypatch):
        """Some POSIX-like environments may lack os.killpg entirely (the
        implementation resolves it defensively via
        ``getattr(os, "killpg", None)`` — flagged by
        scripts/check-windows-footguns.py against a bare ``os.killpg``
        reference). When that resolution comes back None, the fallback must
        be a plain ``proc.kill()`` of just the top-level PID, not an
        AttributeError."""
        import os as os_module

        proc = MagicMock()
        proc.pid = 999
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.delattr(os_module, "killpg", raising=False)

        _legacy_kill_process_tree(proc)

        proc.kill.assert_called_once()

    def test_posix_missing_killpg_fallback_proc_kill_failure_does_not_raise(self, monkeypatch):
        import os as os_module

        proc = MagicMock()
        proc.pid = 999
        proc.kill.side_effect = OSError("already reaped")
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.delattr(os_module, "killpg", raising=False)

        _legacy_kill_process_tree(proc)  # must not raise

    def test_posix_sigterm_permission_denied_does_not_attempt_sigkill(self, monkeypatch):
        """If SIGTERM itself is rejected (e.g. a stale pgid reused by an
        unrelated, unkillable process), the loop must bail out rather than
        plow ahead into a second signal against the wrong target."""
        import signal

        proc = MagicMock()
        proc.pid = 999
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.setattr("os.getpgid", lambda pid: 999)
        killpg_calls = []

        def fake_killpg(pgid, sig):
            killpg_calls.append((pgid, sig))
            raise PermissionError()

        monkeypatch.setattr("os.killpg", fake_killpg)

        _legacy_kill_process_tree(proc)  # must not raise

        assert killpg_calls == [(999, signal.SIGTERM)]

    def test_windows_uses_taskkill_with_tree_and_force_flags(self, monkeypatch):
        proc = MagicMock()
        proc.pid = 4321
        monkeypatch.setattr("os.name", "nt")
        with patch("subprocess.run") as mock_run:
            _legacy_kill_process_tree(proc)

        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        assert cmd == ["taskkill", "/PID", "4321", "/T", "/F"]

    def test_windows_taskkill_failure_does_not_raise(self, monkeypatch):
        proc = MagicMock()
        proc.pid = 4321
        monkeypatch.setattr("os.name", "nt")
        with patch("subprocess.run", side_effect=OSError("taskkill missing")):
            _legacy_kill_process_tree(proc)  # must not raise
