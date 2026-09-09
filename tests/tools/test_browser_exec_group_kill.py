"""Regression tests for browser_exec process-group kill on timeout (#106244).

``browser_exec`` wedges when the browser-use CLI's ``browser_harness`` daemon
grandchild keeps the stdout/stderr pipes open after the CLI child is killed on
timeout: ``subprocess.run``'s internal ``communicate()`` blocks forever on the
grandchild's inherited pipe fds, so the ``TimeoutExpired`` handler never runs and
the worker thread (and its activity heartbeat) wedge forever.

The fix starts the CLI in its own session/process-group (``start_new_session=True``)
and kills the whole group/tree on timeout using the existing cross-platform helpers
(``_kill_process_group_posix`` / ``_kill_process_windows``) so the grandchild dies
and the post-kill ``communicate()`` returns. These tests verify that mechanism.

#106244 — "Wedged browser_exec calls leak tool-activity heartbeats that keep old
desktop sessions pinned at 'now' in sidebar."
"""
import contextlib
import json
import os
import signal
import subprocess
import sys
import textwrap

import pytest

from tools.environments.local import _IS_WINDOWS, _kill_process_group_posix


def _grandchild_holds_pipe_script() -> str:
    """Child that forks a grandchild holding the inherited stdout pipe, then both
    stay alive — mirroring the CLI child (stuck waiting for the harness daemon) and
    the ``browser_harness`` daemon grandchild (keeping the pipes open)."""
    return textwrap.dedent(
        """
        import os, time
        pid = os.fork()
        if pid == 0:
            time.sleep(60)  # grandchild: hold the inherited stdout pipe open
            os._exit(0)
        # child: stay alive too (mirrors the CLI child waiting on the daemon),
        # also holding the pipe so a group kill is needed to reach EOF.
        time.sleep(60)
        os._exit(0)
        """
    )


@pytest.mark.skipif(_IS_WINDOWS, reason="fork-based grandchild simulation is POSIX-only")
def test_group_kill_closes_grandchild_pipes_so_communicate_returns():
    """The fix's mechanism: killing the whole process group closes the
    grandchild-held pipe so a subsequent ``communicate()`` returns instead of
    blocking forever."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _grandchild_holds_pipe_script()],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        # Confirms the wedge: the grandchild keeps stdout open past the child's
        # lifetime, so communicate() cannot reach EOF within the timeout.
        with pytest.raises(subprocess.TimeoutExpired):
            proc.communicate(timeout=2)
        # The child is still alive (stuck waiting) so its pgid is resolvable — the
        # exact state browser_exec's timeout branch is in.
        # The fix: kill the whole process group so the grandchild dies too.
        _kill_process_group_posix(proc)
        # Post-fix: the grandchild is dead, the pipe sees EOF, so communicate()
        # returns promptly instead of blocking forever.
        out, err = proc.communicate(timeout=5)
        assert proc.returncode is not None
    finally:
        with contextlib.suppress(Exception):
            _kill_process_group_posix(proc)


@pytest.mark.skipif(_IS_WINDOWS, reason="fork-based grandchild simulation is POSIX-only")
def test_run_style_kill_cannot_recover_from_grandchild_pipe():
    """Documents the bug the fix replaces: ``subprocess.run(timeout=)`` kills only
    the CLI child, leaving the grandchild holding the pipes, so its internal
    ``communicate()`` blocks forever (the ``TimeoutExpired`` the caller expects never
    surfaces). Pins the failure mode the fix addresses — run-style capture cannot
    recover from a grandchild-held pipe."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _grandchild_holds_pipe_script()],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,  # isolated group so the grandchild is reaped below
    )
    pgid = os.getpgid(proc.pid)  # cache before the child is killed
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            proc.communicate(timeout=2)
        # The run-style kill (SIGKILL the child only) leaves the grandchild alive
        # holding the pipe; communicate() STILL cannot reach EOF.
        proc.kill()
        with pytest.raises(subprocess.TimeoutExpired):
            proc.communicate(timeout=2)
    finally:
        # Reap the orphaned grandchild via the cached group id (the child is dead,
        # so os.getpgid(proc.pid) would raise — the group id is the stable handle).
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.wait(timeout=2)


def test_browser_exec_kills_process_group_on_timeout(monkeypatch):
    """``browser_exec`` must kill the whole process group when the CLI times out, not
    just the CLI child. Verifies the fix wires the cross-platform helper into the
    timeout branch."""
    import tools.browser_use_cli as mod
    import tools.environments.local as local

    kill_calls: list = []

    class _FakeProc:
        returncode = -9
        pid = 12345

        def communicate(self, input=None, timeout=None):
            if kill_calls:
                # After the group-kill call, communicate returns (grandchild dead).
                return ("", "")
            raise subprocess.TimeoutExpired(cmd=["echo"], timeout=timeout)

    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(local, "_IS_WINDOWS", False)
    monkeypatch.setattr(local, "_kill_process_group_posix", lambda proc: kill_calls.append(proc.pid))
    monkeypatch.setattr(local, "_kill_process_windows", lambda proc: kill_calls.append(("win", proc.pid)))
    # Stub the pre-subprocess setup so browser_exec reaches the Popen call.
    monkeypatch.setattr(mod, "_find_cli", lambda: ["echo", "ok"])
    monkeypatch.setattr(mod, "_base_subprocess_env", lambda: {})
    monkeypatch.setattr(mod, "_route_backend", lambda env, session, task_id, local: None)
    monkeypatch.setattr(mod, "_workspace_dir", lambda task_id: None)
    monkeypatch.setattr(mod, "is_legacy_browser_use_cloud_config", lambda cfg: False)
    monkeypatch.setattr(mod, "_read_browser_cfg", lambda: {})

    result = mod.browser_exec(code="# step\n", timeout_s=1)
    assert kill_calls, "process-group kill helper was not invoked on timeout"
    # tool_error returns a JSON string: {"error": "..."}
    parsed = json.loads(result)
    assert "error" in parsed
    # On POSIX (_IS_WINDOWS=False), the POSIX helper must be the one invoked.
    assert kill_calls == [12345]
