"""Browser cleanup fallback contracts, retained after retiring the npx warmer."""

import os
import signal
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from tools.browser_tool_lifecycle import _kill_process_tree, _legacy_kill_process_tree


@pytest.mark.platforms("posix")
def test_posix_kills_process_group_term_then_kill(monkeypatch):
    proc = MagicMock(pid=999)
    monkeypatch.setattr(os, "getpgid", lambda pid: 999)
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    _legacy_kill_process_tree(proc)
    assert calls == [(999, signal.SIGTERM), (999, signal.SIGKILL)]


@pytest.mark.platforms("posix")
def test_posix_missing_process_returns_silently(monkeypatch):
    def missing(pid):
        raise ProcessLookupError()

    monkeypatch.setattr(os, "getpgid", missing)
    _legacy_kill_process_tree(MagicMock(pid=999))


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("kill_error", [None, OSError("already reaped")])
def test_missing_killpg_falls_back_to_proc_kill(monkeypatch, kill_error):
    proc = MagicMock(pid=999)
    proc.kill.side_effect = kill_error
    monkeypatch.delattr(os, "killpg", raising=False)
    _legacy_kill_process_tree(proc)
    proc.kill.assert_called_once()


@pytest.mark.platforms("posix")
def test_permission_denied_does_not_attempt_sigkill(monkeypatch):
    monkeypatch.setattr(os, "getpgid", lambda pid: 999)
    calls = []

    def denied(pgid, sig):
        calls.append((pgid, sig))
        raise PermissionError()

    monkeypatch.setattr(os, "killpg", denied)
    _legacy_kill_process_tree(MagicMock(pid=999))
    assert calls == [(999, signal.SIGTERM)]


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("error", [None, OSError("taskkill missing")])
def test_windows_taskkill_targets_tree_and_is_best_effort(error):
    with patch("subprocess.run", side_effect=error) as run:
        _legacy_kill_process_tree(MagicMock(pid=4321))
    assert run.call_args.args[0] == ["taskkill", "/PID", "4321", "/T", "/F"]

# ---------------------------------------------------------------------------
# Group-leadership guard (#119149): a child that does not lead its own group
# shares OURS — killpg would signal the whole Hermes/test-runner tree.
# ---------------------------------------------------------------------------


@pytest.mark.platforms("posix")
def test_posix_shared_group_is_never_killpgd(monkeypatch):
    """A child spawned without start_new_session shares OUR process group, so
    os.getpgid returns the caller's own pgid — killpg would signal the whole
    Hermes/test-runner tree. Only a group leader may be killpg'd; the direct
    child still gets proc.kill()."""
    proc = MagicMock()
    proc.pid = 999
    monkeypatch.setattr(os, "getpgid", lambda pid: 555)  # child's group != its pid
    killpg_calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

    _legacy_kill_process_tree(proc)

    assert killpg_calls == []
    proc.kill.assert_called_once()


@pytest.mark.platforms("posix")
def test_real_shared_group_child_does_not_signal_us(monkeypatch):
    """End-to-end through _kill_process_tree with the deadline helper forced
    to fail: a real child in our own process group must be proc.kill()ed
    without any killpg — if the group signal fired, this test process would
    be dead before the assertion."""
    from tools.browser_tool_lifecycle import _kill_process_tree

    proc = subprocess.Popen(
        ["sleep", "60"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert os.getpgid(proc.pid) == os.getpgid(0)  # shared group precondition
        monkeypatch.setattr(
            "agent.deadline.kill_process_tree",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("deadline unavailable")),
        )
        _kill_process_tree(proc)
        proc.wait(timeout=5)
        assert proc.returncode is not None
    finally:
        if proc.poll() is None:
            proc.kill()


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the grandchild reparents to init once the
# parent dies; signalling a reparented descendant is exactly the contract under
# test, and the guard's subtree walk races with the reparenting (blocked kills
# surfaced as a 1-2% CI flake: psutil's os.kill routes through the guard after
# the parent kill has landed).
def test_real_shared_group_child_descendants_are_killed(tmp_path):
    """A shared-group child's descendants can hold the capture pipe's write
    end open past proc.kill() (the #68915 communicate() hang), so the
    non-leader path must kill them individually. The grandchild is in OUR
    process group: if the implementation regressed to killpg this test
    process would die before the assertion."""
    pid_file = tmp_path / "grandchild.pid"
    proc = subprocess.Popen(
        ["sh", "-c", f"sleep 60 & echo $! > {pid_file}; wait"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    gcpid = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            text = pid_file.read_text().strip() if pid_file.exists() else ""
            if text:
                gcpid = int(text)
                break
            time.sleep(0.05)
        assert gcpid is not None, "grandchild never wrote its pid file"
        assert os.getpgid(proc.pid) == os.getpgid(0)  # shared group precondition

        _legacy_kill_process_tree(proc)
        proc.wait(timeout=5)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(gcpid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"grandchild pid {gcpid} survived the non-leader tree kill")
    finally:
        if proc.poll() is None:
            proc.kill()
        if gcpid is not None:
            try:
                os.kill(gcpid, signal.SIGKILL)
            except OSError:
                pass


@pytest.mark.platforms("posix")
def test_real_group_leader_child_is_tree_killed(monkeypatch):
    """Control: a child leading its own group (process_group=0) still gets the
    group signal through the same fallback path."""
    from tools.browser_tool_lifecycle import _kill_process_tree

    proc = subprocess.Popen(
        ["sleep", "60"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        process_group=0,
    )
    try:
        assert os.getpgid(proc.pid) == proc.pid  # leader precondition
        monkeypatch.setattr(
            "agent.deadline.kill_process_tree",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("deadline unavailable")),
        )
        _kill_process_tree(proc)
        proc.wait(timeout=5)
        assert proc.returncode is not None
    finally:
        if proc.poll() is None:
            proc.kill()
