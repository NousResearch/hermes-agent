"""Behavioral tests for the host-global Camofox turn lease."""

import os
import subprocess
import sys
import threading
import time

import pytest

from tools import browser_camofox as camofox


@pytest.fixture(autouse=True)
def release_leases_after_test():
    camofox._release_all_turn_leases()
    yield
    camofox._release_all_turn_leases()


def test_queued_owner_acquires_only_after_release(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox._release_all_turn_leases()
    camofox.acquire_turn_lease("first")
    acquired = threading.Event()
    def queued():
        camofox.acquire_turn_lease("queued", timeout=2)
        acquired.set()
    waiter = threading.Thread(target=queued)
    waiter.start()
    time.sleep(0.1)
    assert not acquired.is_set()
    camofox.release_turn_lease("first")
    waiter.join(3)
    assert acquired.is_set()
    camofox.release_turn_lease("queued")


def test_subprocess_queued_owner_acquires_after_release(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox._release_all_turn_leases()
    camofox.acquire_turn_lease("parent")
    code = "from tools.browser_camofox import acquire_turn_lease; acquire_turn_lease('child', timeout=3)"
    proc = subprocess.Popen([sys.executable, "-c", code], env=os.environ.copy())
    time.sleep(0.2)
    assert proc.poll() is None
    camofox.release_turn_lease("parent")
    assert proc.wait(timeout=4) == 0


@pytest.mark.platforms("posix")
def test_forked_same_owner_waits_and_parent_keeps_exclusion(monkeypatch, tmp_path):
    import fcntl
    import select

    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox.acquire_turn_lease("same-task")
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            try:
                camofox.acquire_turn_lease("same-task", timeout=0.25)
            except TimeoutError:
                os.write(write_fd, b"timeout")
            else:
                os.write(write_fd, b"reentered")
        finally:
            os._exit(0)
    os.close(write_fd)
    ready, _, _ = select.select([read_fd], [], [], 3)
    result = os.read(read_fd, 32) if ready else b"no-result"
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert result == b"timeout"
    with open(camofox._turn_lease_path(), "a+b") as contender:
        with pytest.raises(OSError):
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    camofox.release_turn_lease("same-task")


def test_timeout_and_interrupt_never_acquire(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox._release_all_turn_leases()
    camofox.acquire_turn_lease("held")
    with pytest.raises(TimeoutError):
        camofox.acquire_turn_lease("timed", timeout=0.05)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: True)
    with pytest.raises(InterruptedError):
        camofox.acquire_turn_lease("cancelled", timeout=2)
    camofox.release_turn_lease("held")
    assert "timed" not in camofox._turn_lease_handles
    assert "cancelled" not in camofox._turn_lease_handles


def test_deadline_is_rechecked_after_local_waiter_wakes(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox.acquire_turn_lease("holder")
    clock = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    def expire_and_release(_delay):
        clock[0] = 11.0
        camofox.release_turn_lease("holder")
    monkeypatch.setattr(time, "sleep", expire_and_release)
    with pytest.raises(TimeoutError):
        camofox.acquire_turn_lease("late", timeout=0.5)
    assert "late" not in camofox._turn_lease_handles


def test_same_owner_reentry_honors_interrupt(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox.acquire_turn_lease("owner")
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: True)
    with pytest.raises(InterruptedError):
        camofox.acquire_turn_lease("owner")
    camofox.release_turn_lease("owner")


def test_lease_budget_uses_actual_tool_executor_deadlines(monkeypatch):
    seen = []
    def resolve(key, default):
        seen.append(key)
        return {"tools.sequential_call": 40, "tools.concurrent_batch": 25}[key]
    monkeypatch.setattr("agent.deadline.resolve_timeout", resolve)
    assert camofox._lease_wait_budget() == 20
    assert seen == ["tools.sequential_call", "tools.concurrent_batch"]


def test_lease_budget_honors_disabled_deadlines(monkeypatch):
    monkeypatch.setattr("agent.deadline.resolve_timeout", lambda key, default: None)
    assert camofox._lease_wait_budget() == 300


def test_same_owner_reenters_and_different_task_waits(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox._release_all_turn_leases()
    camofox.acquire_turn_lease("owner-a")
    camofox.acquire_turn_lease("owner-a")
    acquired = threading.Event()
    waiter = threading.Thread(target=lambda: (camofox.acquire_turn_lease("owner-b", timeout=2), acquired.set()))
    waiter.start()
    time.sleep(0.1)
    assert not acquired.is_set()
    camofox.release_turn_lease("owner-a")
    waiter.join(3)
    assert acquired.is_set()
    camofox.release_turn_lease("owner-b")


def test_same_owner_network_requests_are_serialized(monkeypatch, tmp_path):
    import threading
    import time

    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    active = 0
    peak = 0
    state_lock = threading.Lock()

    class Response:
        def raise_for_status(self):
            return None

    def request(*_args, **_kwargs):
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        entered.set()
        release.wait(2)
        with state_lock:
            active -= 1
        return Response()

    monkeypatch.setattr(camofox.requests, "get", request)
    camofox.acquire_turn_lease("same-owner")
    first = threading.Thread(target=camofox._request, args=("get", "/one"))
    second = threading.Thread(target=camofox._request, args=("get", "/two"))
    first.start()
    assert entered.wait(1)
    second.start()
    time.sleep(0.05)
    release.set()
    first.join(2)
    second.join(2)
    assert peak == 1


def test_public_navigate_serializes_session_tab_creation(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    session = {"user_id": "shared", "tab_id": None, "session_key": "task", "managed": False}
    entered = threading.Event()
    release = threading.Event()
    calls = []
    active = 0
    peak = 0

    monkeypatch.setattr(camofox, "_get_session", lambda _owner: session)

    def create_tab(path, *_args, **_kwargs):
        nonlocal active, peak
        calls.append(path)
        active += 1
        peak = max(peak, active)
        entered.set()
        release.wait(2)
        active -= 1
        if path == "/tabs":
            session["tab_id"] = "tab-created"
            return {"tabId": "tab-created", "url": "https://example.com"}
        return {"url": "https://example.com"}

    monkeypatch.setattr(camofox, "_post", create_tab)
    monkeypatch.setattr(camofox, "get_vnc_url", lambda: None)
    monkeypatch.setattr(camofox, "_fetch_snapshot", lambda _session: ("", 0))
    first = threading.Thread(target=camofox.camofox_navigate, args=("https://example.com", "same-owner"))
    second = threading.Thread(target=camofox.camofox_navigate, args=("https://example.com", "same-owner"))
    first.start()
    assert entered.wait(1)
    second.start()
    time.sleep(0.05)
    release.set()
    first.join(2)
    second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert peak == 1
    assert calls.count("/tabs") == 1


def test_soft_cleanup_waits_for_same_owner_public_operation(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    session = {"user_id": "shared", "tab_id": "tab", "managed": True}
    camofox._sessions["same-owner"] = session

    def hold_public_action(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return '{"success":true}'

    monkeypatch.setattr(camofox, "_with_tab", hold_public_action)
    monkeypatch.setattr(camofox, "_managed_persistence_enabled", lambda _config: True)
    monkeypatch.setattr(camofox, "_camofox_identity_override", lambda *_args: False)
    operation = threading.Thread(target=camofox.camofox_snapshot, kwargs={"task_id": "same-owner"})
    operation.start()
    assert entered.wait(1)

    cleanup_done = threading.Event()
    cleanup = threading.Thread(
        target=lambda: (camofox.camofox_soft_cleanup("same-owner"), cleanup_done.set())
    )
    cleanup.start()
    time.sleep(0.05)
    assert not cleanup_done.is_set()
    assert camofox._sessions.get("same-owner") is session

    release.set()
    operation.join(2)
    cleanup.join(2)
    assert not operation.is_alive() and not cleanup.is_alive()
    assert cleanup_done.is_set()
    assert "same-owner" not in camofox._sessions


def test_queued_public_action_does_not_block_current_lease_owner(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox._release_all_turn_leases()
    owner_done_action = threading.Event()
    queued_started = threading.Event()
    calls = []
    monkeypatch.setattr(camofox, "_get_session", lambda owner: calls.append(owner) or {"user_id": owner, "tab_id": "tab"})
    monkeypatch.setattr(camofox, "_tab_action", lambda owner, *_a, **_k: calls.append(owner) or '{"success":true}')
    camofox.acquire_turn_lease("owner-a")
    def queued_action():
        queued_started.set()
        camofox.camofox_scroll("down", "owner-b")
    waiter = threading.Thread(target=queued_action)
    waiter.start()
    assert queued_started.wait(1)
    time.sleep(0.1)
    camofox.camofox_scroll("down", "owner-a")
    owner_done_action.set()
    assert calls == ["owner-a"]
    camofox.release_turn_lease("owner-a")
    waiter.join(2)
    assert not waiter.is_alive()
    assert calls[-1] == "owner-b"
    assert owner_done_action.is_set()


def test_release_failure_always_clears_local_owner(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox.acquire_turn_lease("unlock-fails")
    original_flock = camofox._flock
    monkeypatch.setattr(camofox, "_flock", lambda _handle, acquire: (_ for _ in ()).throw(OSError("unlock")) if not acquire else None)
    camofox.release_turn_lease("unlock-fails")
    camofox.acquire_turn_lease("subsequent-owner")
    monkeypatch.setattr(camofox, "_flock", original_flock)
    camofox.release_turn_lease("subsequent-owner")


def test_manual_camofox_close_keeps_turn_lease_until_cleanup(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox.acquire_turn_lease("active-turn")
    monkeypatch.setattr(camofox, "_drop_session", lambda _task_id: None)
    assert '"closed": true' in camofox.camofox_close("active-turn")
    with pytest.raises(TimeoutError):
        camofox.acquire_turn_lease("other-turn", timeout=0.05)
    camofox.release_turn_lease("active-turn")
    camofox.acquire_turn_lease("other-turn")
    camofox.release_turn_lease("other-turn")


def test_separate_process_waits_for_host_global_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox._release_all_turn_leases()
    camofox.acquire_turn_lease("profile-a-task")
    code = "from tools.browser_camofox import acquire_turn_lease; acquire_turn_lease('profile-b-task', timeout=3)"
    proc = subprocess.Popen([sys.executable, "-c", code], env=os.environ.copy())
    time.sleep(0.2)
    assert proc.poll() is None
    camofox.release_turn_lease("profile-a-task")
    assert proc.wait(timeout=4) == 0


def test_cleanup_browser_releases_even_when_cleanup_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    from tools import browser_tool_lifecycle as lifecycle
    camofox.acquire_turn_lease("cleanup-task")
    monkeypatch.setattr(lifecycle._bt, "_is_camofox_mode", lambda: True)
    monkeypatch.setattr(lifecycle, "_cleanup_single_browser_session", lambda _key: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        lifecycle.cleanup_browser("cleanup-task")
    camofox.acquire_turn_lease("another-task")
    camofox.release_turn_lease("another-task")


def test_normal_cleanup_releases_lease(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    from tools import browser_tool_lifecycle as lifecycle
    camofox.acquire_turn_lease("normal-task")
    monkeypatch.setattr(lifecycle._bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(lifecycle, "_cleanup_single_browser_session", lambda _key: None)
    lifecycle.cleanup_browser("normal-task")
    camofox.acquire_turn_lease("next-task")
    camofox.release_turn_lease("next-task")


def test_force_reap_releases_without_session_info(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    from tools import browser_tool_lifecycle as lifecycle
    camofox.acquire_turn_lease("reap-task")
    monkeypatch.setattr(lifecycle._bt, "_is_camofox_mode", lambda: True)
    monkeypatch.setattr(lifecycle._cdp, "_stop_cdp_supervisor", lambda _key: None)
    lifecycle._force_reap_browser_session("reap-task")
    camofox.acquire_turn_lease("new-task")
    camofox.release_turn_lease("new-task")


def test_process_death_releases_os_lock(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path))
    camofox._release_all_turn_leases()
    code = "from tools.browser_camofox import acquire_turn_lease; acquire_turn_lease('dead-owner')"
    proc = subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=False)
    assert proc.returncode == 0
    camofox.acquire_turn_lease("new-owner")
    camofox.release_turn_lease("new-owner")
