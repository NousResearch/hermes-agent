"""Unit tests for _SupervisorRegistry cache-hit healthcheck.

Verifies that get_or_start() does NOT return a cached supervisor whose
thread has exited or whose event loop has stopped. Avoids a real Chrome —
the only thing under test is the registry's cache decision.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tools import browser_supervisor as bs


class _FakeLoop:
    def __init__(self, running: bool) -> None:
        self._running = running

    def is_running(self) -> bool:
        return self._running


def _make_fake_supervisor(cdp_url: str, *, thread_alive: bool, loop_running: bool):
    """Build a minimal stand-in for a CDPSupervisor entry in the registry.

    Only the attributes touched by the healthcheck (_thread, _loop, cdp_url)
    and by the teardown path (stop()) need to exist.
    """

    if thread_alive:
        # A thread that is actually running — parks on an Event we never set.
        hold = threading.Event()
        t = threading.Thread(target=hold.wait, daemon=True)
        t.start()
        # Attach the release hook so the test can let the thread exit.
        setattr(t, "_release", hold.set)
    else:
        # An un-started thread — is_alive() returns False.
        t = threading.Thread(target=lambda: None)

    stop_calls: list[bool] = []

    fake = SimpleNamespace(
        cdp_url=cdp_url,
        _thread=t,
        _loop=_FakeLoop(loop_running),
        stop=lambda: stop_calls.append(True),
    )
    fake._stop_calls = stop_calls  # type: ignore[attr-defined]
    return fake


@pytest.fixture
def isolated_registry():
    """A fresh registry instance, independent of the global SUPERVISOR_REGISTRY."""
    return bs._SupervisorRegistry()


@pytest.fixture
def stub_cdp_supervisor(monkeypatch):
    """Replace CDPSupervisor in the module so recreate paths don't touch Chrome.

    Returns a callable that reads the last-constructed fake out.
    """
    created: list[SimpleNamespace] = []

    class _StubSupervisor:
        def __init__(self, *, task_id, cdp_url, dialog_policy, dialog_timeout_s):
            self.task_id = task_id
            self.cdp_url = cdp_url
            self.dialog_policy = dialog_policy
            self.dialog_timeout_s = dialog_timeout_s
            # Healthy by default — real thread, running "loop".
            hold = threading.Event()
            self._thread = threading.Thread(target=hold.wait, daemon=True)
            self._thread.start()
            self._thread_release = hold.set  # type: ignore[attr-defined]
            self._loop = _FakeLoop(True)
            self.start_called = False
            self.stop_called = False
            created.append(self)

        def start(self, timeout: float = 15.0) -> None:
            self.start_called = True

        def stop(self) -> None:
            self.stop_called = True
            # Release the parked thread so the process exits cleanly.
            release = getattr(self, "_thread_release", None)
            if release is not None:
                release()

    monkeypatch.setattr(bs, "CDPSupervisor", _StubSupervisor)
    yield created
    # Teardown: release any parked threads in stubs the test left behind.
    for s in created:
        release = getattr(s, "_thread_release", None)
        if release is not None:
            release()


def test_cache_hit_returns_same_instance_when_healthy(
    isolated_registry, stub_cdp_supervisor
):
    """Sanity: healthy cached supervisor is returned without recreate."""
    first = isolated_registry.get_or_start(task_id="t1", cdp_url="http://h/1")
    second = isolated_registry.get_or_start(task_id="t1", cdp_url="http://h/1")
    assert first is second
    # Only one CDPSupervisor was ever constructed.
    assert len(stub_cdp_supervisor) == 1
    first.stop()


def test_dead_thread_triggers_recreate(isolated_registry, stub_cdp_supervisor):
    """Cached supervisor with a non-live thread must not be reused."""
    cdp_url = "http://h/2"
    dead = _make_fake_supervisor(cdp_url, thread_alive=False, loop_running=True)
    isolated_registry._by_task["t2"] = dead  # pre-seed cache with a dead entry

    fresh = isolated_registry.get_or_start(task_id="t2", cdp_url=cdp_url)

    assert fresh is not dead, "dead-thread supervisor must be replaced"
    assert dead._stop_calls == [True], "dead supervisor must be torn down"
    assert isolated_registry._by_task["t2"] is fresh
    assert len(stub_cdp_supervisor) == 1
    assert stub_cdp_supervisor[0].start_called
    fresh.stop()


def test_stopped_loop_triggers_recreate(isolated_registry, stub_cdp_supervisor):
    """Cached supervisor whose event loop is no longer running is recreated."""
    cdp_url = "http://h/3"
    broken = _make_fake_supervisor(cdp_url, thread_alive=True, loop_running=False)
    isolated_registry._by_task["t3"] = broken

    fresh = isolated_registry.get_or_start(task_id="t3", cdp_url=cdp_url)

    assert fresh is not broken
    assert broken._stop_calls == [True]
    # Release the still-live thread from the pre-seeded fake so we don't leak.
    release = getattr(broken._thread, "_release", None)
    if release is not None:
        release()
    assert isolated_registry._by_task["t3"] is fresh
    fresh.stop()


def test_missing_thread_and_loop_attrs_trigger_recreate(
    isolated_registry, stub_cdp_supervisor
):
    """Defensive: None _thread or None _loop counts as unhealthy."""
    cdp_url = "http://h/4"
    broken = SimpleNamespace(
        cdp_url=cdp_url,
        _thread=None,
        _loop=None,
        stop=lambda: None,
    )
    isolated_registry._by_task["t4"] = broken

    fresh = isolated_registry.get_or_start(task_id="t4", cdp_url=cdp_url)
    assert fresh is not broken
    assert isolated_registry._by_task["t4"] is fresh
    fresh.stop()
