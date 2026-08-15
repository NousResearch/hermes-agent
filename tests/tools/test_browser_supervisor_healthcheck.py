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


def _make_fake_supervisor(
    cdp_url: str,
    *,
    target_id: str | None = None,
    thread_alive: bool,
    loop_running: bool,
):
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
        target_id=target_id,
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
        def __init__(
            self,
            *,
            task_id,
            cdp_url,
            target_id,
            dialog_policy,
            dialog_timeout_s,
        ):
            self.task_id = task_id
            self.cdp_url = cdp_url
            self.target_id = target_id
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


def test_missing_thread_and_loop_attrs_trigger_recreate(
    isolated_registry, stub_cdp_supervisor
):
    """Defensive: None _thread or None _loop counts as unhealthy."""
    cdp_url = "http://h/4"
    broken = SimpleNamespace(
        cdp_url=cdp_url,
        target_id=None,
        _thread=None,
        _loop=None,
        stop=lambda: None,
    )
    isolated_registry._by_task["t4"] = broken

    fresh = isolated_registry.get_or_start(task_id="t4", cdp_url=cdp_url)
    assert fresh is not broken
    assert isolated_registry._by_task["t4"] is fresh
    fresh.stop()


def test_concurrent_different_targets_leave_one_live_supervisor(
    isolated_registry, monkeypatch
):
    """A publication race must stop the displaced target supervisor."""
    start_barrier = threading.Barrier(2)
    created = []

    class _RacingSupervisor:
        def __init__(
            self,
            *,
            task_id,
            cdp_url,
            target_id,
            dialog_policy,
            dialog_timeout_s,
        ):
            self.task_id = task_id
            self.cdp_url = cdp_url
            self.target_id = target_id
            self.dialog_policy = dialog_policy
            self.dialog_timeout_s = dialog_timeout_s
            self.live = False
            self.stop_calls = 0
            created.append(self)

        def start(self, timeout=15.0):
            self.live = True
            start_barrier.wait(timeout=2)

        def stop(self):
            self.stop_calls += 1
            self.live = False

    monkeypatch.setattr(bs, "CDPSupervisor", _RacingSupervisor)
    results = []
    errors = []

    def run(target_id):
        try:
            results.append(
                isolated_registry.get_or_start(
                    task_id="race-task",
                    cdp_url="ws://shared",
                    target_id=target_id,
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=("TARGET-A",)),
        threading.Thread(target=run, args=("TARGET-B",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 2
    assert len(created) == 2
    managed = isolated_registry.get("race-task")
    assert managed in created
    assert managed.live is True
    assert sum(supervisor.live for supervisor in created) == 1
    assert sum(supervisor.stop_calls for supervisor in created) == 1
    isolated_registry.stop("race-task")
