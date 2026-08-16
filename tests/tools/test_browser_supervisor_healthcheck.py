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


def test_start_failure_is_stopped_and_never_published(
    isolated_registry, monkeypatch
):
    """A missing pinned target must fail visibly without leaking a starter."""
    created = []

    class _FailingSupervisor:
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
            self.stop_calls = 0
            created.append(self)

        def start(self, timeout=15.0):
            raise RuntimeError("requested pinned target is absent")

        def stop(self):
            self.stop_calls += 1

    monkeypatch.setattr(bs, "CDPSupervisor", _FailingSupervisor)

    with pytest.raises(RuntimeError, match="requested pinned target is absent"):
        isolated_registry.get_or_start(
            "missing-target",
            "ws://shared",
            target_id="GONE",
        )

    assert isolated_registry.get("missing-target") is None
    assert "missing-target" not in isolated_registry._starting
    assert len(created) == 1
    assert created[0].stop_calls == 1


def test_stop_fences_inflight_starter_publication(isolated_registry, monkeypatch):
    """A starter that completes after stop must be stopped, never published."""
    started = threading.Event()
    release = threading.Event()
    stopped: list[str] = []

    class _BlockingSupervisor:
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

        def start(self, timeout=15.0):
            started.set()
            assert release.wait(timeout=3)

        def stop(self):
            stopped.append(self.task_id)

    monkeypatch.setattr(bs, "CDPSupervisor", _BlockingSupervisor)
    results = []
    thread = threading.Thread(
        target=lambda: results.append(
            isolated_registry.get_or_start(
                "stop-race",
                "ws://shared",
                target_id="TARGET",
            )
        )
    )
    thread.start()
    assert started.wait(timeout=3)

    isolated_registry.stop("stop-race")
    release.set()
    thread.join(timeout=4)

    assert not thread.is_alive()
    assert len(results) == 1
    assert isolated_registry.get("stop-race") is None
    assert stopped


def test_stop_cannot_pass_between_displacement_and_starter_registration(
    isolated_registry,
    monkeypatch,
):
    """Replacing an old supervisor must register before stopping the old one."""
    old_stop_entered = threading.Event()
    release_old_stop = threading.Event()

    def _stop_old():
        old_stop_entered.set()
        assert release_old_stop.wait(timeout=3)

    old = SimpleNamespace(
        cdp_url="ws://old",
        target_id="OLD",
        _thread=None,
        _loop=None,
        stop=_stop_old,
    )
    isolated_registry._by_task["replacement-race"] = old

    created = []

    class _ReplacementSupervisor:
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
            self.live = False
            self.stop_calls = 0
            created.append(self)

        def start(self, timeout=15.0):
            self.live = True

        def stop(self):
            self.stop_calls += 1
            self.live = False

    monkeypatch.setattr(bs, "CDPSupervisor", _ReplacementSupervisor)
    results = []
    thread = threading.Thread(
        target=lambda: results.append(
            isolated_registry.get_or_start(
                "replacement-race",
                "ws://new",
                target_id="NEW",
            )
        )
    )
    thread.start()
    assert old_stop_entered.wait(timeout=3)

    # stop() must see the already-registered replacement starter even though
    # the creator is still blocked while disposing the displaced instance.
    isolated_registry.stop("replacement-race")
    release_old_stop.set()
    thread.join(timeout=4)

    assert not thread.is_alive()
    assert len(results) == 1
    assert len(created) == 1
    assert isolated_registry.get("replacement-race") is None
    assert created[0].live is False
    assert created[0].stop_calls >= 1
