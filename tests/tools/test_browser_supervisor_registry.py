"""Unit tests for tools.browser_supervisor._SupervisorRegistry.

Covers the registry health-check path without requiring a real Chrome
(the Chrome-backed integration tests live in test_browser_supervisor.py).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

from tools.browser_supervisor import _SupervisorRegistry


class _FakeThread:
    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class _FakeLoop:
    def __init__(self, running: bool) -> None:
        self._running = running

    def is_running(self) -> bool:
        return self._running


def _fake_supervisor(cdp_url: str, *, thread_alive: bool, loop_running: bool) -> SimpleNamespace:
    """Build a supervisor double that satisfies the registry's health check."""
    return SimpleNamespace(
        cdp_url=cdp_url,
        _thread=_FakeThread(thread_alive),
        _loop=_FakeLoop(loop_running),
        start=lambda timeout=15.0: None,
        stop=lambda timeout=5.0: None,
    )


def test_get_or_start_reuses_healthy_cached_supervisor():
    """When cached supervisor matches URL and thread+loop are alive, reuse it."""
    registry = _SupervisorRegistry()
    cached = _fake_supervisor("ws://fake/devtools/page/1", thread_alive=True, loop_running=True)
    registry._by_task["task-a"] = cached  # noqa: SLF001

    with patch("tools.browser_supervisor.CDPSupervisor") as mock_ctor:
        got = registry.get_or_start(task_id="task-a", cdp_url="ws://fake/devtools/page/1")

    assert got is cached
    mock_ctor.assert_not_called()


def test_get_or_start_replaces_when_thread_is_dead():
    """Regression: cached supervisor with same URL but dead thread is replaced.

    Before the health check was added, the registry returned the cached
    supervisor as long as the cdp_url matched — even if its background
    thread had crashed or exited, leaving a zombie object that could not
    service any CDP calls.
    """
    registry = _SupervisorRegistry()
    stopped = threading.Event()

    dead = _fake_supervisor("ws://fake/devtools/page/1", thread_alive=False, loop_running=True)
    dead.stop = lambda timeout=5.0: stopped.set()
    registry._by_task["task-b"] = dead  # noqa: SLF001

    fresh = _fake_supervisor("ws://fake/devtools/page/1", thread_alive=True, loop_running=True)

    with patch("tools.browser_supervisor.CDPSupervisor", return_value=fresh) as mock_ctor:
        got = registry.get_or_start(task_id="task-b", cdp_url="ws://fake/devtools/page/1")

    assert got is fresh
    assert got is not dead
    mock_ctor.assert_called_once()
    assert stopped.is_set(), "dead supervisor should have been stopped"


def test_get_or_start_replaces_when_loop_is_not_running():
    """Same URL + live thread but loop not running → treat as unhealthy."""
    registry = _SupervisorRegistry()
    unhealthy = _fake_supervisor("ws://fake/devtools/page/1", thread_alive=True, loop_running=False)
    registry._by_task["task-c"] = unhealthy  # noqa: SLF001

    fresh = _fake_supervisor("ws://fake/devtools/page/1", thread_alive=True, loop_running=True)

    with patch("tools.browser_supervisor.CDPSupervisor", return_value=fresh) as mock_ctor:
        got = registry.get_or_start(task_id="task-c", cdp_url="ws://fake/devtools/page/1")

    assert got is fresh
    mock_ctor.assert_called_once()


def test_get_or_start_replaces_when_url_differs():
    """Existing behavior preserved: different cdp_url always rebuilds."""
    registry = _SupervisorRegistry()
    stopped = threading.Event()

    old = _fake_supervisor("ws://fake/devtools/page/1", thread_alive=True, loop_running=True)
    old.stop = lambda timeout=5.0: stopped.set()
    registry._by_task["task-d"] = old  # noqa: SLF001

    fresh = _fake_supervisor("ws://fake/devtools/page/2", thread_alive=True, loop_running=True)

    with patch("tools.browser_supervisor.CDPSupervisor", return_value=fresh) as mock_ctor:
        got = registry.get_or_start(task_id="task-d", cdp_url="ws://fake/devtools/page/2")

    assert got is fresh
    mock_ctor.assert_called_once()
    assert stopped.is_set()
