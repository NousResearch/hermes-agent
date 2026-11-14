"""The orphan-browser reap must not depend on a browser request (#145010).

``reap_orphaned_browsers`` was only driven by the browser janitor thread, and the janitor was only
started from the browser-session path. A gateway killed uncleanly therefore left its Chromium tree
to a successor that reaped it only if some later browser request happened to create the thread.
"""

from unittest.mock import MagicMock

from gateway.run_startup import GatewayStartupMixin
from tools import browser_session
from tools import browser_session_lifecycle as lifecycle


class _Runner(GatewayStartupMixin):
    """Enough of a gateway to reach the watcher-spawning seam without a live one."""

    def __init__(self) -> None:
        self._PRE_RECONNECT_WATCHERS = ()
        self._POST_RECONNECT_WATCHERS = ()
        self._failed_platforms = []
        self._spawn_supervised = MagicMock()
        self._spawn_reconnect_watcher = MagicMock()
        self._scale_to_zero_should_arm = lambda: False
        self._log_scale_to_zero_not_armed_reason = MagicMock()
        self._drain_control_watcher = MagicMock()


def test_gateway_start_starts_the_browser_janitor(monkeypatch):
    started = []
    monkeypatch.setattr(lifecycle, "start_cleanup_thread", lambda: started.append(True))

    _Runner()._start_spawn_background_watchers()

    assert started == [True]


def test_janitor_failure_does_not_abort_watcher_spawn(monkeypatch):
    def boom():
        raise RuntimeError("browser modules unavailable")

    monkeypatch.setattr(lifecycle, "start_cleanup_thread", boom)
    runner = _Runner()

    runner._start_spawn_background_watchers()

    spawned = [c.args[1] for c in runner._spawn_supervised.call_args_list]
    assert "drain_control_watcher" in spawned


def test_janitor_reaps_on_its_first_cycle(monkeypatch):
    """The first cycle reaps; the 300 s interval only gates the repeats."""
    calls = []

    def reap():
        calls.append("reap")
        browser_session._cleanup_running = False
        return 0

    monkeypatch.setattr(lifecycle, "reap_orphaned_browser_sessions", reap)
    monkeypatch.setattr(lifecycle, "cleanup_inactive_sessions", lambda: calls.append("idle"))
    monkeypatch.setattr(browser_session, "_cleanup_running", True)

    lifecycle._janitor()

    assert calls[0] == "reap"
