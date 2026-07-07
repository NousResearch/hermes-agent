"""Test _watch_parent_death, the desktop-backend orphan-exit watchdog.

Desktop-spawned backends (HERMES_DESKTOP=1) must not outlive the Electron
app that spawned them: an orphaned backend keeps a live cron ticker thread
and dead stdout/stderr pipes (BrokenPipeError on any stderr write), which is
how the Jul 6 orphans broke the midnight cron jobs (BUILD-188/BUILD-190).

``_watch_parent_death`` polls ``getppid()`` against the ppid captured at
startup; once it changes (a reparent to launchd/init signals the original
parent died), it fires ``on_parent_death`` exactly once and returns.
"""

import threading

from hermes_cli.web_server import _watch_parent_death


def test_fires_when_parent_dies():
    """getppid changing from the initial value must fire the callback."""
    calls = []
    ppids = iter([100, 100, 1])  # unchanged, unchanged, then reparented to init

    def fake_getppid():
        return next(ppids)

    _watch_parent_death(
        initial_ppid=100,
        on_parent_death=lambda: calls.append(1),
        stop_event=threading.Event(),
        poll_interval=0.01,
        getppid=fake_getppid,
    )

    assert calls == [1]


def test_does_not_fire_when_stopped_first():
    """A stop_event set before any ppid change must suppress the callback
    and the function must return promptly (no hang waiting on a reparent
    that will never come, e.g. graceful app shutdown)."""
    calls = []
    stop_event = threading.Event()
    stop_event.set()

    def fake_getppid():
        # Would signal a reparent if ever consulted, but stop must win first.
        return 999

    _watch_parent_death(
        initial_ppid=100,
        on_parent_death=lambda: calls.append(1),
        stop_event=stop_event,
        poll_interval=0.01,
        getppid=fake_getppid,
    )

    assert calls == []


def test_returns_immediately_when_initial_ppid_is_init():
    """initial_ppid <= 1 means we're already init-parented (or the ppid is
    unknown/0) — nothing meaningful to watch, so return without polling or
    firing the callback."""
    calls = []

    def fake_getppid():
        raise AssertionError("getppid must not be consulted when initial_ppid <= 1")

    _watch_parent_death(
        initial_ppid=1,
        on_parent_death=lambda: calls.append(1),
        stop_event=threading.Event(),
        poll_interval=0.01,
        getppid=fake_getppid,
    )

    assert calls == []


def test_returns_immediately_when_initial_ppid_is_zero():
    """Same guard for a 0 ppid (defensive; some platforms could report it)."""
    calls = []

    def fake_getppid():
        raise AssertionError("getppid must not be consulted when initial_ppid <= 1")

    _watch_parent_death(
        initial_ppid=0,
        on_parent_death=lambda: calls.append(1),
        stop_event=threading.Event(),
        poll_interval=0.01,
        getppid=fake_getppid,
    )

    assert calls == []


def test_fires_at_most_once_even_if_ppid_keeps_changing():
    """Once fired, the function must return — it must not loop forever
    calling on_parent_death again for every subsequent differing ppid."""
    calls = []
    ppids = iter([100, 1, 2, 3])  # changes on first poll, and keeps changing after

    def fake_getppid():
        return next(ppids)

    _watch_parent_death(
        initial_ppid=100,
        on_parent_death=lambda: calls.append(1),
        stop_event=threading.Event(),
        poll_interval=0.01,
        getppid=fake_getppid,
    )

    assert calls == [1]
