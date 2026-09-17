"""Unit tests for the post-attach reconnect give-up cap in ``CDPSupervisor._run()``.

Regression tests for #114172: after one successful attach, a dead CDP endpoint
(a finished cron session's short-lived Chrome) used to be retried forever —
~6 warnings/min for the life of the host process. The fix bounds reconnecting
to ``RECONNECT_GIVE_UP_AFTER_S`` since the last successful attach, then gives
up (single warning, thread exits; the registry healthcheck rebuilds on demand).

No real browser: ``websockets.connect`` is faked to raise (or hand back a stub
connection) and ``_run()`` is driven directly on a throwaway event loop.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import List

import pytest

from tools import browser_supervisor as bs


class _ConnectFailed(Exception):
    """Stand-in for e.g. ``[Errno 61] Connect call failed``."""


class _FakeWS:
    async def close(self) -> None:  # ``_close_ws`` awaits this on teardown
        pass


def _patch_connect(monkeypatch, script) -> List[int]:
    """``websockets.connect`` driven by ``script``: each entry is a connect attempt
    number; attempts not listed raise. Returns the list of attempts made."""
    calls: List[int] = []

    async def fake_connect(url, **kwargs):
        attempt = len(calls) + 1
        calls.append(attempt)
        if attempt in script:
            return _FakeWS()
        raise _ConnectFailed(f"Connect call failed ('127.0.0.1', 51277)")

    import websockets
    monkeypatch.setattr(websockets, "connect", fake_connect)
    return calls


def _make_supervisor(*, attached_recently_at: float) -> bs.CDPSupervisor:
    sup = bs.CDPSupervisor("task-reconnect-cap", "ws://127.0.0.1:51277/devtools/page/x")
    # Simulate a supervisor that attached once (ready set, timestamp recorded).
    sup._ready_event.set()
    sup._last_attach_ok_at = attached_recently_at
    return sup


class TestReconnectGiveUpCap:
    def test_gives_up_once_window_expired(self, monkeypatch, caplog):
        """Endpoint dead longer than the window → ``_run`` returns instead of
        retrying forever, logging the give-up exactly once."""
        monkeypatch.setattr(bs, "RECONNECT_GIVE_UP_AFTER_S", 30.0)
        sup = _make_supervisor(attached_recently_at=time.time() - 31.0)
        calls = _patch_connect(monkeypatch, script=set())

        with caplog.at_level(logging.WARNING, logger="tools.browser_supervisor"):
            asyncio.run(asyncio.wait_for(sup._run(), timeout=5))  # TimeoutError ⇒ still looping

        assert len(calls) == 1  # gave up on the first failure, no retry storm
        assert sup._active is False
        gives_up = [r for r in caplog.records if "giving up reconnecting" in r.message]
        assert len(gives_up) == 1

    def test_keeps_retrying_within_window(self, monkeypatch):
        """Failures inside the window still retry (backoff 0.5s → ≥2 attempts in <1s)."""
        monkeypatch.setattr(bs, "RECONNECT_GIVE_UP_AFTER_S", 30.0)
        sup = _make_supervisor(attached_recently_at=time.time())
        calls = _patch_connect(monkeypatch, script=set())

        async def scenario():
            task = asyncio.create_task(sup._run())
            await asyncio.sleep(0.7)
            assert not task.done()  # cap must not fire while the window is open
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        assert len(calls) >= 2

    def test_successful_attach_refreshes_window(self, monkeypatch):
        """A reconnect that succeeds restarts the clock: an attach older than the
        window must not cause a give-up right after a fresh successful attach."""
        monkeypatch.setattr(bs, "RECONNECT_GIVE_UP_AFTER_S", 30.0)
        sup = _make_supervisor(attached_recently_at=time.time() - 3600.0)
        # Connect succeeds on attempt 1 (stale timestamp refreshed), the session
        # then drops; attempts 2+ fail but stay within the fresh window.
        calls = _patch_connect(monkeypatch, script={1})

        async def noop_attach():
            return None

        async def drop_immediately():
            raise RuntimeError("session dropped")

        monkeypatch.setattr(sup, "_attach_initial_page", noop_attach)
        monkeypatch.setattr(sup, "_read_loop", drop_immediately)

        async def scenario():
            task = asyncio.create_task(sup._run())
            await asyncio.sleep(0.7)
            assert not task.done()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        # 1 successful attach + ≥1 failed reconnect that must NOT trip the cap:
        # the fresh ``_last_attach_ok_at`` replaces the 3600s-stale one.
        assert len(calls) >= 2
        assert sup._last_attach_ok_at is not None and sup._last_attach_ok_at > time.time() - 60

    def test_pre_first_attach_failure_stays_fatal(self, monkeypatch):
        """The give-up cap only applies after a successful attach: a supervisor
        that never got ready must keep failing fast into ``start()``."""
        monkeypatch.setattr(bs, "RECONNECT_GIVE_UP_AFTER_S", 0.0)  # even a zero window
        sup = bs.CDPSupervisor("task-never-ready", "ws://127.0.0.1:1/devtools/page/x")
        assert not sup._ready_event.is_set()
        calls = _patch_connect(monkeypatch, script=set())

        asyncio.run(asyncio.wait_for(sup._run(), timeout=5))

        assert len(calls) == 1
        assert isinstance(sup._start_error, _ConnectFailed)
        assert sup._last_attach_ok_at is None
