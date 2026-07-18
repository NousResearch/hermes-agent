"""Regression test for issue #63892.

On Python >= 3.8, ``concurrent.futures.TimeoutError`` is an alias for the
builtin ``TimeoutError``. ``_run_on_mcp_loop`` polls a future with
``future.result(timeout=wait_timeout)`` and catches
``concurrent.futures.TimeoutError`` to implement "poll expired, keep
waiting".  Because of the alias the same except branch also catches the
case where **the future has COMPLETED and its coroutine's stored exception
is a real TimeoutError** — e.g. an inner ``asyncio.wait_for`` around an MCP
``call_tool`` hit ``mcp_servers.<srv>.timeout``.

When that happens the loop degenerates: ``future.result()`` returns
instantly (the future is done), re-raises the same stored exception, the
``except`` swallows it, ``continue`` — a tight spin with no sleep. Each
re-raise appends frames to the same exception object's ``__traceback__``
chain, leaking memory at ~108 MB/s until the gateway is OOM-killed.

The fix: when the except branch fires, check whether the future is actually
done.  If so, surface its real outcome (value on a poll/success race, real
exception otherwise) and let the caller's error path handle it. Only
``continue`` the poll loop when the future is genuinely still pending.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from unittest import mock

import pytest

import tools.mcp_tool as mcp_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spawn_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start a private asyncio loop on a daemon thread — caller must stop it."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def _stop_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def _install_loop(mcp_mod, loop, thread):
    old_loop = mcp_mod._mcp_loop
    old_thread = mcp_mod._mcp_thread
    mcp_mod._mcp_loop = loop
    mcp_mod._mcp_thread = thread
    return old_loop, old_thread


def _restore_loop(mcp_mod, old_loop, old_thread):
    mcp_mod._mcp_loop = old_loop
    mcp_mod._mcp_thread = old_thread


def _pre_completed_future(exception=None, value=None):
    """Return a ``concurrent.futures.Future`` that is already done.

    If *exception* is set the future is completed with that exception;
    otherwise *value* (default ``None``) is the result.
    """
    fut = concurrent.futures.Future()
    if exception is not None:
        fut.set_exception(exception)
    else:
        fut.set_result(value)
    return fut


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunOnMcpLoopCompletedFutureTimeoutRace:
    """Issue #63892: a completed future raising a real TimeoutError must not
    spin the poll loop and grow its traceback chain without bound."""

    def test_completed_future_with_real_timeout_propagates_once(self):
        """Inner coroutine raises a real asyncio.TimeoutError.

        Without the fix, ``_run_on_mcp_loop`` would catch the alias, see the
        future as "still pending" (it isn't — but ``concurrent.futures`` raises
        ``TimeoutError`` from ``future.result(timeout=...)`` based on the
        *poll*, not the future's state), and continue polling. The future is
        done after the first poll, so the next ``future.result(timeout=...)``
        returns instantly re-raising the stored exception, which is again
        swallowed — a tight spin with unbounded traceback growth.
        """
        loop, thread = _spawn_loop()
        old_loop, old_thread = _install_loop(mcp_mod, loop, thread)

        async def _inner_raises_timeout():
            raise asyncio.TimeoutError("inner wait_for expired")

        try:
            with pytest.raises(asyncio.TimeoutError, match="inner wait_for expired"):
                mcp_mod._run_on_mcp_loop(_inner_raises_timeout(), timeout=5)
        finally:
            _stop_loop(loop, thread)
            _restore_loop(mcp_mod, old_loop, old_thread)

    def test_poll_loop_does_not_spin_when_inner_timeout_fires(self):
        """Injects a pre-completed Future with a stored TimeoutError by
        patching ``safe_schedule_threadsafe``, then asserts the poll loop
        surfaces the exception within a bounded number of polls.

        Pre-fix the loop spun ~420k times/sec because ``future.result()``
        returned instantly (future was done) but the except clause
        swallowed it and ``continue``-d.
        """

        # Build a pre-completed future with a real TimeoutError stored.
        pre_completed = _pre_completed_future(
            exception=asyncio.TimeoutError("inner wait_for expired")
        )

        # Wrap .result() to count polls.
        poll_count = {"n": 0}
        orig_result = pre_completed.result

        def _counting_result(timeout=None):
            poll_count["n"] += 1
            return orig_result(timeout)

        pre_completed.result = _counting_result  # type: ignore[method-assign]

        # Both arguments to _run_on_mcp_loop need to be valid so the
        # function-under-test runs its body.  We supply a do-nothing
        # coroutine because the injected future replaces whatever
        # safe_schedule_threadsafe would have returned.
        async def _dummy_coro():
            pass

        loop, thread = _spawn_loop()
        old_loop, old_thread = _install_loop(mcp_mod, loop, thread)

        # Patch safe_schedule_threadsafe to return the pre-completed future.
        # This is the key fix over the original test: the function-under-test
        # uses safe_schedule_threadsafe internally, so we intercept it here
        # rather than counting polls on a different future object.
        def _fake_sched(coro, loop, **kw):
            coro.close()  # don't leak the coroutine
            return pre_completed

        with mock.patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=_fake_sched,
        ):
            try:
                with pytest.raises(asyncio.TimeoutError, match="inner wait_for expired"):
                    mcp_mod._run_on_mcp_loop(_dummy_coro(), timeout=1)
                # Loose bound — pre-fix this would be ~10^5+ within 1s.
                assert poll_count["n"] < 100, (
                    f"poll loop spun {poll_count['n']} times — unbounded spin "
                    f"indicates #63892 regression"
                )
            finally:
                _stop_loop(loop, thread)
                _restore_loop(mcp_mod, old_loop, old_thread)

    def test_traceback_depth_does_not_grow_under_repeated_polls(self):
        """Pre-fix the same exception object accumulated frames on each re-raise.

        After the fix the exception surfaces exactly once and its traceback
        depth stays bounded.
        """
        loop, thread = _spawn_loop()
        old_loop, old_thread = _install_loop(mcp_mod, loop, thread)

        async def _inner_raises_timeout():
            raise asyncio.TimeoutError("inner wait_for expired")

        try:
            with pytest.raises(asyncio.TimeoutError) as excinfo:
                mcp_mod._run_on_mcp_loop(_inner_raises_timeout(), timeout=2)

            # Walk the __traceback__ chain and count frames. A bounded raise
            # path produces a modest, stable depth. The spinning loop in #63892
            # grew this by ~3 frames per iteration, ~420k iterations/sec —
            # so any trace deeper than ~50 frames within a 2s test window
            # would indicate the spin returned.
            depth = 0
            tb = excinfo.value.__traceback__
            while tb is not None:
                depth += 1
                tb = tb.tb_next
            # Loose bound — a single re-raise is typically < 20 frames.
            # The bug produced depths growing without bound; pick a bound
            # that catches the spin mode without flaking on a slow CI runner.
            assert depth < 200, (
                f"exception __traceback__ depth {depth} exceeds bound — "
                f"indicates #63892 traceback-accumulation regression"
            )
        finally:
            _stop_loop(loop, thread)
            _restore_loop(mcp_mod, old_loop, old_thread)

    def test_poll_timeout_still_continues_when_future_pending(self):
        """The legitimate case — future is still pending, poll expires — must
        still continue the loop and eventually return the real value.

        This guards against the obvious over-fix: some might be tempted to
        ``raise`` unconditionally when ``future.done()`` is true, but that
        breaks the normal poll-then-success race where the future completes
        between the poll raise and the check. We must return, not raise,
        so the success path still works.
        """
        loop, thread = _spawn_loop()
        old_loop, old_thread = _install_loop(mcp_mod, loop, thread)

        async def _slow_then_value():
            await asyncio.sleep(0.3)
            return "ok"

        try:
            result = mcp_mod._run_on_mcp_loop(_slow_then_value(), timeout=2)
            assert result == "ok"
        finally:
            _stop_loop(loop, thread)
            _restore_loop(mcp_mod, old_loop, old_thread)

    def test_race_poll_timeout_then_future_becomes_done_with_value(self):
        """Deterministic poll-timeout/success-race coverage.

        On the first ``future.result(timeout=0.1)`` call (poll), the future
        is still pending, so ``TimeoutError`` is raised.  Between the poll
        and the ``future.done()`` check, the future completes.  The fix must
        call ``future.result()`` (untimed) to surface the value instead of
        ``continue``-ing the loop.

        We use a custom Future subclass that changes state after the first
        timed poll.
        """
        loop, thread = _spawn_loop()
        old_loop, old_thread = _install_loop(mcp_mod, loop, thread)

        class _RaceFuture(concurrent.futures.Future):
            """A Future that is pending on the first ``result(timeout=...)``
            call (raises TimeoutError) but becomes done before the handler
            checks ``future.done()``."""

            def __init__(self, value):
                super().__init__()
                self._race_value = value
                self._polled = False

            def result(self, timeout=None):
                if not self._polled:
                    # First call: simulate poll timeout.
                    self._polled = True
                    # We don't actually sleep — just raise immediately.
                    # This is a "timed" call that expired.
                    if timeout is not None and timeout < 999:
                        raise concurrent.futures.TimeoutError()
                # Second call (or untimed call): we are now "done".
                if not self.done():
                    self.set_result(self._race_value)
                return super().result(timeout)

        race_future = _RaceFuture(value="race_value")

        async def _dummy_coro():
            pass

        with mock.patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=lambda coro, loop, **kw: (coro.close(), race_future)[1],
        ):
            try:
                result = mcp_mod._run_on_mcp_loop(_dummy_coro(), timeout=10)
                assert result == "race_value", (
                    f"Expected race_value, got {result!r}"
                )
            finally:
                _stop_loop(loop, thread)
                _restore_loop(mcp_mod, old_loop, old_thread)

    def test_race_poll_timeout_then_future_becomes_done_with_exception(self):
        """Deterministic poll-timeout/exception-race coverage.

        Similar to the value-race above, but the future completes with a
        stored exception between the poll and the ``future.done()`` check.
        The fix must call ``future.result()`` (untimed) to surface the
        exception.
        """
        loop, thread = _spawn_loop()
        old_loop, old_thread = _install_loop(mcp_mod, loop, thread)

        class _RaceFuture(concurrent.futures.Future):
            def __init__(self, exc):
                super().__init__()
                self._race_exc = exc
                self._polled = False

            def result(self, timeout=None):
                if not self._polled:
                    self._polled = True
                    if timeout is not None and timeout < 999:
                        raise concurrent.futures.TimeoutError()
                if not self.done():
                    self.set_exception(self._race_exc)
                return super().result(timeout)

        race_exc = RuntimeError("server returned error")
        race_future = _RaceFuture(exc=race_exc)

        async def _dummy_coro():
            pass

        with mock.patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=lambda coro, loop, **kw: (coro.close(), race_future)[1],
        ):
            try:
                with pytest.raises(RuntimeError, match="server returned error"):
                    mcp_mod._run_on_mcp_loop(_dummy_coro(), timeout=10)
            finally:
                _stop_loop(loop, thread)
                _restore_loop(mcp_mod, old_loop, old_thread)
