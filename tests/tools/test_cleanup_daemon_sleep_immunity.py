"""The cleanup daemons must not be turnable into a full-speed spin.

Both ``tools.terminal_tool`` and ``tools.browser_tool`` run a background
cleanup daemon that idles between sweeps. The daemon outlives whatever started
it, so *any* later code — in practice a test doing
``patch("<module>.time.sleep")``, which rebinds the attribute on the shared
``time`` module and therefore unblocks every sleeper in the process — used to
be able to turn that idle wait into a busy loop. Each turn recorded a ``_Call``
on the MagicMock; measured in the test suite, about two minutes of that was
13.7 GB of RSS and a hard OOM that took the whole run down.

These tests pin the property that makes the class impossible: the daemons idle
on ``threading.Event.wait``, which a patched ``time.sleep`` cannot touch, and
stopping them is immediate instead of costing up to one idle interval.
"""

import contextlib
import time
from unittest.mock import MagicMock, patch

import tools.browser_tool as browser_tool
import tools.terminal_tool as terminal_tool

#: Captured before any test can patch it, so the observation window below is a
#: genuine wait even while ``time.sleep`` is mocked.
_REAL_SLEEP = time.sleep

#: How long to let a daemon idle while ``time.sleep`` is a no-op. A spinning
#: loop reaches millions of calls in this window; a correct one reaches zero.
_OBSERVE_SECONDS = 0.4

#: Generous ceiling: a daemon may legitimately touch a patched sleep a handful
#: of times through code it calls, but never thousands.
_SANE_CALL_CEILING = 100


def _calls_while_idling(start, stop, sweep_targets):
    """Run a cleanup daemon with ``time.sleep`` mocked; return the mock."""
    fake_sleep = MagicMock(name="time.sleep")
    with contextlib.ExitStack() as stack:
        for target in sweep_targets:
            stack.enter_context(patch(target))
        stack.enter_context(patch("time.sleep", fake_sleep))
        start()
        try:
            _REAL_SLEEP(_OBSERVE_SECONDS)
        finally:
            stop()
    return fake_sleep


def test_terminal_cleanup_daemon_ignores_a_globally_patched_sleep():
    fake_sleep = _calls_while_idling(
        terminal_tool._start_cleanup_thread,
        terminal_tool._stop_cleanup_thread,
        ["tools.terminal_tool._cleanup_inactive_envs"],
    )

    assert fake_sleep.call_count <= _SANE_CALL_CEILING, (
        "the terminal cleanup daemon spun on a patched time.sleep "
        f"({fake_sleep.call_count} calls in {_OBSERVE_SECONDS}s) — it must idle "
        "on threading.Event.wait, which patching time.sleep cannot disable"
    )


def test_browser_cleanup_daemon_ignores_a_globally_patched_sleep():
    fake_sleep = _calls_while_idling(
        browser_tool._start_browser_cleanup_thread,
        browser_tool._stop_browser_cleanup_thread,
        [
            "tools.browser_tool._cleanup_inactive_browser_sessions",
            "tools.browser_tool._reap_orphaned_browser_sessions",
        ],
    )

    assert fake_sleep.call_count <= _SANE_CALL_CEILING, (
        "the browser cleanup daemon spun on a patched time.sleep "
        f"({fake_sleep.call_count} calls in {_OBSERVE_SECONDS}s)"
    )


def test_stopping_the_terminal_daemon_does_not_wait_out_the_idle_interval():
    """Event-based idling also means shutdown is immediate, not up to a cycle."""
    assert terminal_tool._CLEANUP_INTERVAL_SECONDS >= 30, (
        "this test is only meaningful while the idle interval is long"
    )
    with patch("tools.terminal_tool._cleanup_inactive_envs"):
        terminal_tool._start_cleanup_thread()
        thread = terminal_tool._cleanup_thread
        started = time.monotonic()
        terminal_tool._stop_cleanup_thread()
        elapsed = time.monotonic() - started

    assert not thread.is_alive()
    assert elapsed < 5, f"stopping the daemon took {elapsed:.1f}s"
