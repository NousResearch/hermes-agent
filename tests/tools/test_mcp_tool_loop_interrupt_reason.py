"""Regression tests for rendering the true interrupt cause in the MCP tool loop (#133539).

``_run_on_mcp_loop`` polls ``is_interrupted()`` and used to raise a hardcoded
``InterruptedError("User sent a new message")`` for ANY interrupt — a system abort
(watchdog, lease loss, terminal batch timeout) was misattributed to the user.
``tools.interrupt`` already records a per-tid cause (``set_interrupt(reason=...)``);
these tests pin that ``interrupt_reason()`` surfaces it, that the MCP raise site words
a system stop as an abort while a human stop keeps the legacy wording, and that
``_dispatch`` no longer overwrites the rendered cause on catch.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
from unittest.mock import patch

import pytest

from tools import mcp_tool_loop, mcp_tool_handlers
from tools.interrupt import interrupt_reason, set_interrupt


@pytest.fixture
def interrupted_thread():
    """Set an interrupt with a cause on the CURRENT thread; always clear it after."""
    tid = threading.current_thread().ident

    def _set(reason):
        set_interrupt(True, tid, reason=reason)

    yield _set
    set_interrupt(False, tid)


def test_interrupt_reason_reads_the_recorded_cause(interrupted_thread):
    interrupted_thread("terminal batch timeout")
    assert interrupt_reason() == "terminal batch timeout"


def test_interrupt_reason_is_none_when_no_cause_was_recorded(interrupted_thread):
    interrupted_thread(None)
    assert interrupt_reason() is None


def test_stop_notice_words_a_system_abort_from_the_recorded_reason(interrupted_thread):
    interrupted_thread("lease lost")
    assert mcp_tool_loop._interrupt_stop_notice() == "Turn aborted — lease lost"


@pytest.mark.parametrize(
    "reason",
    [None, "user sent a new message", "user interrupt", "explicit stop requested"],
)
def test_stop_notice_keeps_the_legacy_wording_for_human_stops(
    interrupted_thread, reason
):
    interrupted_thread(reason)
    assert mcp_tool_loop._interrupt_stop_notice() == "User sent a new message"


def _run_with_pending_future():
    """Drive ``_run_on_mcp_loop`` against a fake running loop whose scheduled future
    never resolves, so the poll loop reaches the interrupt check on the first pass."""

    class _FakeLoop:
        @staticmethod
        def is_running():
            return True

    async def _never():
        # Never awaited to completion: the interrupt fires before the first result poll.
        await concurrent.futures.Future()

    def _schedule_pending(coro, loop, **kwargs):
        coro.close()  # nobody awaits the wrapped coroutine: consume it here
        return concurrent.futures.Future()

    with (
        patch.object(mcp_tool_loop, "_running_loop", return_value=_FakeLoop()),
        patch(
            "agent.async_utils.safe_schedule_threadsafe", side_effect=_schedule_pending
        ),
    ):
        return mcp_tool_loop._run_on_mcp_loop(_never)


def test_run_on_mcp_loop_raises_the_system_abort_notice(interrupted_thread):
    interrupted_thread("terminal batch timeout")
    with pytest.raises(InterruptedError) as excinfo:
        _run_with_pending_future()
    assert str(excinfo.value) == "Turn aborted — terminal batch timeout"


def test_run_on_mcp_loop_keeps_user_wording_without_a_recorded_cause(
    interrupted_thread,
):
    interrupted_thread(None)
    with pytest.raises(InterruptedError) as excinfo:
        _run_with_pending_future()
    assert str(excinfo.value) == "User sent a new message"


def test_dispatch_reports_the_true_interrupt_cause_not_a_hardcoded_user_stop():
    def _call():
        raise AssertionError(
            "the coroutine must not run: the loop raises before result polling"
        )

    with patch.object(
        mcp_tool_loop,
        "_run_on_mcp_loop",
        side_effect=InterruptedError("Turn aborted — lease lost"),
    ):
        result = mcp_tool_handlers._dispatch(
            "srv",
            None,
            "tools/call",
            _call,
            1.0,
            recoverers=[],
            on_final_failure=lambda exc: pytest.fail(
                "interrupts are not final failures"
            ),
        )
    assert (
        json.loads(result)["error"] == "MCP call interrupted: Turn aborted — lease lost"
    )


def test_dispatch_falls_back_to_user_wording_for_a_messageless_interrupt():
    def _call():
        raise AssertionError(
            "the coroutine must not run: the loop raises before result polling"
        )

    with patch.object(
        mcp_tool_loop, "_run_on_mcp_loop", side_effect=InterruptedError()
    ):
        result = mcp_tool_handlers._dispatch(
            "srv",
            None,
            "tools/call",
            _call,
            1.0,
            recoverers=[],
            on_final_failure=lambda exc: pytest.fail(
                "interrupts are not final failures"
            ),
        )
    assert (
        json.loads(result)["error"] == "MCP call interrupted: user sent a new message"
    )
