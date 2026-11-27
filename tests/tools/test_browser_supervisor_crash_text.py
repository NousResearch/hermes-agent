"""Crash-text egress for the CDP supervisor (#133216).

Every field occurrence of ``CDP supervisor <task> crashed:`` carried an empty message:
the supervisor died in a message-less ``TimeoutError()`` (``str()`` is ``""``), so the
crash handler's ``%s`` printed nothing and the actual cause stayed masked 26/26 times.
These tests pin the three egress fixes: the redactor never returns an empty string,
a ``_cdp`` timeout names the command that never answered, and the thread-level crash
log carries the exception type instead of a blank.
"""

from __future__ import annotations

import asyncio

import pytest

from tools import browser_supervisor as bs


def test_redact_falls_back_to_type_name_for_messageless_errors():
    # A message-less exception must not redact to "" — that is the #133216 mask.
    assert bs._redact_cdp_error_text(TimeoutError()) == "TimeoutError()"
    assert bs._redact_cdp_error_text(asyncio.CancelledError()) == "CancelledError()"
    # A message-carrying exception keeps its text; the fallback never replaces it.
    assert "connection refused" in bs._redact_cdp_error_text(
        RuntimeError("connection refused")
    )


class _SilentWebSocket:
    """A WebSocket that accepts sends but never answers — the #133216 endpoint shape."""

    async def send(self, _payload):
        pass


def test_cdp_timeout_names_the_command_that_never_answered():
    supervisor = bs.CDPSupervisor(
        task_id="silent-endpoint", cdp_url="ws://127.0.0.1:9222"
    )
    supervisor._ws = _SilentWebSocket()

    async def call():
        return await supervisor._cdp("Target.getTargets", timeout=0.05)

    # Before #133216 this raised a bare TimeoutError() whose str() was "", so neither the
    # crash log nor start()'s re-raise could say WHICH CDP round trip never came back.
    with pytest.raises(TimeoutError) as excinfo:
        asyncio.run(call())
    message = str(excinfo.value)
    assert "Target.getTargets" in message
    assert "timed out" in message
    # The abandoned call id is not left behind in the pending map.
    assert supervisor._pending_calls == {}


def test_thread_main_crash_log_carries_the_type_not_a_blank(caplog):
    supervisor = bs.CDPSupervisor(task_id="crash-blank", cdp_url="ws://127.0.0.1:9222")

    async def ready_then_die():
        supervisor._ready_event.set()  # attach "succeeded"…
        raise TimeoutError()  # …then the loop dies with a message-less error

    supervisor._run = ready_then_die
    with caplog.at_level("WARNING", logger="tools.browser_supervisor"):
        supervisor._thread_main()
    crash_logs = [r for r in caplog.records if "crashed" in r.message]
    assert len(crash_logs) == 1
    # Not "crashed: " — the type name survives, and the entry is redacted like every egress.
    assert "crashed: TimeoutError()" in crash_logs[0].getMessage()
