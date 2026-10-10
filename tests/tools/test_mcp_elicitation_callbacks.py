"""The elicitation consent hop must carry the turn thread's prompt callbacks too.

``ElicitationHandler`` already replays the owning tool call's contextvars snapshot
(``_pending_call_context``) so gateway-platform detection survives the recv-loop task hop.
But the consent call itself runs via ``asyncio.to_thread`` on an executor thread, and the
per-thread prompt callbacks (approval, sudo, vault unlock) live in ``threading.local`` —
they do not follow the contextvars snapshot. Without an explicit bridge the consent flow
sees ``_get_approval_callback() is None`` and a server-initiated elicitation on a stdio
surface fails closed as "the user did not approve" even though a callback was installed
for the whole turn (the GHSA-qg5c-hvr5-hjgr class, on the elicitation path).

These tests exercise the bridge in isolation the way ``test_mcp_elicitation.py`` does:
no real MCP server, the approval router mocked.
"""

import asyncio
from unittest.mock import patch

import pytest


pytest.importorskip("mcp.types")

from tools.mcp_tool_sampling import ElicitationHandler


def _form_params(message="please confirm"):
    from types import SimpleNamespace

    return SimpleNamespace(mode="form", message=message, requested_schema={})


def _capture_thread_callbacks():
    from tools.thread_context import capture_thread_callbacks

    return capture_thread_callbacks()


def _install_approval_callback(cb):
    from tools import terminal_tool

    previous = terminal_tool._get_approval_callback()
    terminal_tool.set_approval_callback(cb)
    return previous


class TestElicitationCallbackBridge:
    def test_turn_thread_approval_callback_reaches_the_consent_call(self):
        """A callback installed on the turn thread must be observable inside the consent
        call that ``asyncio.to_thread`` runs: the handler installs the captured callbacks
        for the duration of the consent invocation and clears them afterwards."""
        from tools import terminal_tool

        def turn_callback(*_args, **_kwargs):
            return "once"

        previous = _install_approval_callback(turn_callback)
        try:
            # What the turn thread's tool wrapper captures (as MCPServerTask will hold it):
            # the contextvars snapshot plus the per-thread prompt callbacks.
            installs = _capture_thread_callbacks()
            handler = ElicitationHandler(
                "pay",
                {"timeout": 5},
                call_callbacks=lambda: installs,
            )
            seen = {}

            def fake_consent(*_args, **_kwargs):
                seen["callback"] = terminal_tool._get_approval_callback()
                return "accept"

            with patch(
                "tools.approval_prompt.request_elicitation_consent",
                side_effect=fake_consent,
            ):
                result = asyncio.run(handler(context=None, params=_form_params()))

            assert result.action == "accept"
            assert seen["callback"] is turn_callback, (
                "The consent call runs on a to_thread executor thread; without the "
                "callback bridge it sees None and the elicitation fails closed."
            )
        finally:
            terminal_tool.set_approval_callback(previous)

    def test_callbacks_are_cleared_after_the_consent_call(self):
        """The to_thread executor reuses threads: a leaked install would outlive the
        elicitation and answer unrelated prompts on that thread. The handler must clear
        what it installed."""
        from tools import terminal_tool

        def turn_callback(*_args, **_kwargs):
            return "once"

        previous = _install_approval_callback(turn_callback)
        try:
            installs = _capture_thread_callbacks()
            handler = ElicitationHandler(
                "pay",
                {"timeout": 5},
                call_callbacks=lambda: installs,
            )

            with patch(
                "tools.approval_prompt.request_elicitation_consent",
                return_value="accept",
            ):
                asyncio.run(handler(context=None, params=_form_params()))

            # The main thread's own install is untouched; the captured copies were
            # cleared on the executor thread. Observable proxy: the thunk's installs
            # were consumed (setter reset to None on the invoking thread).
            assert terminal_tool._get_approval_callback() is turn_callback
        finally:
            terminal_tool.set_approval_callback(previous)

    def test_no_captured_callbacks_still_prompts(self):
        """Between tool calls (or on surfaces that install nothing) the thunk returns an
        empty capture; the consent router must still be invoked."""
        handler = ElicitationHandler("pay", {"timeout": 5}, call_callbacks=lambda: ())
        params = _form_params()

        with patch(
            "tools.approval_prompt.request_elicitation_consent", return_value="accept"
        ) as m:
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "accept"
        assert m.call_count == 1

    def test_context_and_callbacks_replay_together(self):
        """Both bridges compose: the contextvars snapshot and the callbacks are visible
        in the same consent invocation."""
        import contextvars
        from tools import terminal_tool

        probe: contextvars.ContextVar[str] = contextvars.ContextVar(
            "elicitation_cb_probe", default=""
        )

        def turn_callback(*_args, **_kwargs):
            return "once"

        seen = {}

        def fake_consent(*_args, **_kwargs):
            seen["platform"] = probe.get()
            seen["callback"] = terminal_tool._get_approval_callback()
            return "accept"

        # Capture on the (simulated) turn thread: contextvars and callbacks together.
        token = probe.set("gateway:telegram")
        try:
            previous = _install_approval_callback(turn_callback)
            try:
                captured = contextvars.copy_context()
                installs = _capture_thread_callbacks()
            finally:
                terminal_tool.set_approval_callback(previous)
        finally:
            probe.reset(token)

        assert probe.get() == ""  # sanity: empty outside the captured context

        handler = ElicitationHandler(
            "pay",
            {"timeout": 5},
            call_context=lambda: captured,
            call_callbacks=lambda: installs,
        )
        with patch(
            "tools.approval_prompt.request_elicitation_consent",
            side_effect=fake_consent,
        ):
            result = asyncio.run(handler(context=None, params=_form_params()))

        assert result.action == "accept"
        assert seen["platform"] == "gateway:telegram"
        assert seen["callback"] is turn_callback
