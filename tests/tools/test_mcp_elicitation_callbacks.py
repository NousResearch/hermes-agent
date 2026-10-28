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
        what it installed — read the callback back on the very worker thread the consent
        call ran on, by pinning a single-thread default executor that stays alive for the
        whole test (the main-thread-only check cannot observe the executor's cleanup)."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from tools import terminal_tool

        def turn_callback(*_args, **_kwargs):
            return "once"

        loop = asyncio.new_event_loop()
        worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="elicitation-consent"
        )
        try:
            loop.set_default_executor(worker)
            previous = _install_approval_callback(turn_callback)
            try:
                installs = _capture_thread_callbacks()
                assert installs, (
                    "sanity: the capture must see the turn thread's install"
                )
                handler = ElicitationHandler(
                    "pay",
                    {"timeout": 5},
                    call_callbacks=lambda: installs,
                )

                consent_threads = []

                def fake_consent(*_args, **_kwargs):
                    consent_threads.append(threading.get_ident())
                    return "accept"

                with patch(
                    "tools.approval_prompt.request_elicitation_consent",
                    side_effect=fake_consent,
                ):
                    result = loop.run_until_complete(
                        handler(context=None, params=_form_params())
                    )

                assert result.action == "accept"
                assert len(set(consent_threads)) == 1, (
                    "A max_workers=1 executor must run every consent call on one thread"
                )
                assert consent_threads[0] != threading.get_ident(), (
                    "the consent call must run on the executor thread, not the main thread"
                )

                # Same worker thread (the loop's default executor): whatever the handler
                # installed there must be gone again after the consent call returned.
                leftover = loop.run_until_complete(
                    asyncio.to_thread(terminal_tool._get_approval_callback)
                )
                assert leftover is None, (
                    "The install must be cleared on the executor thread itself; a leaked "
                    "install would answer unrelated prompts on that reused thread."
                )
                # The main thread's own install is untouched.
                assert terminal_tool._get_approval_callback() is turn_callback
            finally:
                terminal_tool.set_approval_callback(previous)
        finally:
            worker.shutdown(wait=True)
            loop.close()

    def test_cleanup_survives_consent_errors_and_reuse_sees_only_the_newest(self):
        """The finally-cleanup must hold when the consent call raises (executor threads
        are reused across elicitation requests), and a subsequent elicitation carrying a
        different capture must observe only that newest callback on the same thread."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from tools import terminal_tool

        def first_callback(*_args, **_kwargs):
            return "once"

        def second_callback(*_args, **_kwargs):
            return "once"

        loop = asyncio.new_event_loop()
        worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="elicitation-consent"
        )
        try:
            loop.set_default_executor(worker)
            previous = _install_approval_callback(first_callback)
            try:
                first_installs = _capture_thread_callbacks()
                assert first_installs, "sanity: capture sees the turn thread's install"

                def raising_consent(*_args, **_kwargs):
                    raise RuntimeError("consent surface exploded")

                with patch(
                    "tools.approval_prompt.request_elicitation_consent",
                    side_effect=raising_consent,
                ):
                    result = loop.run_until_complete(
                        ElicitationHandler(
                            "pay",
                            {"timeout": 5},
                            call_callbacks=lambda: first_installs,
                        )(context=None, params=_form_params())
                    )
                # The handler converts the consent failure to a decline — but the
                # worker thread must not keep the install past the failed call.
                assert result.action == "decline"
                leftover = loop.run_until_complete(
                    asyncio.to_thread(terminal_tool._get_approval_callback)
                )
                assert leftover is None, (
                    "the error path must clear the install on the executor thread too"
                )

                # Second elicitation on the same reused worker, carrying a different
                # capture: the consent call must observe exactly that newest callback.
                terminal_tool.set_approval_callback(second_callback)
                second_installs = _capture_thread_callbacks()
                try:
                    seen = {}
                    consent_threads = []

                    def fake_consent(*_args, **_kwargs):
                        consent_threads.append(threading.get_ident())
                        seen["callback"] = terminal_tool._get_approval_callback()
                        return "accept"

                    with patch(
                        "tools.approval_prompt.request_elicitation_consent",
                        side_effect=fake_consent,
                    ):
                        result = loop.run_until_complete(
                            ElicitationHandler(
                                "pay",
                                {"timeout": 5},
                                call_callbacks=lambda: second_installs,
                            )(context=None, params=_form_params())
                        )

                    assert result.action == "accept"
                    assert seen["callback"] is second_callback, (
                        "the reused worker must carry the newest capture, not a stale one"
                    )
                    assert (
                        consent_threads and consent_threads[0] != threading.get_ident()
                    )
                finally:
                    terminal_tool.set_approval_callback(first_callback)
            finally:
                terminal_tool.set_approval_callback(previous)
        finally:
            worker.shutdown(wait=True)
            loop.close()

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
