#!/usr/bin/env python3
"""Tests for tools/clarify_bridge — gateway-side wiring for the clarify tool.

Mirrors the pattern in tests/tools/test_approval.py: the gateway registers a
notify callback, the agent thread blocks on a threading.Event, an inbound user
message unblocks it by calling resolve_gateway_clarify, and session cleanup
signals all pending waits so nothing hangs.
"""

import json
import threading
import time
import unittest
from unittest.mock import MagicMock

from tools.clarify_bridge import (
    ClarifyTimeout,
    ClarifyUnavailable,
    clear_session_clarifies,
    gateway_clarify_callback,
    get_clarify_timeout,
    has_blocking_clarify,
    register_gateway_clarify_notify,
    resolve_gateway_clarify,
    set_clarify_timeout,
    unregister_gateway_clarify_notify,
)
from tools.clarify_tool import clarify_tool


_SESSION = "test-session-1"
_OTHER_SESSION = "test-session-2"


class ClarifyBridgeTests(unittest.TestCase):
    """Happy-path + edge cases for the gateway clarify bridge."""

    def setUp(self):
        # Short timeout so timeout-path tests don't slow the suite down.
        self._orig_timeout = get_clarify_timeout()
        set_clarify_timeout(2.0)

    def tearDown(self):
        # Always clean up so one test's leftover state can't leak into another.
        clear_session_clarifies(_SESSION)
        clear_session_clarifies(_OTHER_SESSION)
        unregister_gateway_clarify_notify(_SESSION)
        unregister_gateway_clarify_notify(_OTHER_SESSION)
        set_clarify_timeout(self._orig_timeout)

    # -------------------------------------------------------------------
    # Happy path
    # -------------------------------------------------------------------

    def test_happy_path_open_ended(self):
        """Agent thread blocks, user reply resolves it."""
        notify = MagicMock()
        register_gateway_clarify_notify(_SESSION, notify)

        callback = gateway_clarify_callback(_SESSION)

        results = {}

        def agent_thread():
            results["answer"] = callback("What is your name?", None)

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()

        # Wait for the agent thread to register the blocking entry.
        deadline = time.monotonic() + 1.0
        while not has_blocking_clarify(_SESSION) and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertTrue(has_blocking_clarify(_SESSION))
        notify.assert_called_once_with("What is your name?", None)

        # User replies (from the asyncio loop in real code).
        resolved = resolve_gateway_clarify(_SESSION, "sx")
        self.assertEqual(resolved, 1)

        t.join(timeout=2.0)
        self.assertFalse(t.is_alive(), "agent thread should have unblocked")
        self.assertEqual(results["answer"], "sx")
        self.assertFalse(has_blocking_clarify(_SESSION))

    def test_happy_path_multiple_choice(self):
        """Choices are forwarded to the notify callback and the answer returned."""
        notify = MagicMock()
        register_gateway_clarify_notify(_SESSION, notify)
        callback = gateway_clarify_callback(_SESSION)

        results = {}

        def agent_thread():
            results["answer"] = callback(
                "Pick one", ["option A", "option B", "option C"]
            )

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()

        deadline = time.monotonic() + 1.0
        while not has_blocking_clarify(_SESSION) and time.monotonic() < deadline:
            time.sleep(0.01)

        notify.assert_called_once()
        args = notify.call_args
        self.assertEqual(args.args[0], "Pick one")
        self.assertEqual(args.args[1], ["option A", "option B", "option C"])

        resolve_gateway_clarify(_SESSION, "option B")
        t.join(timeout=2.0)
        self.assertEqual(results["answer"], "option B")

    # -------------------------------------------------------------------
    # Unregistered / unavailable
    # -------------------------------------------------------------------

    def test_callback_without_notify_raises_unavailable(self):
        """Calling the callback before notify is registered must fail fast."""
        callback = gateway_clarify_callback(_SESSION)
        with self.assertRaises(ClarifyUnavailable):
            callback("Question?", None)

    def test_clarify_tool_returns_error_json_when_unavailable(self):
        """Integration: clarify_tool wraps ClarifyUnavailable into error JSON."""
        callback = gateway_clarify_callback(_SESSION)  # no notify registered
        result_json = clarify_tool("Question?", callback=callback)
        result = json.loads(result_json)
        self.assertIn("error", result)
        self.assertIn("Failed to get user input", result["error"])

    # -------------------------------------------------------------------
    # Timeout
    # -------------------------------------------------------------------

    def test_timeout_path(self):
        """User never answers — callback raises ClarifyTimeout and clears state."""
        set_clarify_timeout(0.2)
        notify = MagicMock()
        register_gateway_clarify_notify(_SESSION, notify)
        callback = gateway_clarify_callback(_SESSION)

        with self.assertRaises(ClarifyTimeout):
            callback("slow question", None)

        # Queue must be drained so a late user message doesn't bind to a dead entry.
        self.assertFalse(has_blocking_clarify(_SESSION))

    # -------------------------------------------------------------------
    # Session clear / unregister
    # -------------------------------------------------------------------

    def test_clear_session_unblocks_waiters(self):
        """/new (or equivalent session reset) must wake blocked agent threads."""
        notify = MagicMock()
        register_gateway_clarify_notify(_SESSION, notify)
        callback = gateway_clarify_callback(_SESSION)

        outcome = {}

        def agent_thread():
            try:
                outcome["answer"] = callback("Q?", None)
            except ClarifyUnavailable as exc:
                outcome["error"] = str(exc)

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()

        deadline = time.monotonic() + 1.0
        while not has_blocking_clarify(_SESSION) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(has_blocking_clarify(_SESSION))

        clear_session_clarifies(_SESSION)
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())
        self.assertIn("error", outcome)
        self.assertIn("cancelled", outcome["error"].lower())

    def test_unregister_notify_also_unblocks_waiters(self):
        """unregister (turn cleanup) must signal all pending entries."""
        notify = MagicMock()
        register_gateway_clarify_notify(_SESSION, notify)
        callback = gateway_clarify_callback(_SESSION)

        outcome = {}

        def agent_thread():
            try:
                outcome["answer"] = callback("Q?", None)
            except ClarifyUnavailable:
                outcome["cancelled"] = True

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()

        deadline = time.monotonic() + 1.0
        while not has_blocking_clarify(_SESSION) and time.monotonic() < deadline:
            time.sleep(0.01)

        unregister_gateway_clarify_notify(_SESSION)
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())
        self.assertTrue(outcome.get("cancelled"))

    # -------------------------------------------------------------------
    # Session isolation
    # -------------------------------------------------------------------

    def test_resolve_other_session_does_not_unblock(self):
        """A user message on session B must not resolve session A's clarify."""
        notify_a = MagicMock()
        register_gateway_clarify_notify(_SESSION, notify_a)
        callback_a = gateway_clarify_callback(_SESSION)

        outcome = {}

        def agent_thread_a():
            try:
                outcome["answer"] = callback_a("A?", None)
            except Exception as exc:
                outcome["err"] = type(exc).__name__

        t = threading.Thread(target=agent_thread_a, daemon=True)
        t.start()

        deadline = time.monotonic() + 1.0
        while not has_blocking_clarify(_SESSION) and time.monotonic() < deadline:
            time.sleep(0.01)

        # Resolve the *other* session — must be a no-op for _SESSION.
        resolved = resolve_gateway_clarify(_OTHER_SESSION, "wrong session")
        self.assertEqual(resolved, 0)
        self.assertTrue(has_blocking_clarify(_SESSION))

        # Resolve the right session — unblocks cleanly.
        resolve_gateway_clarify(_SESSION, "correct answer")
        t.join(timeout=2.0)
        self.assertEqual(outcome.get("answer"), "correct answer")

    # -------------------------------------------------------------------
    # Notify failure
    # -------------------------------------------------------------------

    def test_notify_exception_does_not_leak_blocking_entry(self):
        """If notify raises, the entry must be removed so has_blocking_clarify is accurate."""
        def failing_notify(q, c):
            raise RuntimeError("send failed")

        register_gateway_clarify_notify(_SESSION, failing_notify)
        callback = gateway_clarify_callback(_SESSION)

        with self.assertRaises(RuntimeError):
            callback("Q?", None)

        self.assertFalse(has_blocking_clarify(_SESSION))

    # -------------------------------------------------------------------
    # resolve with no pending entries
    # -------------------------------------------------------------------

    def test_resolve_with_empty_queue_returns_zero(self):
        """No pending clarify — resolve_gateway_clarify returns 0 and is safe."""
        resolved = resolve_gateway_clarify(_SESSION, "answer")
        self.assertEqual(resolved, 0)

    def test_has_blocking_empty_session_key(self):
        """Empty / None session keys must not break has_blocking_clarify."""
        self.assertFalse(has_blocking_clarify(""))
        self.assertFalse(has_blocking_clarify(None))  # type: ignore

    def test_set_clarify_timeout_validates_positive(self):
        with self.assertRaises(ValueError):
            set_clarify_timeout(0)
        with self.assertRaises(ValueError):
            set_clarify_timeout(-1)


if __name__ == "__main__":
    unittest.main()
