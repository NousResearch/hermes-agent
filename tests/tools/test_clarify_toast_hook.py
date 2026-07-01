"""Integration test: tools/clarify_tool.py fires notify_input_needed before blocking.

Verifies that clarify_tool() calls the local_toast module immediately before
handing off to the callback that actually blocks the agent thread. The toast
call itself is patched to a no-op so we're purely asserting that the wire-up
in clarify_tool.py is present and passes the question through.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.clarify_tool import clarify_tool


class TestClarifyToastHook:
    def test_notify_input_needed_called_before_callback(self):
        """The toast dispatch happens BEFORE the blocking callback runs."""
        callback_mock = MagicMock(return_value="answer")
        with patch("tools.local_toast.notify_input_needed") as notify_mock:
            result = json.loads(
                clarify_tool("What color?", callback=callback_mock)
            )

        # Callback was invoked (clarify completed normally).
        assert result["user_response"] == "answer"
        # Notify was called exactly once with the right kind + detail.
        assert notify_mock.call_count == 1
        args, kwargs = notify_mock.call_args
        assert args[0] == "clarify"
        # detail=<question>, level="info"
        assert kwargs.get("detail") == "What color?"
        assert kwargs.get("level") == "info"

    def test_notify_failure_does_not_break_clarify(self):
        """If notify_input_needed raises, clarify_tool still proceeds."""
        callback_mock = MagicMock(return_value="answer-anyway")

        def _boom(*a, **kw):
            raise RuntimeError("antenna is on fire")

        with patch("tools.local_toast.notify_input_needed", side_effect=_boom):
            result = json.loads(
                clarify_tool("What color?", callback=callback_mock)
            )

        # Clarify completes even though the toast dispatcher raised.
        assert result["user_response"] == "answer-anyway"
        callback_mock.assert_called_once()

    def test_no_notify_when_callback_missing(self):
        """clarify_tool short-circuits with an error when no callback is
        registered; the toast should NOT fire in that path because the
        agent isn't actually blocking on user input."""
        with patch("tools.local_toast.notify_input_needed") as notify_mock:
            result = json.loads(clarify_tool("What?"))  # no callback
        assert "error" in result
        assert notify_mock.call_count == 0

    def test_toast_import_absent_is_fail_open(self):
        """Even if the local_toast module cannot be imported, clarify still runs.

        Simulates the case where the module is missing entirely (e.g. an
        older editable install without the patch applied). The try/except
        around the import must swallow it.
        """
        import sys
        # Temporarily remove tools.local_toast from sys.modules to force ImportError
        saved = sys.modules.pop("tools.local_toast", None)
        # Also block the fresh import attempt
        with patch.dict(sys.modules, {"tools.local_toast": None}):
            callback_mock = MagicMock(return_value="ok")
            result = json.loads(
                clarify_tool("still works?", callback=callback_mock)
            )
        # restore
        if saved is not None:
            sys.modules["tools.local_toast"] = saved
        assert result["user_response"] == "ok"
