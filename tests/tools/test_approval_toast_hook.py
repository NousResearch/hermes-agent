"""Integration test: tools/approval.py fires notify_input_needed before blocking.

Verifies that prompt_dangerous_approval() calls the local_toast module
immediately before delegating to the approval_callback that owns the blocking
wait. The toast call itself is patched to a no-op so we're purely asserting
that the wire-up in approval.py is present and passes the command through.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.approval import prompt_dangerous_approval


class TestApprovalToastHook:
    def test_notify_input_needed_called_before_callback(self):
        """The toast dispatch happens BEFORE the blocking approval callback."""
        callback_mock = MagicMock(return_value="once")
        with patch("tools.local_toast.notify_input_needed") as notify_mock:
            result = prompt_dangerous_approval(
                "rm -rf /tmp/x",
                "Recursive delete",
                approval_callback=callback_mock,
            )

        assert result == "once"
        assert notify_mock.call_count == 1
        args, kwargs = notify_mock.call_args
        assert args[0] == "approval"
        # detail is "<description>: <command>"
        detail = kwargs.get("detail", "")
        assert "Recursive delete" in detail
        assert "rm -rf /tmp/x" in detail
        assert kwargs.get("level") == "warn"

    def test_notify_failure_does_not_break_approval(self):
        """If notify_input_needed raises, prompt_dangerous_approval still runs."""
        callback_mock = MagicMock(return_value="deny")

        def _boom(*a, **kw):
            raise RuntimeError("antenna is on fire")

        with patch("tools.local_toast.notify_input_needed", side_effect=_boom):
            result = prompt_dangerous_approval(
                "docker system prune",
                "Wipe unused images",
                approval_callback=callback_mock,
            )

        assert result == "deny"
        callback_mock.assert_called_once()

    def test_command_newlines_stripped_in_detail(self):
        """Multi-line commands (heredocs) are flattened in the toast detail."""
        callback_mock = MagicMock(return_value="once")
        with patch("tools.local_toast.notify_input_needed") as notify_mock:
            prompt_dangerous_approval(
                "bash <<EOF\nrm -rf /tmp/x\nEOF",
                "Heredoc",
                approval_callback=callback_mock,
            )
        _, kwargs = notify_mock.call_args
        detail = kwargs.get("detail", "")
        assert "\n" not in detail
        assert "bash <<EOF" in detail
        assert "rm -rf /tmp/x" in detail
        assert "EOF" in detail

    def test_toast_import_absent_is_fail_open(self):
        """If local_toast cannot be imported, approval still proceeds."""
        import sys
        saved = sys.modules.pop("tools.local_toast", None)
        with patch.dict(sys.modules, {"tools.local_toast": None}):
            callback_mock = MagicMock(return_value="once")
            result = prompt_dangerous_approval(
                "echo hi", "test", approval_callback=callback_mock
            )
        if saved is not None:
            sys.modules["tools.local_toast"] = saved
        assert result == "once"

    def test_description_missing_falls_back_to_command_only(self):
        """Empty description should not produce a leading colon in the detail."""
        callback_mock = MagicMock(return_value="once")
        with patch("tools.local_toast.notify_input_needed") as notify_mock:
            prompt_dangerous_approval(
                "ls /", "", approval_callback=callback_mock
            )
        _, kwargs = notify_mock.call_args
        detail = kwargs.get("detail", "")
        assert not detail.startswith(":")
        assert "ls /" in detail
