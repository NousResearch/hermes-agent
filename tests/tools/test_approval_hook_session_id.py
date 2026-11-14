"""Approval hooks must carry the Hermes session id to observer plugins.

Staging defect 2026-08-10: approval marks were emitted under a synthetic
"default" relay session because the approval hook payload carried only
turn_id/tool_call_id — the observability plugin's ``_session_id()`` fell
back to "default", parented the marks to a session scope that never
closes, and close-time exporters never shipped them.  The audit board's
approval tables stayed empty while approvals were demonstrably firing.

Contract: when the dispatch layer binds an observability context with a
session id, every approval hook payload carries that session id; when no
context is bound, the payload omits it (legacy behavior preserved).
"""

from __future__ import annotations

from unittest.mock import patch

from tools import approval as approval_mod
from tools import approval_context


def _capture_hook(captured):
    def _invoke(hook_name, **kwargs):
        captured.append((hook_name, kwargs))
    return _invoke


class TestApprovalHookSessionId:
    def test_session_id_forwarded_when_bound(self):
        captured = []
        tokens = approval_context.set_current_observability_context(
            turn_id="turn-1",
            tool_call_id="call-1",
            session_id="20260810_test_session",
        )
        try:
            with patch(
                "hermes_cli.lifecycle.invoke_hook",
                side_effect=_capture_hook(captured),
            ):
                approval_context._fire_approval_hook(
                    "pre_approval_request",
                    command="rm -rf /etc/hosts",
                    description="dangerous",
                    surface="gateway",
                )
        finally:
            approval_context.reset_current_observability_context(tokens)

        assert captured, "hook must dispatch"
        _, kwargs = captured[0]
        assert kwargs.get("session_id") == "20260810_test_session"
        assert kwargs.get("turn_id") == "turn-1"
        assert kwargs.get("tool_call_id") == "call-1"

    def test_explicit_session_id_not_clobbered(self):
        captured = []
        tokens = approval_context.set_current_observability_context(
            session_id="context-session",
        )
        try:
            with patch(
                "hermes_cli.lifecycle.invoke_hook",
                side_effect=_capture_hook(captured),
            ):
                approval_context._fire_approval_hook(
                    "post_approval_response",
                    session_id="explicit-session",
                    choice="approved",
                )
        finally:
            approval_context.reset_current_observability_context(tokens)

        _, kwargs = captured[0]
        assert kwargs.get("session_id") == "explicit-session"

    def test_absent_when_unbound(self):
        captured = []
        with patch(
            "hermes_cli.lifecycle.invoke_hook",
            side_effect=_capture_hook(captured),
        ):
            approval_context._fire_approval_hook(
                "pre_approval_request",
                command="x",
                description="y",
            )
        _, kwargs = captured[0]
        assert "session_id" not in kwargs, (
            "no synthetic session id when none is bound — the observer's "
            "own fallback owns that decision"
        )
