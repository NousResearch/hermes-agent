"""Requester and approver identity on the gateway approval path.

A shared Discord thread has several senders; the agent's cached ``_user_id`` is
whoever built it. Approval observers therefore get the requester from the
per-message session contextvars (``user_id`` / ``platform`` on both hooks, and
``requester_user_id`` / ``requester_platform`` on the queued entry the gateway
renders), and the resolver reports who answered via ``decided_by``.
"""

import threading
import time
from unittest.mock import patch

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from tools.approval import clear_session, list_gateway_approvals, resolve_gateway_approval
from tools.approval_gateway_wait import _await_gateway_decision

SESSION = "discord:chat:shared-thread"


@pytest.fixture
def clean_queue():
    yield
    clear_session(SESSION)


def _wait_for_pending(timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not list_gateway_approvals(SESSION):
        assert time.monotonic() < deadline, "approval never enqueued"
        time.sleep(0.01)


def test_hooks_and_entry_carry_requester_and_decider(clean_queue):
    captured, notified, decision = [], [], {}

    def worker():
        # Contextvars are per-thread: bind the requester the way the gateway does for a turn.
        tokens = set_session_vars(platform="discord", user_id="222", session_key=SESSION)
        try:
            decision["value"] = _await_gateway_decision(
                SESSION, notified.append,
                {"command": "rm -rf /tmp/x", "pattern_key": "rm -rf", "pattern_keys": ["rm -rf"],
                 "description": "deletes files"},
            )
        finally:
            clear_session_vars(tokens)

    with patch("hermes_cli.lifecycle.invoke_hook",
               side_effect=lambda name, **kw: captured.append((name, kw)) or []):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        _wait_for_pending()
        assert resolve_gateway_approval(SESSION, "once", decided_by="discord:111") == 1
        thread.join(timeout=5)
    assert not thread.is_alive()

    assert notified[0]["requester_user_id"] == "222"
    assert notified[0]["requester_platform"] == "discord"
    pre = next(kw for name, kw in captured if name == "pre_approval_request")
    post = next(kw for name, kw in captured if name == "post_approval_response")
    assert pre["user_id"] == post["user_id"] == "222"
    assert pre["platform"] == post["platform"] == "discord"
    assert post["choice"] == "once"
    assert post["decided_by"] == "discord:111"
    assert decision["value"]["decided_by"] == "discord:111"


def test_unknown_decider_leaves_decided_by_absent(clean_queue):
    """Legacy resolvers pass no ``decided_by``; the payload must not fake one
    (smart approvals own ``decided_by="aux_llm"``)."""
    captured = []

    def worker():
        _await_gateway_decision(SESSION, lambda data: None,
                                {"command": "rm -rf /tmp/y", "pattern_key": "rm -rf", "pattern_keys": ["rm -rf"]})

    with patch("hermes_cli.lifecycle.invoke_hook",
               side_effect=lambda name, **kw: captured.append((name, kw)) or []):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        _wait_for_pending()
        assert resolve_gateway_approval(SESSION, "deny") == 1
        thread.join(timeout=5)

    post = next(kw for name, kw in captured if name == "post_approval_response")
    assert "decided_by" not in post
    assert post["user_id"] == ""  # unknown requester is "" on the wire, never missing
