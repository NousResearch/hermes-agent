"""Tests for delegate_tool Phase C+ — background execution and send_message."""

import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import (
    send_message,
    get_subagent_messages,
    get_subagent_result,
    _register_subagent,
    _unregister_subagent,
    _get_background_executor,
    _deliver_pending_messages,
    _subagent_message_queues,
    _subagent_message_lock,
    _background_results,
    _background_results_lock,
)


# ── Background Executor ─────────────────────────────────────────────────

class TestBackgroundExecutor:

    def test_executor_is_singleton(self):
        e1 = _get_background_executor()
        e2 = _get_background_executor()
        assert e1 is e2

    def test_get_result_unknown_returns_none(self):
        assert get_subagent_result("nonexistent") is None

    def test_get_result_stores_and_retrieves(self):
        with _background_results_lock:
            _background_results["sa-test"] = {"status": "completed", "summary": "done"}

        result = get_subagent_result("sa-test")
        assert result is not None
        assert result["status"] == "completed"

        # Second retrieval returns None (already consumed)
        result2 = get_subagent_result("sa-test")
        assert result2 is None


# ── send_message ────────────────────────────────────────────────────────

class TestSendMessage:

    def setup_method(self):
        self.test_id = f"sa-test-msg-{time.time()}"
        _register_subagent({
            "subagent_id": self.test_id,
            "parent_id": None,
            "depth": 0,
            "goal": "test",
            "model": None,
            "started_at": time.time(),
            "status": "running",
            "tool_count": 0,
            "agent": None,
        })

    def teardown_method(self):
        _unregister_subagent(self.test_id)
        with _subagent_message_lock:
            _subagent_message_queues.pop(self.test_id, None)

    def test_send_message_to_running_subagent_succeeds(self):
        result = send_message(self.test_id, "Please check the results", sender="parent")
        assert result is True

    def test_send_message_to_unknown_subagent_fails(self):
        result = send_message("nonexistent-id", "hello")
        assert result is False

    def test_get_messages_retrieves_queued(self):
        send_message(self.test_id, "msg 1")
        send_message(self.test_id, "msg 2")
        messages = get_subagent_messages(self.test_id)
        assert len(messages) == 2
        assert messages[0]["content"] == "msg 1"
        assert messages[1]["content"] == "msg 2"
        assert messages[0]["sender"] == "parent"

    def test_get_messages_mark_read_clears_queue(self):
        send_message(self.test_id, "msg")
        first = get_subagent_messages(self.test_id, mark_read=True)
        assert len(first) == 1
        second = get_subagent_messages(self.test_id)
        assert len(second) == 0

    def test_get_messages_unknown_returns_none(self):
        assert get_subagent_messages("nonexistent") is None

    def test_multiple_senders(self):
        send_message(self.test_id, "from parent", sender="parent")
        send_message(self.test_id, "from user", sender="user")
        messages = get_subagent_messages(self.test_id)
        assert len(messages) == 2
        senders = {m["sender"] for m in messages}
        assert "parent" in senders
        assert "user" in senders


# ── Message Delivery ────────────────────────────────────────────────────

class TestDeliverPendingMessages:

    def test_deliver_injects_into_conversation_history(self):
        test_id = "sa-test-deliver"
        _register_subagent({
            "subagent_id": test_id,
            "parent_id": None,
            "depth": 0,
            "goal": "test",
            "model": None,
            "started_at": time.time(),
            "status": "running",
            "tool_count": 0,
            "agent": None,
        })

        try:
            send_message(test_id, "check results", sender="parent")

            mock_agent = MagicMock()
            mock_agent._conversation_history = []

            delivered = _deliver_pending_messages(test_id, child_agent=mock_agent)
            assert len(delivered) == 1
            assert len(mock_agent._conversation_history) == 1
            injected = mock_agent._conversation_history[0]
            assert injected["role"] == "user"
            assert "check results" in injected["content"]
            assert "parent" in injected["content"]
        finally:
            _unregister_subagent(test_id)
            with _subagent_message_lock:
                _subagent_message_queues.pop(test_id, None)

    def test_deliver_no_messages_is_noop(self):
        mock_agent = MagicMock()
        mock_agent._conversation_history = []
        delivered = _deliver_pending_messages("no-messages", child_agent=mock_agent)
        assert delivered == []
        assert mock_agent._conversation_history == []

    def test_deliver_without_agent_ref_returns_messages(self):
        test_id = "sa-test-no-agent"
        _register_subagent({
            "subagent_id": test_id,
            "parent_id": None,
            "depth": 0,
            "goal": "test",
            "model": None,
            "started_at": time.time(),
            "status": "running",
            "tool_count": 0,
            "agent": None,
        })

        try:
            send_message(test_id, "cannot deliver")
            delivered = _deliver_pending_messages(test_id, child_agent=None)
            # Returns messages but can't deliver (no agent ref)
            assert len(delivered) == 1
        finally:
            _unregister_subagent(test_id)
            with _subagent_message_lock:
                _subagent_message_queues.pop(test_id, None)
