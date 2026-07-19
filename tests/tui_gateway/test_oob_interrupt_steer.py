"""
Tests for the OOB steer injection on mid-turn interrupt.

When busy_input_mode='interrupt' and a user submits mid-turn,
the agent should receive:
1. agent.steer() called with OOB-formatted message
2. agent.interrupt() called
3. message queued as next turn
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import MagicMock, patch, call
from tui_gateway.server import _handle_busy_submit


def _make_session(agent=None):
    return {
        "agent": agent,
        "queued_prompt": None,
        "last_active": 0.0,
    }


def _make_agent():
    agent = MagicMock()
    agent.steer = MagicMock(return_value=True)
    agent.interrupt = MagicMock()
    return agent


class TestOOBSteerOnInterrupt:

    def test_interrupt_mode_calls_steer_then_interrupt(self):
        """In interrupt mode, steer() is called with OOB marker before interrupt()."""
        agent = _make_agent()
        session = _make_session(agent)
        text = "Stop! You missed the point."

        with patch("tui_gateway.server._load_busy_input_mode", return_value="interrupt"):
            result = _handle_busy_submit("rid1", "sid1", session, text, None)

        assert result.get("result", {}).get("status") in ("queued", None) or "status" in str(result)
        assert agent.steer.called, "steer() should be called in interrupt mode"
        assert agent.interrupt.called, "interrupt() should be called in interrupt mode"

        # Verify steer was called with OOB-formatted text
        steer_arg = agent.steer.call_args[0][0]
        assert "OUT-OF-BAND USER MESSAGE" in steer_arg, "steer arg should contain OOB marker"
        assert text in steer_arg, "steer arg should contain the user's message"

    def test_interrupt_mode_steer_called_before_interrupt(self):
        """steer() must be called before interrupt() so the message is in the queue."""
        agent = _make_agent()
        session = _make_session(agent)
        call_order = []
        agent.steer.side_effect = lambda x: call_order.append("steer") or True
        agent.interrupt.side_effect = lambda: call_order.append("interrupt")

        with patch("tui_gateway.server._load_busy_input_mode", return_value="interrupt"):
            _handle_busy_submit("rid2", "sid2", session, "hello", None)

        assert call_order == ["steer", "interrupt"], f"Expected steer before interrupt, got {call_order}"

    def test_queue_mode_does_not_steer_or_interrupt(self):
        """In queue mode, neither steer() nor interrupt() should be called."""
        agent = _make_agent()
        session = _make_session(agent)

        with patch("tui_gateway.server._load_busy_input_mode", return_value="queue"):
            _handle_busy_submit("rid3", "sid3", session, "queued message", None)

        agent.steer.assert_not_called()
        agent.interrupt.assert_not_called()

    def test_steer_mode_uses_agent_steer_not_interrupt(self):
        """In steer mode, agent.steer() is called but interrupt() is not (steer accepted)."""
        agent = _make_agent()
        agent.steer.return_value = True  # steer accepted
        session = _make_session(agent)

        with patch("tui_gateway.server._load_busy_input_mode", return_value="steer"):
            _handle_busy_submit("rid4", "sid4", session, "steer message", None)

        agent.steer.assert_called_once()
        agent.interrupt.assert_not_called()

    def test_message_still_queued_in_interrupt_mode(self):
        """Even with OOB steer, the message is still queued as the next turn."""
        agent = _make_agent()
        session = _make_session(agent)
        text = "important message"

        with patch("tui_gateway.server._load_busy_input_mode", return_value="interrupt"):
            _handle_busy_submit("rid5", "sid5", session, text, None)

        queued = session.get("queued_prompt")
        assert queued is not None, "Message should be queued as next turn"
        assert queued.get("text") == text, "Queued text should be original user message"

    def test_oob_steer_does_not_replace_queued_message(self):
        """The OOB steer is injected into running context; the queue gets the original text."""
        agent = _make_agent()
        session = _make_session(agent)
        text = "stop and reconsider"

        with patch("tui_gateway.server._load_busy_input_mode", return_value="interrupt"):
            _handle_busy_submit("rid6", "sid6", session, text, None)

        # Queue should have the plain original text, not the OOB-formatted version
        queued_text = session.get("queued_prompt", {}).get("text", "")
        assert "OUT-OF-BAND" not in queued_text, "Queue should contain plain text, not OOB marker"
        assert queued_text == text

    def test_no_agent_still_queues_message(self):
        """If no agent is present, message is still queued without error."""
        session = _make_session(agent=None)
        text = "no agent running"

        with patch("tui_gateway.server._load_busy_input_mode", return_value="interrupt"):
            result = _handle_busy_submit("rid7", "sid7", session, text, None)

        assert session.get("queued_prompt") is not None
