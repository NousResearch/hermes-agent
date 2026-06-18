"""Regression tests for gateway model-switch history markers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from tui_gateway.server import _append_model_switch_marker


def test_model_switch_marker_is_user_message_not_mid_history_system():
    """Model switches must not inject a system role after conversation turns.

    Strict OpenAI-compatible providers such as vLLM reject message arrays that
    contain system messages after the beginning of the conversation.
    """
    db = MagicMock()
    session = {
        "session_key": "sess-1",
        "history": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "history_version": 7,
        "agent": SimpleNamespace(_session_db=db),
    }

    _append_model_switch_marker(
        session,
        model="qwen3.6-35b",
        provider="api.lucasnicolas.dev",
    )

    marker = session["history"][-1]
    assert marker["role"] == "user"
    assert "active model for this chat has changed to qwen3.6-35b" in marker["content"]
    assert session["history_version"] == 8
    db.append_message.assert_called_once_with(
        session_id="sess-1",
        role="user",
        content=marker["content"],
    )
