"""Tests for TUI gateway /compress lock-hold signalling."""
from unittest.mock import MagicMock, patch

import pytest


def test_compress_session_history_raises_on_lock_skip():
    """When _compression_skipped_due_to_lock is set on the agent,
    _compress_session_history must raise CompressionLockHeld with
    the holder string so callers can surface a clear message."""
    from tui_gateway.server import _compress_session_history, CompressionLockHeld

    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    agent = MagicMock()
    agent._cached_system_prompt = ""
    agent.tools = None
    agent._compression_skipped_due_to_lock = "pid=99999:tid=1:agent=1:nonce=abc"

    def _fake_compress(msgs=None, *_args, **_kwargs):
        return (msgs or history, "")

    agent._compress_context.side_effect = _fake_compress

    session = {
        "agent": agent,
        "history_lock": MagicMock(),
        "history": history,
        "history_version": 1,
    }

    with (
        patch(
            "agent.model_metadata.estimate_request_tokens_rough", return_value=100
        ),
        pytest.raises(CompressionLockHeld) as exc_info,
    ):
        _compress_session_history(session)

    assert exc_info.value.holder == "pid=99999:tid=1:agent=1:nonce=abc"


def test_compress_session_history_clears_signal_after_raise():
    """The signal attribute must be cleared when the exception is raised
    so stale signals don't leak into subsequent operations."""
    from tui_gateway.server import _compress_session_history, CompressionLockHeld

    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    agent = MagicMock()
    agent._cached_system_prompt = ""
    agent.tools = None
    agent._compression_skipped_due_to_lock = True

    def _fake_compress(msgs=None, *_args, **_kwargs):
        return (msgs or history, "")

    agent._compress_context.side_effect = _fake_compress

    session = {
        "agent": agent,
        "history_lock": MagicMock(),
        "history": history,
        "history_version": 1,
    }

    with (
        patch(
            "agent.model_metadata.estimate_request_tokens_rough", return_value=100
        ),
        pytest.raises(CompressionLockHeld),
    ):
        _compress_session_history(session)

    # Signal must be cleared after the raise.
    assert agent._compression_skipped_due_to_lock is None
