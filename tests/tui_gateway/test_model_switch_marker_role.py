"""Tests for _append_model_switch_marker role semantics (issue #48338).

The model switch marker is tagged and uses role="system" for correct transcript
semantics. The pre-call sanitizer demotes only tagged Hermes system markers to
role="user" for strict-provider compatibility; persisted transcript and Desktop
rendering keep the system role.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from tui_gateway.server import _append_model_switch_marker


class TestAppendModelSwitchMarkerRole:
    """Verify the marker uses role='system' (demoted to 'user' at API call time)."""

    def test_marker_uses_system_role(self) -> None:
        """The history entry must be role='system' for correct semantics."""
        session: dict = {"session_key": "test-session", "history": []}
        _append_model_switch_marker(session, model="gpt-4o", provider="openai")
        assert len(session["history"]) == 1
        entry = session["history"][0]
        assert entry["role"] == "system", (
            f"Expected role='system' but got role='{entry['role']}'. "
            "The sanitizer demotes to 'user' at API call time (#48338)."
        )
        assert entry["_hermes_internal_system_marker"] is True

    def test_marker_content_preserved(self) -> None:
        """The marker content must still describe the model switch."""
        session: dict = {"session_key": "s", "history": []}
        _append_model_switch_marker(session, model="qwen3.6-35b", provider="vllm")
        content = session["history"][0]["content"]
        assert "qwen3.6-35b" in content
        assert "vllm" in content
        assert "model" in content.lower()

    def test_marker_with_empty_provider(self) -> None:
        """Provider part should be omitted when provider is empty."""
        session: dict = {"session_key": "s", "history": []}
        _append_model_switch_marker(session, model="claude-sonnet-4", provider="")
        content = session["history"][0]["content"]
        assert "claude-sonnet-4" in content
        assert "via provider" not in content

    def test_marker_with_lock(self) -> None:
        """Marker should work correctly when session has a history_lock."""
        session: dict = {
            "session_key": "s",
            "history": [],
            "history_lock": threading.Lock(),
        }
        _append_model_switch_marker(session, model="gpt-4o", provider="openai")
        assert len(session["history"]) == 1
        assert session["history"][0]["role"] == "system"

    def test_marker_increments_history_version(self) -> None:
        """history_version should be incremented after appending."""
        session: dict = {"session_key": "s", "history": [], "history_version": 5}
        _append_model_switch_marker(session, model="gpt-4o", provider="openai")
        assert session["history_version"] == 6

    def test_no_marker_for_none_session(self) -> None:
        """None session should be a no-op."""
        _append_model_switch_marker(None, model="gpt-4o", provider="openai")

    def test_no_marker_for_empty_session_key(self) -> None:
        """Empty session_key should be a no-op."""
        session: dict = {"session_key": "", "history": []}
        _append_model_switch_marker(session, model="gpt-4o", provider="openai")
        assert len(session["history"]) == 0

    def test_marker_role_after_turns(self) -> None:
        """The marker appended after real turns uses role='system'.

        The stored transcript has role='system' for correct Desktop rendering.
        The pre-call sanitizer demotes it to 'user' for provider compatibility
        (#48338), so strict OpenAI-compatible providers never see a mid-
        conversation system message on the wire.
        """
        db = MagicMock()
        session: dict = {
            "session_key": "sess-1",
            "history": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            "history_version": 7,
            "agent": SimpleNamespace(_session_db=db),
        }
        _append_model_switch_marker(
            session, model="qwen3.6-35b", provider="vllm"
        )
        marker = session["history"][-1]
        assert marker["role"] == "system"
        assert session["history_version"] == 8
        # The persisted row must mirror the in-memory role.
        db.append_message.assert_called_once_with(
            session_id="sess-1",
            role="system",
            content=marker["content"],
            internal_system_marker=True,
        )
