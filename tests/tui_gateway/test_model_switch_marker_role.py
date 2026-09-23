"""Tests for _append_model_switch_marker role fix (issue #48338).

The model switch marker must NOT use role="system" because strict providers
(vLLM, Qwen) reject system messages that appear mid-conversation. Using
role="user" is safe — the system prompt is prepended to the API message list,
so a user-role marker can appear at any later position, and the gateway's
sanitize/merge pass already coalesces consecutive user messages.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from tui_gateway.server import _append_model_switch_marker


class TestAppendModelSwitchMarkerRole:
    """Verify the marker uses role='user', not role='system'."""

    def test_marker_uses_user_role(self) -> None:
        """The history entry must be role='user', not role='system'."""
        session: dict = {"session_key": "test-session", "history": []}
        _append_model_switch_marker(session, model="gpt-4o", provider="openai")
        assert len(session["history"]) == 1
        entry = session["history"][0]
        assert entry["role"] == "user", (
            f"Expected role='user' but got role='{entry['role']}'. "
            "Strict providers (vLLM, Qwen) reject mid-conversation system messages."
        )


    def test_no_marker_for_none_session(self) -> None:
        """None session should be a no-op."""
        _append_model_switch_marker(None, model="gpt-4o", provider="openai")


class TestModelSwitchMarkerDedup:
    """#65891: only the newest marker is meaningful; older ones must not
    accumulate in the live history and burn context tokens every turn."""

    @staticmethod
    def _markers(session: dict) -> list:
        from tui_gateway.server import _is_model_switch_marker

        return [h for h in session["history"] if _is_model_switch_marker(h)]

    def test_second_switch_replaces_first_marker(self) -> None:
        session: dict = {"session_key": "s", "history": []}
        _append_model_switch_marker(session, model="model-a", provider="p")
        _append_model_switch_marker(session, model="model-b", provider="p")
        markers = self._markers(session)
        assert len(markers) == 1, "a second switch must replace, not stack, the marker"
        assert "model-b" in markers[0]["content"]
        assert "model-a" not in markers[0]["content"]
        # The surviving marker is the last history entry.
        assert session["history"][-1] is markers[0]

    def test_five_switches_leave_one_marker(self) -> None:
        # Mirrors the issue's screenshot: 5 consecutive MoA preset switches.
        session: dict = {"session_key": "s", "history": []}
        for name in ("质量-非高峰", "省钱-非高峰", "代码编程-非高峰", "日常对话-非高峰", "智能-高峰"):
            _append_model_switch_marker(session, model=name, provider="moa")
        markers = self._markers(session)
        assert len(markers) == 1
        assert "智能-高峰" in markers[0]["content"]

    def test_dedup_preserves_real_conversation_turns(self) -> None:
        session: dict = {
            "session_key": "s",
            "history": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
        _append_model_switch_marker(session, model="model-a", provider="p")
        _append_model_switch_marker(session, model="model-b", provider="p")
        # Real turns untouched; exactly one marker, appended at the end.
        assert session["history"][0] == {"role": "user", "content": "hello"}
        assert session["history"][1] == {"role": "assistant", "content": "hi"}
        assert len(self._markers(session)) == 1
        assert len(session["history"]) == 3

    def test_prior_marker_between_turns_is_removed(self) -> None:
        # A stale marker not at the tail (a later turn followed it) is still
        # stripped on the next switch.
        session: dict = {
            "session_key": "s",
            "history": [
                {"role": "user", "content": "q1"},
                _make_marker_entry("model-a"),
                {"role": "assistant", "content": "a1"},
            ],
        }
        _append_model_switch_marker(session, model="model-b", provider="p")
        markers = self._markers(session)
        assert len(markers) == 1
        assert "model-b" in markers[0]["content"]
        # The real turns are preserved in order.
        assert [h["content"] for h in session["history"] if not _is_marker(h)] == ["q1", "a1"]

    def test_history_version_increments_once_on_replace(self) -> None:
        session: dict = {"session_key": "s", "history": [], "history_version": 0}
        _append_model_switch_marker(session, model="model-a", provider="p")
        _append_model_switch_marker(session, model="model-b", provider="p")
        assert session["history_version"] == 2  # one increment per switch


class TestModelSwitchMarkerMergedWithUserTurn:
    """The per-request alternation repair merges a marker with the user turn after it, and that
    merged entry is committed back to session history. The next switch must peel the stale marker
    off, not drop the entry: dropping it lost the user's message and glued the assistant replies
    around it into one turn."""

    def test_next_switch_keeps_user_turn_merged_into_marker(self) -> None:
        from agent.agent_runtime_helpers import repair_message_sequence

        session: dict = {"session_key": "s", "history": [
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]}
        _append_model_switch_marker(session, model="model-b", provider="p")
        # One turn after the switch, as the gateway runs it: repair before the request, then commit.
        messages = list(session["history"]) + [{"role": "user", "content": "q2"}]
        repair_message_sequence(None, messages)
        session["history"] = messages + [{"role": "assistant", "content": "a2"}]

        _append_model_switch_marker(session, model="model-c", provider="p")

        assert [(h["role"], h["content"]) for h in session["history"] if not _is_marker(h)] == [
            ("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]
        markers = [h for h in session["history"] if _is_marker(h)]
        assert len(markers) == 1 and "model-c" in markers[0]["content"]

    def test_unmerged_entry_loses_marker_metadata(self) -> None:
        from tui_gateway.server import _MODEL_SWITCH_MARKER_PREFIX

        merged = {
            "role": "user",
            # Model names can carry brackets; the marker must still peel off whole.
            "content": f"{_MODEL_SWITCH_MARKER_PREFIX}claude-opus[1m] via provider p. Use it.]\n\nq2",
            "display_kind": "model_switch",
            "api_content": "stale sidecar",
        }
        session: dict = {"session_key": "s", "history": [merged]}
        assert not _is_marker(merged)
        _append_model_switch_marker(session, model="model-c", provider="p")
        assert session["history"][0] is merged  # peeled in place, not replaced
        assert merged == {"role": "user", "content": "q2"}

    def test_stacked_markers_merged_into_one_entry_are_all_peeled(self) -> None:
        stacked = {"role": "user", "content": "\n\n".join(
            [_make_marker_entry("model-a")["content"], _make_marker_entry("model-b")["content"], "q2"])}
        session: dict = {"session_key": "s", "history": [stacked]}
        _append_model_switch_marker(session, model="model-c", provider="p")
        assert stacked["content"] == "q2"

    def test_markers_merged_only_with_each_other_are_dropped(self) -> None:
        both = {"role": "user", "content": "\n\n".join(
            [_make_marker_entry("model-a")["content"], _make_marker_entry("model-b")["content"]])}
        assert _is_marker(both)
        session: dict = {"session_key": "s", "history": [both]}
        _append_model_switch_marker(session, model="model-c", provider="p")
        assert len(session["history"]) == 1 and "model-c" in session["history"][0]["content"]


def _make_marker_entry(model: str) -> dict:
    from tui_gateway.server import _MODEL_SWITCH_MARKER_PREFIX

    return {"role": "user", "content": f"{_MODEL_SWITCH_MARKER_PREFIX}{model}.]"}


def _is_marker(entry: dict) -> bool:
    from tui_gateway.server import _is_model_switch_marker

    return _is_model_switch_marker(entry)
