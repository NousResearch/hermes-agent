"""Tests for _append_model_switch_marker role fix (issue #48338).

The model switch marker must NOT use role="system" because strict providers
(vLLM, Qwen) reject system messages that appear mid-conversation. Using
role="user" is safe — the system prompt is prepended to the API message list,
so a user-role marker can appear at any later position, and the gateway's
sanitize/merge pass already coalesces consecutive user messages.
"""

from __future__ import annotations

import threading

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


class TestMarkerMergedWithPrompt:
    """#131382: the marker is role=user, so the turn loop's alternation repair folds the next real
    prompt into it (``marker + "\\n\\n" + prompt``). The next switch drops the stale marker, and must
    keep the prompt it absorbed: the transcript still shows it, so the model must still see it."""

    @staticmethod
    def _switched_then_prompted() -> dict:
        from agent.agent_runtime_helpers import repair_message_sequence

        session: dict = {
            "session_key": "s",
            "history": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}],
        }
        _append_model_switch_marker(session, model="model-a", provider="p")
        # What the turn loop does: append the prompt, repair the user;user pair in place, and that
        # list (plus the reply) becomes the session history.
        turn = session["history"] + [{"role": "user", "content": "the prompt"}]
        assert repair_message_sequence(None, turn) == 1
        session["history"] = turn + [{"role": "assistant", "content": "a2"}]
        return session

    def test_next_switch_keeps_the_absorbed_prompt(self) -> None:
        session = self._switched_then_prompted()
        _append_model_switch_marker(session, model="model-b", provider="p")
        contents = [h["content"] for h in session["history"]]
        assert contents[:4] == ["q1", "a1", "the prompt", "a2"]
        assert len(contents) == 5 and "model-b" in contents[4]
        assert not any("model-a" in c for c in contents), "the stale marker text is still dropped"

    def test_switch_during_the_next_turn_keeps_both_turns(self) -> None:
        from tui_gateway.server import _commit_turn_history

        session = self._switched_then_prompted()
        session.update(history_lock=threading.Lock(), history_version=5)
        turn_start = list(session["history"])
        result = {"messages": turn_start + [{"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"}]}
        _append_model_switch_marker(session, model="model-b", provider="p")  # lands while the turn runs
        assert _commit_turn_history(session, result, turn_start, 5) is None
        contents = [h["content"] for h in session["history"]]
        assert contents[:4] == ["q1", "a1", "the prompt", "a2"]
        assert contents[-2:] == ["q3", "a3"]


def _make_marker_entry(model: str) -> dict:
    from tui_gateway.server import _MODEL_SWITCH_MARKER_PREFIX

    return {"role": "user", "content": f"{_MODEL_SWITCH_MARKER_PREFIX}{model}.]"}


def _is_marker(entry: dict) -> bool:
    from tui_gateway.server import _is_model_switch_marker

    return _is_model_switch_marker(entry)
