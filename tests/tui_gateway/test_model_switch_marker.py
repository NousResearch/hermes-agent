"""The TUI model-switch marker must NOT inject a mid-conversation system message
(strict OpenAI-compatible backends like vLLM/Qwen reject "System message must be
at the beginning"). It stages a one-shot pending note for the next user turn.
"""
import tui_gateway.server as srv


def test_marker_stages_pending_note_and_no_system_history():
    session = {
        "session_key": "s1",
        "history": [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"}],
        "history_version": 0,
        "history_lock": __import__("threading").RLock(),
    }
    srv._append_model_switch_marker(session, model="main", provider="custom:main-think")

    # No role:"system" message was appended to the conversation history.
    assert all(m.get("role") != "system" for m in session["history"])
    # History length unchanged (no extra message injected).
    assert len(session["history"]) == 2
    # A one-shot note was staged, naming the new model + provider.
    note = session.get("pending_model_note")
    assert note and "main" in note and "custom:main-think" in note


def test_marker_noop_without_session():
    # Must not raise on a missing session.
    srv._append_model_switch_marker(None, model="m", provider="p")
