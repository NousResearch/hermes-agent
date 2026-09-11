"""Provider/display history regression tests for model-switch markers."""

import threading

from agent.agent_runtime_helpers import repair_message_sequence
from tui_gateway.prompt_turn import (
    _build_display_history,
    _canonicalize_model_switch_history,
    _prepend_model_switch_marker,
)

MARKER = {
    "role": "user",
    "display_kind": "model_switch",
    "content": "[System: The active model for this chat has changed to model-b via provider p. From this point forward, use this runtime metadata when answering questions about what model/provider is active.]",
}


def test_tail_marker_is_folded_for_provider_without_mutating_display_history():
    raw = [{"role": "assistant", "content": "previous"}, MARKER.copy()]
    provider, marker, marker_tail, changed = _canonicalize_model_switch_history(raw)
    messages = provider + [{"role": "user", "content": _prepend_model_switch_marker(marker, "hello")}]
    assert changed is True
    assert marker == MARKER["content"]
    assert [m["role"] for m in messages] == ["assistant", "user"]
    assert repair_message_sequence(None, messages) == 0
    assert raw[1] == MARKER


def test_historical_marker_is_folded_only_in_provider_copy():
    raw = [
        {"role": "assistant", "content": "previous"}, MARKER.copy(),
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "reply"},
    ]
    provider, marker, _, changed = _canonicalize_model_switch_history(raw)
    assert changed is True and marker is None
    assert [m["role"] for m in provider] == ["assistant", "user", "assistant"]
    assert MARKER["content"] in provider[1]["content"]
    assert raw[1]["display_kind"] == "model_switch" and raw[2]["content"] == "hello"


def test_display_history_restores_marker_and_clean_user():
    raw = [{"role": "assistant", "content": "previous"}, MARKER.copy()]
    provider, marker, marker_tail, _ = _canonicalize_model_switch_history(raw)
    result = provider + [
        {"role": "user", "content": _prepend_model_switch_marker(marker, "hello")},
        {"role": "assistant", "content": "reply"},
    ]
    assert _build_display_history(raw, result, len(provider), marker, marker_tail) == raw + [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "reply"}
    ]


def test_commit_uses_display_history_override():
    from tui_gateway.prompt_turn import _commit_turn_history

    raw = [{"role": "assistant", "content": "previous"}, MARKER.copy()]
    display = raw + [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "reply"},
    ]
    result = {
        "messages": [{"role": "user", "content": "provider copy"}],
        "_tui_display_messages": display,
    }
    session = {"history": raw, "history_version": 0, "history_lock": threading.RLock()}
    assert _commit_turn_history(session, result, raw, 0) is None
    assert session["history"] == display
    assert "_tui_display_messages" not in result


def test_polluted_marker_preserves_legacy_text_once_with_new_user():
    polluted = MARKER.copy()
    polluted["content"] += "\n\nold user text"
    provider, marker, marker_tail, changed = _canonicalize_model_switch_history([polluted])
    combined = _prepend_model_switch_marker(
        marker,
        _prepend_model_switch_marker(marker_tail, "new user text"),
    )
    assert changed is True and provider == []
    assert combined.count("old user text") == 1
    assert combined.count("new user text") == 1


def test_history_without_marker_is_a_no_op():
    raw = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "reply"},
    ]
    provider, marker, marker_tail, changed = _canonicalize_model_switch_history(raw)
    assert provider == raw
    assert marker is None and marker_tail is None and changed is False


def test_marker_preserves_multimodal_content_parts():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    original = [image, {"type": "text", "text": "hello"}]
    result = _prepend_model_switch_marker(MARKER["content"], original)
    assert isinstance(result, list)
    assert result[0] == {"type": "text", "text": MARKER["content"] + "\n\n"}
    assert result[1:] == original
    assert result[1] is not image
