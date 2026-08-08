"""text_to_speech must reject non-string text without AttributeError."""

from __future__ import annotations

import json

from tools.tts_tool import text_to_speech_tool


def test_non_string_text_returns_error_not_attribute_error():
    for bad in (42, ["hello"], {"t": "x"}, None, ""):
        result = json.loads(text_to_speech_tool(text=bad))
        assert result.get("success") is False, bad
        assert "Text is required" in result["error"]


def test_blank_text_returns_error():
    result = json.loads(text_to_speech_tool(text="   "))
    assert result.get("success") is False
    assert "Text is required" in result["error"]


def test_non_string_provider_returns_error():
    result = json.loads(text_to_speech_tool(text="hello", provider=123))
    assert result.get("success") is False
    assert "provider must be a string" in result["error"]


def test_malformed_speed_does_not_raise_before_provider(monkeypatch):
    """Non-numeric speed must not ValueError on float(speed)."""
    monkeypatch.setattr(
        "tools.tts_tool._load_tts_config",
        lambda: {"provider": "edge"},
    )
    # Stop right after speed coercion by forcing a known tool_error path:
    # empty provider name after override is invalid for our purposes — use a
    # string provider and stub _get_provider so we never hit the network.
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda _cfg: "edge")
    monkeypatch.setattr(
        "tools.tts_tool._resolve_command_provider_config",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "tools.tts_tool._resolve_max_text_length",
        lambda *_a, **_k: 1000,
    )

    # Stub the provider dispatch table entry if present; otherwise accept any
    # tool_error JSON as long as float(speed) did not raise.
    import tools.tts_tool as mod

    def _stop(*_a, **_k):
        return json.dumps({"success": False, "error": "stubbed"})

    for name in (
        "_run_edge_tts",
        "synthesize_edge",
        "_synthesize_edge",
        "_dispatch_provider",
    ):
        if hasattr(mod, name):
            monkeypatch.setattr(mod, name, _stop)

    try:
        result = json.loads(
            text_to_speech_tool(text="hello world", speed="fast")
        )
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"malformed speed crashed: {exc}") from exc

    assert isinstance(result, dict)
