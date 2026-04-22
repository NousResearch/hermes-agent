"""Regression tests for OpenAI-compatible TTS config resolution."""

from types import SimpleNamespace

from tools import tts_tool


def _disable_managed_gateway(monkeypatch) -> None:
    monkeypatch.setattr(tts_tool, "prefers_gateway", lambda _tool: False)
    monkeypatch.setattr(tts_tool, "resolve_managed_tool_gateway", lambda _tool: None)


def test_config_credentials_take_precedence_over_environment(monkeypatch):
    _disable_managed_gateway(monkeypatch)
    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {
            "openai": {
                "api_key": "config-key",
                "base_url": "http://localhost:8080/v1",
            }
        },
    )
    monkeypatch.setattr(
        tts_tool,
        "resolve_openai_audio_api_key",
        lambda: "environment-key",
    )

    assert tts_tool._resolve_openai_audio_client_config() == (
        "config-key",
        "http://localhost:8080/v1",
        False,
    )


def test_config_base_url_combines_with_environment_key(monkeypatch):
    _disable_managed_gateway(monkeypatch)
    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"openai": {"base_url": "http://localhost:8080/v1"}},
    )
    monkeypatch.setattr(
        tts_tool,
        "resolve_openai_audio_api_key",
        lambda: "environment-key",
    )

    assert tts_tool._resolve_openai_audio_client_config() == (
        "environment-key",
        "http://localhost:8080/v1",
        False,
    )


def test_gateway_preference_preserves_managed_result(monkeypatch):
    monkeypatch.setattr(tts_tool, "prefers_gateway", lambda _tool: True)
    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"openai": {"api_key": "config-key"}},
    )
    monkeypatch.setattr(
        tts_tool,
        "resolve_managed_tool_gateway",
        lambda _tool: SimpleNamespace(
            nous_user_token="managed-token",
            gateway_origin="https://gateway.example",
        ),
    )

    assert tts_tool._resolve_openai_audio_client_config() == (
        "managed-token",
        "https://gateway.example/v1",
        True,
    )


def test_config_only_credentials_make_openai_available(monkeypatch):
    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"openai": {"api_key": "config-key"}},
    )
    monkeypatch.setattr(tts_tool, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(tts_tool, "resolve_managed_tool_gateway", lambda _tool: None)

    assert tts_tool._has_openai_audio_backend() is True
