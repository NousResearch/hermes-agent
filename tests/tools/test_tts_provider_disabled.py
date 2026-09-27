"""``tts.provider`` set to a disabled value must never synthesize audio.

Before the fix ``_get_provider()`` coalesced YAML's bool ``False`` (``provider: off`` / ``false`` /
``no``) into the Edge default, and a string ``none`` reached ``_select_builtin_engine()``'s
"unknown name -> Edge default" branch: a user who had switched TTS off still got an edge-tts
voice message in front of every gateway text reply (#118050).
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_cli.config as hc


@pytest.mark.parametrize("value", ["none", "off", "disabled", "false", "no", " None ", False])
def test_disabled_spellings_resolve_to_none(value):
    """Every disabled spelling — including the bool PyYAML produces for ``off``/``false``/``no`` —
    resolves to the canonical ``"none"``; nothing about the default changes for real names."""
    from tools.tts_tool import DEFAULT_PROVIDER, _get_provider

    assert _get_provider({"provider": value}) == "none"
    assert _get_provider({}) == DEFAULT_PROVIDER
    assert _get_provider({"provider": "elevenlabs"}) == "elevenlabs"


@pytest.mark.parametrize("yaml_value", ["none", "off"])
def test_config_yaml_disabled_provider_synthesizes_nothing(yaml_value, tmp_path, monkeypatch):
    """E2E through a real ``config.yaml`` in a temp HERMES_HOME: the registered ``text_to_speech``
    handler refuses, the toolset ``check_fn`` hides the tool, the gateway auto voice reply is
    skipped, and the Edge engine is never invoked. ``off`` covers the YAML-bool case."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(f"tts:\n  provider: {yaml_value}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    hc._LOAD_CONFIG_CACHE.clear()
    try:
        import tools.tts_tool as tts_tool
        from gateway.config import Platform
        from gateway.platforms.event import MessageEvent
        from gateway.run_voice import GatewayVoiceMixin
        from gateway.session import SessionSource
        from tools.registry import registry

        with patch.object(tts_tool, "_run_edge_tts") as edge:
            result = json.loads(registry.dispatch("text_to_speech", {"text": "Hello world"}))
            assert result["success"] is False and "disabled" in result["error"]
            assert tts_tool.check_tts_requirements() is False

            mixin = object.__new__(GatewayVoiceMixin)
            mixin._voice_mode = {"discord:chat": "all"}
            mixin._voice_key_for_source = lambda source: "discord:chat"
            mixin._delivery_adapter_for = lambda source: SimpleNamespace(
                _should_auto_tts_for_chat=lambda chat_id: True)
            event = MessageEvent(
                text="hi", source=SessionSource(platform=Platform.DISCORD, chat_id="chat", chat_type="dm"))
            assert mixin._should_send_voice_reply(event, "Hello world", []) is False
        edge.assert_not_called()
    finally:
        hc._LOAD_CONFIG_CACHE.clear()
