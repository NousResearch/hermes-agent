"""Behavioral coverage for ElevenLabs options shared by sync and streaming TTS."""

from unittest.mock import MagicMock, patch

import pytest


def _setting(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def _setting_is_absent(value, name):
    return name not in value if isinstance(value, dict) else getattr(value, name) is None


def _client():
    client = MagicMock()
    client.text_to_speech.convert.return_value = iter([b"audio"])
    return client


def test_sync_forwards_valid_options_and_global_speed(tmp_path):
    from tools import tts_tool, tts_tool_providers

    client = _client()
    config = {
        "speed": 1.1,
        "elevenlabs": {
            "language_code": " en ",
            "voice_settings": {"stability": 0.5},
            "convert_options": {"seed": 42},
        },
    }
    with patch.object(tts_tool_providers, "_require_key", return_value="key"), patch.object(
        tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=client)
    ):
        tts_tool._generate_elevenlabs("hello", str(tmp_path / "out.mp3"), config)

    kwargs = client.text_to_speech.convert.call_args.kwargs
    assert kwargs["language_code"] == "en"
    assert kwargs["seed"] == 42
    assert _setting(kwargs["voice_settings"], "stability") == 0.5
    assert _setting(kwargs["voice_settings"], "speed") == 1.1


def test_streaming_uses_same_top_level_compatibility_options():
    from tools import tts_streaming, tts_tool

    client = _client()
    section = {"stability": 0.4, "speed": 0.9, "convert_options": {"previous_text": "before"}}
    with patch.object(tts_streaming, "_resolve_key", return_value="key"), patch.object(
        tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=client)
    ):
        list(tts_streaming.ElevenLabsStreamer({}, section).stream("hello"))

    kwargs = client.text_to_speech.convert.call_args.kwargs
    assert kwargs["previous_text"] == "before"
    assert _setting(kwargs["voice_settings"], "stability") == 0.4
    assert _setting(kwargs["voice_settings"], "speed") == 0.9


def test_nested_voice_settings_override_top_level_and_preserve_other_legacy_values():
    from tools.tts_elevenlabs_options import build_elevenlabs_convert_kwargs

    kwargs = build_elevenlabs_convert_kwargs(
        text="hello",
        voice_id="voice",
        model_id="eleven_multilingual_v2",
        output_format="pcm_24000",
        el_config={
            "stability": 0.2,
            "similarity_boost": 0.7,
            "voice_settings": {"stability": 0.9},
        },
    )

    assert _setting(kwargs["voice_settings"], "stability") == 0.9
    assert _setting(kwargs["voice_settings"], "similarity_boost") == 0.7


@pytest.mark.parametrize("path", ["sync", "stream"])
def test_v4_allows_language_stability_and_similarity_without_default_speed(path, tmp_path):
    from tools import tts_streaming, tts_tool, tts_tool_providers

    client = _client()
    section = {
        "model_id": "eleven_v4",
        "language_code": "fr",
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.8,
            "use_speaker_boost": False,
        },
    }
    config = {"speed": 1.0, "elevenlabs": section}
    with patch.object(tts_streaming, "_resolve_key", return_value="key"), patch.object(
        tts_tool_providers, "_require_key", return_value="key"
    ), patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=client)):
        if path == "sync":
            tts_tool._generate_elevenlabs("bonjour", str(tmp_path / "out.mp3"), config)
        else:
            list(tts_streaming.ElevenLabsStreamer(config, section).stream("bonjour"))

    kwargs = client.text_to_speech.convert.call_args.kwargs
    assert kwargs["language_code"] == "fr"
    assert _setting(kwargs["voice_settings"], "stability") == 0.6
    assert _setting(kwargs["voice_settings"], "similarity_boost") == 0.8
    assert _setting_is_absent(kwargs["voice_settings"], "speed")
    assert _setting_is_absent(kwargs["voice_settings"], "use_speaker_boost")


@pytest.mark.parametrize("path", ["sync", "stream"])
@pytest.mark.parametrize(
    "section,global_speed,message",
    [
        ({"voice_settings": {"style": 0.2}}, None, "style"),
        ({"voice_settings": {"speed": 1.1}}, None, "speed"),
        ({"voice_settings": {"use_speaker_boost": True}}, None, "use_speaker_boost"),
        ({}, 1.1, "speed"),
    ],
)
def test_v4_rejects_unsupported_nondefault_settings_before_sdk_call(
    path, section, global_speed, message, tmp_path,
):
    from tools import tts_streaming, tts_tool, tts_tool_providers

    client = _client()
    section = {"model_id": "eleven_v4_turbo", **section}
    config = {"elevenlabs": section}
    if global_speed is not None:
        config["speed"] = global_speed
    with patch.object(tts_streaming, "_resolve_key", return_value="key"), patch.object(
        tts_tool_providers, "_require_key", return_value="key"
    ), patch.object(
        tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=client)
    ), pytest.raises(ValueError, match=message):
        if path == "sync":
            tts_tool._generate_elevenlabs("hello", str(tmp_path / "out.mp3"), config)
        else:
            list(tts_streaming.ElevenLabsStreamer(config, section).stream("hello"))
    client.text_to_speech.convert.assert_not_called()


@pytest.mark.parametrize("path", ["sync", "stream"])
def test_default_config_preserves_managed_only_request_shape(path, tmp_path):
    from tools import tts_streaming, tts_tool, tts_tool_providers

    client = _client()
    with patch.object(tts_streaming, "_resolve_key", return_value="key"), patch.object(
        tts_tool_providers, "_require_key", return_value="key"
    ), patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=client)):
        if path == "sync":
            tts_tool._generate_elevenlabs("hello", str(tmp_path / "out.mp3"), {})
        else:
            list(tts_streaming.ElevenLabsStreamer({}, {}).stream("hello"))

    assert set(client.text_to_speech.convert.call_args.kwargs) == {
        "text", "voice_id", "model_id", "output_format",
    }


@pytest.mark.parametrize("path", ["sync", "stream"])
def test_base_and_websocket_urls_coexist_with_generation_options(path, tmp_path):
    from tools import tts_streaming, tts_tool, tts_tool_providers

    client = _client()
    factory = MagicMock(return_value=client)
    environment = object()
    section = {
        "base_url": "https://proxy.example",
        "wss_url": "wss://proxy.example",
        "language_code": "en",
    }
    config = {"elevenlabs": section}
    with patch.object(tts_streaming, "_resolve_key", return_value="key"), patch.object(
        tts_tool_providers, "_require_key", return_value="key"
    ), patch.object(
        tts_tool_providers, "_elevenlabs_environment_kwargs", return_value={"environment": environment}
    ) as environment_kwargs, patch.object(tts_tool, "_import_elevenlabs", return_value=factory):
        if path == "sync":
            tts_tool._generate_elevenlabs("hello", str(tmp_path / "out.mp3"), config)
        else:
            list(tts_streaming.ElevenLabsStreamer(config, section).stream("hello"))

    environment_kwargs.assert_called_once_with(section)
    factory.assert_called_once_with(api_key="key", environment=environment)
    assert client.text_to_speech.convert.call_args.kwargs["language_code"] == "en"


@pytest.mark.parametrize(
    "options,message",
    [({"model_id": "other"}, "Hermes-managed"), ({"unknown": True}, "Unknown ElevenLabs")],
)
def test_convert_options_are_validated_before_sdk_call(options, message, tmp_path):
    from tools import tts_tool, tts_tool_providers

    client = _client()
    config = {"elevenlabs": {"convert_options": options}}
    with patch.object(tts_tool_providers, "_require_key", return_value="key"), patch.object(
        tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=client)
    ), pytest.raises(ValueError, match=message):
        tts_tool._generate_elevenlabs("hello", str(tmp_path / "out.mp3"), config)
    client.text_to_speech.convert.assert_not_called()


def test_older_models_keep_speaker_boost():
    from tools.tts_elevenlabs_options import build_elevenlabs_convert_kwargs

    kwargs = build_elevenlabs_convert_kwargs(
        text="hello",
        voice_id="voice",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        el_config={"voice_settings": {"use_speaker_boost": True}},
    )

    assert _setting(kwargs["voice_settings"], "use_speaker_boost") is True
