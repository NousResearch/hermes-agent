"""Model-selected ElevenLabs voices: ``text_to_speech`` gains ``voice_id`` and ``find_voices`` (a
paginated voice search) only while ElevenLabs is the configured provider."""

import json
from unittest.mock import patch

import httpx
import pytest

from tools import tts_tool

_EL_CONFIG = {"provider": "elevenlabs", "elevenlabs": {"voice_id": "configuredVoice001"}}


@pytest.mark.parametrize("config, exposed", [
    (_EL_CONFIG, True), ({"provider": "edge"}, False), ({"provider": "openai"}, False)])
def test_voice_selection_surface_follows_provider(config, exposed):
    with patch("tools.tts_tool._load_tts_config", return_value=config):
        schema = tts_tool._tts_schema_overrides()
        searched = json.loads(tts_tool.text_to_speech_tool("", find_voices={"query": "orc"}))
    props = schema["parameters"]["properties"]
    assert ("voice_id" in props) is exposed and ("find_voices" in props) is exposed
    assert schema["parameters"]["required"] == ([] if exposed else ["text"])
    assert ("find_voices" in schema.get("description", "")) is exposed
    if not exposed:  # a schema cached before a provider switch must not search
        assert not searched["success"] and "elevenlabs" in searched["error"]


def test_voice_id_reaches_elevenlabs_and_is_ignored_elsewhere(tmp_path):
    seen = []

    class _FakeElevenLabs:  # the SDK client: records the voice the real generator sends
        def __init__(self, **_kwargs):
            self.text_to_speech = self

        def convert(self, *, voice_id, **_kwargs):
            seen.append(voice_id)
            return [b"ID3\x03\x00\x00\x00\x00\x00\x00"]

    with patch("tools.tts_tool._import_elevenlabs", return_value=_FakeElevenLabs), \
         patch("tools.tts_tool._resolve_provider_key", return_value="el-test-key"), \
         patch("tools.tts_tool._load_tts_config", return_value=_EL_CONFIG):
        ok = json.loads(tts_tool.text_to_speech_tool(
            "Hail, traveler.", output_path=str(tmp_path / "a.mp3"), voice_id="pickedVoice0042"))
        default = json.loads(tts_tool.text_to_speech_tool(
            "Hail, traveler.", output_path=str(tmp_path / "b.mp3")))
        bad = json.loads(tts_tool.text_to_speech_tool(
            "Hail.", output_path=str(tmp_path / "c.mp3"), voice_id="../v1/user"))
    assert ok["success"] and default["success"] and not bad["success"]
    assert seen == ["pickedVoice0042", "configuredVoice001"]
    assert _EL_CONFIG["elevenlabs"]["voice_id"] == "configuredVoice001"  # cached config untouched

    config, error = tts_tool._apply_voice_override({"provider": "edge"}, "edge", "pickedVoice0042")
    assert error is None and "elevenlabs" not in config


@pytest.mark.parametrize("find, path, expected_params, body, next_page", [
    ({"query": "gravelly old", "gender": "Male", "use_case": "characters_animation", "page": "3",
      "page_size": 500}, "/v1/shared-voices",
     {"search": "gravelly old", "gender": "male", "use_cases": ["characters_animation"],
      "sort": "usage_character_count_1y", "page": 3, "page_size": 100},
     {"voices": [{"voice_id": "lib1", "name": "Gruff", "gender": "male", "age": "old",
                  "accent": "scottish", "descriptive": "gruff", "use_case": "characters_animation",
                  "category": "professional", "rate": 2, "description": "x" * 400}],
      "has_more": True, "total_count": 812}, "4"),
    ({"query": "orc", "source": "account", "page": "tok-1"}, "/v2/voices",
     {"search": "orc", "next_page_token": "tok-1", "page_size": 20},
     {"voices": [{"voice_id": "acct1", "name": "Bram", "category": "premade",
                  "labels": {"gender": "male", "age": "old", "accent": "scottish",
                             "use case": "characters"},
                  "preview_url": "https://example.test/bram.mp3"}],
      "has_more": True, "next_page_token": "tok-2"}, "tok-2"),
    ({}, "/v1/shared-voices",  # an empty object browses the library with the defaults
     {"sort": "usage_character_count_1y", "page": 0, "page_size": 20},
     {"voices": [{"voice_id": "lib2", "name": "Bram", "gender": "male", "accent": "scottish"}],
      "has_more": False}, None),
])
def test_find_voices_pages_both_sources(find, path, expected_params, body, next_page):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, dict(params or {}), headers))
        return httpx.Response(200, json=body)

    with patch("tools.tts_tool._load_tts_config", return_value=_EL_CONFIG), \
         patch("tools.tts_tool._resolve_provider_key", return_value="el-test-key"), \
         patch("tools.tts_elevenlabs_voices.httpx.get", side_effect=fake_get):
        from tools.registry import registry  # through the registered handler, as the model calls it
        entry = registry.get_entry("text_to_speech")
        assert entry is not None
        result = json.loads(entry.handler({"find_voices": find}))
        bad_page = json.loads(tts_tool.text_to_speech_tool("", find_voices={"page": "tok-2"}))
    url, params, headers = calls[0]
    assert url == "https://api.elevenlabs.io" + path and headers["xi-api-key"] == "el-test-key"
    assert {k: params[k] for k in expected_params} == expected_params
    row = result["voices"][0]
    assert result["success"] and result["next_page"] == next_page
    assert row["gender"] == "male" and row["accent"] == "scottish"
    assert result["configured_voice_id"] == "configuredVoice001"
    if "total_count" in body:
        assert result["total"] == 812 and len(row["description"]) == 240 and row["credit_rate"] == 2
    assert not bad_page["success"] and len(calls) == 1  # a cursor token is not a library page


def test_provider_switch_reshapes_the_schema_at_once(monkeypatch):
    """The ElevenLabs fields come from config at every listing: a switch right after an OpenAI
    listing must add them, and switching back must drop them."""
    import os

    import model_tools
    from hermes_constants import get_hermes_home

    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config_path = get_hermes_home() / "config.yaml"

    def tts_props(provider: str, stamp: int) -> set:
        config_path.write_text(f"tts:\n  provider: {provider}\n")
        os.utime(config_path, (stamp, stamp))
        defs = model_tools.get_tool_definitions(enabled_toolsets=["tts"], quiet_mode=True)
        (tts,) = [d for d in defs if d["function"]["name"] == "text_to_speech"]
        return set(tts["function"]["parameters"]["properties"])

    assert "find_voices" not in tts_props("openai", 1_700_000_000)
    assert {"find_voices", "voice_id"} <= tts_props("elevenlabs", 1_700_000_100)
    assert "find_voices" not in tts_props("openai", 1_700_000_200)
