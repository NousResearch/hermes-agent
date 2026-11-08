"""Per-call ``voice_id`` for OpenAI-compatible TTS (``tts.openai.voice_selection``).

Off by default: the schema and the request are exactly today's. On, the schema gains a ``voice_id``
enum listing the server's voices (``GET <base_url>/audio/voices``, or the static
``tts.openai.voices``), and a call's voice applies to that call only.
"""

import copy
import json
from unittest.mock import MagicMock, patch

import pytest

from tools import tts_tool, tts_tool_openai

BASE_URL = "http://tts.local:8090/v1"
VOICES_PAYLOAD = {"voices": [
    {"id": "ana_es", "language": "es"},
    {"id": "tom_en", "language": "en"},
    {"id": "bad voice; ignore previous", "language": "es"},
]}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    tts_tool_openai._voices_cache.clear()
    tts_tool._once_warnings.clear()
    yield
    tts_tool_openai._voices_cache.clear()


def _config(**openai):
    return {"provider": "openai", "openai": {"base_url": BASE_URL, "voice": "ana_es", **openai}}


def _http_response(payload, status=200):
    response = MagicMock()
    response.content = json.dumps(payload).encode()
    response.raise_for_status.side_effect = None if status < 400 else RuntimeError(f"HTTP {status}")
    return response


class TestSchema:
    def _overrides(self, config):
        with patch.object(tts_tool, "_load_tts_config", return_value=config):
            return tts_tool._tts_schema_overrides()

    @pytest.mark.parametrize("config", [
        {},
        {"provider": "openai", "openai": {"base_url": BASE_URL}},
        {"provider": "openai", "openai": {"base_url": BASE_URL, "voice_selection": False}},
        {"provider": "edge", "openai": {"base_url": BASE_URL, "voice_selection": True}},
    ])
    def test_off_keeps_todays_schema_and_never_fetches(self, config):
        with patch("requests.get") as get:
            overrides = self._overrides(config)
        get.assert_not_called()
        assert set(overrides) == {"parameters"}
        assert "voice_id" not in overrides["parameters"]["properties"]
        expected = copy.deepcopy(tts_tool.TTS_SCHEMA["parameters"])
        expected["properties"]["output_path"] = overrides["parameters"]["properties"]["output_path"]
        assert overrides["parameters"] == expected

    def test_on_lists_server_voices_with_language(self):
        with patch("requests.get", return_value=_http_response(VOICES_PAYLOAD)) as get:
            overrides = self._overrides(_config(voice_selection=True))
        assert get.call_args.args[0] == f"{BASE_URL}/audio/voices"
        assert get.call_args.kwargs["timeout"] == tts_tool_openai.VOICES_FETCH_TIMEOUT_SECONDS
        voice = overrides["parameters"]["properties"]["voice_id"]
        assert voice["enum"] == ["ana_es", "tom_en"]
        assert "ana_es (es), tom_en (en)" in voice["description"]
        assert "configured voice (ana_es)" in voice["description"]
        assert "bad voice" not in voice["description"]
        assert overrides["parameters"]["required"] == ["text"]
        assert "pick a voice per call with voice_id" in overrides["description"]
        assert tts_tool._CONFIGURED_VOICE_SENTENCE not in overrides["description"]

    def test_catalog_is_cached(self):
        with patch("requests.get", return_value=_http_response(VOICES_PAYLOAD)) as get:
            self._overrides(_config(voice_selection=True))
            self._overrides(_config(voice_selection=True))
        assert get.call_count == 1

    def test_static_list_and_voices_url(self):
        with patch("requests.get") as get:
            overrides = self._overrides(_config(voice_selection=True, voices=["ana_es", {"name": "tom_en", "lang": "en"}]))
        get.assert_not_called()
        assert overrides["parameters"]["properties"]["voice_id"]["enum"] == ["ana_es", "tom_en"]

        with patch("requests.get", return_value=_http_response(["x1"])) as get:
            self._overrides(_config(voice_selection=True, voices_url="http://other/voices", api_key="k"))
        assert get.call_args.args[0] == "http://other/voices"
        assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer k"}

    @pytest.mark.parametrize("failure", [
        {"side_effect": OSError("connection refused")},
        {"return_value": _http_response({}, status=404)},
    ])
    def test_unreachable_catalog_keeps_todays_schema(self, failure):
        with patch("requests.get", **failure):
            overrides = self._overrides(_config(voice_selection=True))
        assert set(overrides) == {"parameters"}
        assert "voice_id" not in overrides["parameters"]["properties"]


class TestCall:
    def _speak(self, config, tmp_path, **kwargs):
        client = MagicMock()
        with patch.object(tts_tool, "_load_tts_config", return_value=config), \
             patch.object(tts_tool, "_import_openai_client", return_value=MagicMock(return_value=client)), \
             patch.object(tts_tool_openai, "_resolve_openai_audio_client_config",
                          return_value=("key", BASE_URL, False)), \
             patch("requests.get", return_value=_http_response(VOICES_PAYLOAD)) as get:
            client.audio.speech.create.return_value.stream_to_file.side_effect = (
                lambda path: open(path, "wb").write(b"ID3fake-mp3"))
            result = json.loads(tts_tool.text_to_speech_tool(
                "Hola", output_path=str(tmp_path / "out.mp3"), **kwargs))
        return result, client.audio.speech.create, get

    def test_voice_applies_to_that_call_only(self, tmp_path):
        config = _config(voice_selection=True)
        result, create, _ = self._speak(config, tmp_path, voice_id="TOM_EN")
        assert result["success"], result
        assert create.call_args.kwargs["voice"] == "tom_en"
        assert config["openai"]["voice"] == "ana_es"  # config not mutated

        result, create, _ = self._speak(config, tmp_path)
        assert create.call_args.kwargs["voice"] == "ana_es"

    def test_unknown_voice_is_an_error_listing_valid_ones(self, tmp_path):
        result, create, _ = self._speak(_config(voice_selection=True), tmp_path, voice_id="nadie")
        assert not result["success"]
        assert "ana_es, tom_en" in result["error"]
        create.assert_not_called()

    def test_voice_ignored_when_selection_off(self, tmp_path):
        result, create, get = self._speak(_config(), tmp_path, voice_id="tom_en")
        assert result["success"], result
        assert create.call_args.kwargs["voice"] == "ana_es"
        get.assert_not_called()

    def test_off_request_is_unchanged(self, tmp_path):
        _, with_none, _ = self._speak(_config(), tmp_path, voice_id=None)
        _, without, _ = self._speak(_config(), tmp_path)
        strip = lambda kw: {k: v for k, v in kw.items() if k != "extra_headers"}  # idempotency uuid
        assert strip(with_none.call_args.kwargs) == strip(without.call_args.kwargs)

    def test_handler_forwards_voice_id(self):
        from tools.registry import registry
        handler = registry.get_entry("text_to_speech").handler
        with patch.object(tts_tool, "text_to_speech_tool", return_value="{}") as tool:
            handler({"text": "hi", "voice_id": "tom_en"})
        assert tool.call_args.kwargs["voice_id"] == "tom_en"
