"""Residual coverage for the Gemini TTS provider cluster and the xAI auto
speech tags in ``tools/tts_tool.py``.

This is the coverage-completion file for the Gemini provider (the biggest
remaining uncovered block) plus the still-uncovered ``_apply_xai_auto_speech_tags``
branches. It targets:

* the pure/config-driven Gemini helpers — ``_resolve_gemini_persona_prompt_path``,
  ``_read_gemini_persona_prompt``, ``_gemini_model_supports_audio_tags``,
  ``_gemini_audio_tags_enabled``, ``_clean_gemini_audio_tag_rewrite``,
  ``_extract_auxiliary_message_content``, ``_rewrite_gemini_tts_audio_tags``,
  ``_compose_gemini_tts_prompt``;
* ``_wrap_pcm_as_wav`` header packing;
* ``_generate_gemini_tts`` driven entirely offline (``requests.post`` is faked,
  no SDK import, no network);
* the ``_xai_bool_config`` wrapper and the empty-input / already-tagged /
  auxiliary-rewriter fallback branches of ``_apply_xai_auto_speech_tags``.

No network or real credential lookup ever happens.
"""

import base64
import json
import struct
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools import tts_tool


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_BASE_URL",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_keys(monkeypatch):
    """Deterministic ``_resolve_provider_key`` backed by a dict."""
    keys: dict = {}

    def _resolve(env_var, provider_id):
        return keys.get(env_var, "")

    monkeypatch.setattr("tools.tts_tool._resolve_provider_key", _resolve)
    return keys


@pytest.fixture(autouse=True)
def fake_env_value(monkeypatch):
    """Pin ``get_env_value`` so GEMINI_BASE_URL can't leak from config/env."""
    monkeypatch.setattr(
        "tools.tts_tool.get_env_value", lambda name, default=None: ""
    )


# ---------------------------------------------------------------------------
# _xai_bool_config / _apply_xai_auto_speech_tags residual branches
# ---------------------------------------------------------------------------


class TestXaiBoolConfig:
    def test_boolean_passthrough(self):
        assert tts_tool._xai_bool_config(True) is True
        assert tts_tool._xai_bool_config(False, default=True) is False

    def test_string_coercion_and_unknown_default(self):
        assert tts_tool._xai_bool_config("yes") is True
        assert tts_tool._xai_bool_config("disabled") is False
        # Unrecognised strings fall back to the provided default.
        assert tts_tool._xai_bool_config("maybe", default=True) is True


class TestApplyXaiAutoSpeechTagsResidual:
    def test_empty_input_returns_verbatim(self):
        assert tts_tool._apply_xai_auto_speech_tags("   \n  ") == "   \n  "

    def test_explicit_tags_skips_auxiliary_rewrite(self):
        with patch("agent.auxiliary_client.call_llm") as mock_call:
            result = tts_tool._apply_xai_auto_speech_tags(
                "Bonjour. [pause] <whisper>Déjà balisé.</whisper>"
            )
        mock_call.assert_not_called()
        assert result == "Bonjour. [pause] <whisper>Déjà balisé.</whisper>"

    def test_auxiliary_failure_falls_back_to_local_pause_text(self):
        with patch(
            "agent.auxiliary_client.call_llm", side_effect=RuntimeError("boom")
        ):
            result = tts_tool._apply_xai_auto_speech_tags(
                "Bonjour Monsieur Talbot. Ceci est un test de réponse vocale."
            )
        assert result == (
            "Bonjour Monsieur Talbot. [pause] Ceci est un test de réponse vocale."
        )

    def test_auxiliary_success_returns_rewritten_and_strips_fence(self):
        fenced = "```text\n[warmly] Bonjour. [soft laugh]\n```"
        response = SimpleNamespace(
            choices=[SimpleNamespace(message={"content": fenced})]
        )
        with patch("agent.auxiliary_client.call_llm", return_value=response) as mock_call:
            result = tts_tool._apply_xai_auto_speech_tags(
                "Bonjour Monsieur Talbot. Ceci est un test de réponse vocale."
            )
        assert result == "[warmly] Bonjour. [soft laugh]"
        mock_call.assert_called_once()


# ---------------------------------------------------------------------------
# Gemini persona prompt resolution / reading
# ---------------------------------------------------------------------------


class TestResolveGeminiPersonaPromptPath:
    def test_missing_or_non_str_returns_none(self):
        assert tts_tool._resolve_gemini_persona_prompt_path({}) is None
        assert (
            tts_tool._resolve_gemini_persona_prompt_path(
                {"persona_prompt_file": 123}
            )
            is None
        )
        assert (
            tts_tool._resolve_gemini_persona_prompt_path(
                {"persona_prompt_file": "   "}
            )
            is None
        )

    def test_absolute_path_kept(self, tmp_path):
        p = tmp_path / "persona.txt"
        assert (
            tts_tool._resolve_gemini_persona_prompt_path(
                {"persona_prompt_file": str(p)}
            )
            == p
        )

    def test_relative_path_uses_hermes_home_when_available(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
        result = tts_tool._resolve_gemini_persona_prompt_path(
            {"persona_prompt_file": "persona.txt"}
        )
        assert result == tmp_path / "persona.txt"

    def test_relative_path_falls_back_to_cwd_when_resolver_fails(
        self, tmp_path, monkeypatch
    ):
        # get_hermes_home returns None -> calling it raises TypeError -> fallback.
        monkeypatch.setattr("hermes_constants.get_hermes_home", None)
        result = tts_tool._resolve_gemini_persona_prompt_path(
            {"persona_prompt_file": "persona.txt"}
        )
        assert result.is_absolute()
        assert result.name == "persona.txt"


class TestReadGeminiPersonaPrompt:
    def test_missing_path_returns_empty(self):
        assert tts_tool._read_gemini_persona_prompt({}) == ""

    def test_reads_file_and_strips(self, tmp_path):
        p = tmp_path / "persona.txt"
        p.write_text("  Speak like a narrator.  ", encoding="utf-8")
        assert (
            tts_tool._read_gemini_persona_prompt(
                {"persona_prompt_file": str(p)}
            )
            == "Speak like a narrator."
        )

    def test_unreadable_file_returns_empty_and_warns(self, tmp_path, caplog):
        missing = tmp_path / "nope.txt"
        with caplog.at_level("WARNING"):
            assert (
                tts_tool._read_gemini_persona_prompt(
                    {"persona_prompt_file": str(missing)}
                )
                == ""
            )
        assert "persona prompt file unavailable" in caplog.text


# ---------------------------------------------------------------------------
# Gemini audio-tag gating / cleaning / extraction
# ---------------------------------------------------------------------------


class TestGeminiModelSupportsAudioTags:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gemini-3.1-flash-tts-preview", True),
            ("models/gemini-3.1-flash-tts-preview", True),
            ("GEMINI-3.1-FLASH-TTS", True),
            ("gemini-3.1-flash", False),
            ("gemini-2.5-flash-preview-tts", False),
            ("", False),
            (None, False),
        ],
    )
    def test_support(self, model, expected):
        assert (
            tts_tool._gemini_model_supports_audio_tags(model) is expected
        )


class TestGeminiAudioTagsEnabled:
    def test_disabled_by_default(self):
        assert (
            tts_tool._gemini_audio_tags_enabled({}, "gemini-3.1-flash-tts")
            is False
        )

    def test_dict_enabled_true_and_supported_model(self):
        cfg = {"audio_tags": {"enabled": True}}
        assert (
            tts_tool._gemini_audio_tags_enabled(cfg, "gemini-3.1-flash-tts")
            is True
        )

    def test_dict_enabled_false(self):
        cfg = {"audio_tags": {"enabled": False}}
        assert (
            tts_tool._gemini_audio_tags_enabled(cfg, "gemini-3.1-flash-tts")
            is False
        )

    def test_bool_enabled_true(self):
        assert (
            tts_tool._gemini_audio_tags_enabled(
                {"audio_tags": True}, "gemini-3.1-flash-tts"
            )
            is True
        )

    def test_model_gate_warns_and_disables(self, caplog):
        with caplog.at_level("WARNING"):
            result = tts_tool._gemini_audio_tags_enabled(
                {"audio_tags": True}, "gemini-2.5-flash-tts"
            )
        assert result is False
        assert "not known to support" in caplog.text

    def test_none_config_returns_false(self):
        assert (
            tts_tool._gemini_audio_tags_enabled(
                {"audio_tags": None}, "gemini-3.1-flash-tts"
            )
            is False
        )


class TestCleanGeminiAudioTagRewrite:
    def test_strips_and_passthrough(self):
        assert tts_tool._clean_gemini_audio_tag_rewrite("  hello  ") == "hello"
        assert tts_tool._clean_gemini_audio_tag_rewrite(None) == ""

    def test_fenced_output_stripped(self):
        assert (
            tts_tool._clean_gemini_audio_tag_rewrite(
                "```\n[whispers] Hi.\n```"
            )
            == "[whispers] Hi."
        )


class TestExtractAuxiliaryMessageContent:
    def test_dict_message_content(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message={"content": "hi"})]
        )
        assert tts_tool._extract_auxiliary_message_content(response) == "hi"

    def test_attr_message_content(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hey"))]
        )
        assert tts_tool._extract_auxiliary_message_content(response) == "hey"

    def test_none_content_returns_empty(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message={"content": None})]
        )
        assert tts_tool._extract_auxiliary_message_content(response) == ""

    def test_scope_errors_return_empty(self):
        assert tts_tool._extract_auxiliary_message_content(None) == ""
        assert (
            tts_tool._extract_auxiliary_message_content(
                SimpleNamespace(choices=[])
            )
            == ""
        )


# ---------------------------------------------------------------------------
# Gemini audio-tag rewrite + prompt composition
# ---------------------------------------------------------------------------


class TestRewriteGeminiTtsAudioTags:
    def test_empty_text_returns_verbatim(self):
        assert tts_tool._rewrite_gemini_tts_audio_tags("   ") == "   "

    def test_happy_path_returns_rewritten_content(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message={"content": "[whispers] Hello."})]
        )
        with patch(
            "agent.auxiliary_client.call_llm", return_value=response
        ) as mock_call:
            result = tts_tool._rewrite_gemini_tts_audio_tags(
                "Hello.", persona_prompt="narrator"
            )
        assert result == "[whispers] Hello."
        kwargs = mock_call.call_args.kwargs
        assert kwargs["task"] == tts_tool.GEMINI_AUDIO_TAG_REWRITE_TASK
        assert kwargs["temperature"] == 0.7
        # persona_prompt flows into the user message.
        assert "narrator" in kwargs["messages"][1]["content"]

    def test_fenced_output_cleaned(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message={"content": "```text\n[whispers] Hi.\n```"}
                )
            ]
        )
        with patch("agent.auxiliary_client.call_llm", return_value=response):
            result = tts_tool._rewrite_gemini_tts_audio_tags("Hi.")
        assert result == "[whispers] Hi."

    def test_blank_rewritten_output_falls_back_to_original(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message={"content": "   "})]
        )
        with patch("agent.auxiliary_client.call_llm", return_value=response):
            result = tts_tool._rewrite_gemini_tts_audio_tags("Original text.")
        assert result == "Original text."

    def test_call_llm_error_falls_back_to_original_warns(self, caplog):
        with patch(
            "agent.auxiliary_client.call_llm", side_effect=RuntimeError("boom")
        ):
            with caplog.at_level("WARNING"):
                result = tts_tool._rewrite_gemini_tts_audio_tags(
                    "Original text."
                )
        assert result == "Original text."
        assert "audio tag rewrite failed" in caplog.text


class TestComposeGeminiTtsPrompt:
    def test_no_persona_returns_transcript(self):
        assert (
            tts_tool._compose_gemini_tts_prompt(
                "  Hi there.  ", {}, persona_prompt=None
            )
            == "Hi there."
        )

    def test_persona_with_double_brace_transcript_placeholder(self):
        persona = "Speak like {{ transcript }}."
        result = tts_tool._compose_gemini_tts_prompt(
            "HELLO", {}, persona_prompt=persona
        )
        assert "HELLO" in result
        assert "{{ transcript }}" not in result
        assert result.startswith("Synthesize speech from the TRANSCRIPT only")

    def test_persona_with_single_brace_transcript_placeholder(self):
        persona = "Speak like { transcript } for this."
        result = tts_tool._compose_gemini_tts_prompt(
            "HELLO", {}, persona_prompt=persona
        )
        assert "HELLO" in result
        assert "{ transcript }" not in result

    def test_persona_no_placeholder_appends_transcript_section(self):
        persona = "Be warm and soothing."
        result = tts_tool._compose_gemini_tts_prompt(
            "Hi!", {}, persona_prompt=persona
        )
        assert result.startswith("Synthesize speech from the TRANSCRIPT only")
        assert "Be warm and soothing." in result
        assert "#### TRANSCRIPT\nHi!" in result

    def test_reads_persona_from_config_when_none(self, tmp_path):
        p = tmp_path / "persona.txt"
        p.write_text("Read like {{ transcription }}.", encoding="utf-8")
        result = tts_tool._compose_gemini_tts_prompt(
            "GO", {"persona_prompt_file": str(p)}, persona_prompt=None
        )
        assert "GO" in result
        assert "Read like {{ transcription }}." in result


# ---------------------------------------------------------------------------
# _wrap_pcm_as_wav
# ---------------------------------------------------------------------------


class TestWrapPcmAsWav:
    def test_default_header_fields_and_payload(self):
        pcm = b"\x00\x01\x02\x03" * 10
        wav = tts_tool._wrap_pcm_as_wav(pcm)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        # audio format == 1 (PCM)
        assert struct.unpack("<H", wav[20:22])[0] == 1
        # channels
        assert struct.unpack("<H", wav[22:24])[0] == 1
        # sample rate
        assert struct.unpack("<I", wav[24:28])[0] == 24000
        # byte rate = rate * channels * sample_width
        assert struct.unpack("<I", wav[28:32])[0] == 24000 * 1 * 2
        # bits per sample
        assert struct.unpack("<H", wav[34:36])[0] == 16
        assert wav[36:40] == b"data"
        assert wav[44:] == pcm

    def test_custom_channels_rate_width(self):
        pcm = b"\x01\x02" * 8
        wav = tts_tool._wrap_pcm_as_wav(
            pcm, sample_rate=16000, channels=2, sample_width=1
        )
        assert struct.unpack("<H", wav[22:24])[0] == 2
        assert struct.unpack("<I", wav[24:28])[0] == 16000
        assert struct.unpack("<I", wav[28:32])[0] == 16000 * 2 * 1
        assert struct.unpack("<H", wav[34:36])[0] == 8
        assert wav[44:] == pcm


# ---------------------------------------------------------------------------
# _generate_gemini_tts
# ---------------------------------------------------------------------------


def _gemini_payload(audio_b64):
    return {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"data": audio_b64}}]}}
        ]
    }


def _ok_bytes_response(audio_b64, status=200):
    """A requests-shaped response whose JSON rides in ``content`` bytes.

    This exercises the streaming/bytes branch of ``_read_tts_response_json``.
    """
    return SimpleNamespace(
        status_code=status,
        content=json.dumps(_gemini_payload(audio_b64)).encode("utf-8"),
    )


def _ok_mock_response(audio_b64):
    """A MagicMock response that only exposes ``.json()``.

    This exercises the ``.json()`` fallback branch of
    ``_read_tts_response_json`` for unit-test doubles.
    """
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _gemini_payload(audio_b64)
    return resp


def _error_bytes_response(status, body):
    return SimpleNamespace(status_code=status, content=body)


@pytest.fixture
def gemini_http(monkeypatch):
    """Capture the requests.post call and let each test set ``_response``."""
    captured = {}

    def fake_post(url, params=None, headers=None, json=None, timeout=None,
                  stream=False, **kw):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        captured["stream"] = stream
        return captured["_response"]

    monkeypatch.setattr("requests.post", fake_post)
    return captured


class TestGenerateGeminiTts:
    @pytest.fixture(autouse=True)
    def _use_key(self, fake_keys):
        fake_keys["GEMINI_API_KEY"] = "test-gemini-key"

    def test_missing_key_raises_value_error(self, gemini_http, fake_keys,
                                             tmp_path):
        fake_keys.pop("GEMINI_API_KEY", None)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            tts_tool._generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})

    def test_google_api_key_fallback(self, gemini_http, fake_keys, tmp_path):
        fake_keys.pop("GEMINI_API_KEY", None)
        pcm = b"\x00" * 4800
        gemini_http["_response"] = _ok_bytes_response(base64.b64encode(pcm).decode())
        fake_keys["GOOGLE_API_KEY"] = "google-key"
        tts_tool._generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})
        assert gemini_http["params"]["key"] == "google-key"

    def test_wav_happy_path_defaults(self, gemini_http, fake_keys, tmp_path):
        pcm = b"\x00\x01\x02\x03" * 1200
        gemini_http["_response"] = _ok_bytes_response(base64.b64encode(pcm).decode())
        out = tmp_path / "out.wav"
        result = tts_tool._generate_gemini_tts(
            "Hello world", str(out), {}
        )
        assert result == str(out)
        data = out.read_bytes()
        assert data[:4] == b"RIFF"
        assert data[8:12] == b"WAVE"
        assert data[44:] == pcm
        # default google base_url -> X-Goog client header, key in params
        assert gemini_http["params"]["key"] == "test-gemini-key"
        assert "X-Goog-Api-Client" in gemini_http["headers"]
        assert gemini_http["stream"] is True
        # model / voice defaults and prompt text in the payload
        payload = gemini_http["json"]
        assert payload["generationConfig"]["responseModalities"] == ["AUDIO"]
        voice_cfg = payload["generationConfig"]["speechConfig"]["voiceConfig"]
        assert voice_cfg["prebuiltVoiceConfig"]["voiceName"] == tts_tool.DEFAULT_GEMINI_TTS_VOICE
        # endpoint carries DEFAULT_GEMINI_TTS_MODEL
        assert tts_tool.DEFAULT_GEMINI_TTS_MODEL in gemini_http["url"]
        assert payload["contents"][0]["parts"][0]["text"] == "Hello world"

    def test_json_fallback_response(self, gemini_http, fake_keys, tmp_path):
        pcm = b"\xff" * 100
        gemini_http["_response"] = _ok_mock_response(base64.b64encode(pcm).decode())
        out = tmp_path / "out.wav"
        tts_tool._generate_gemini_tts("Hi", str(out), {})
        # _read_tts_response_json used .json() fallback -> payload decoded fine
        assert out.read_bytes()[44:] == pcm

    def test_custom_model_voice_base_url(self, gemini_http, fake_keys, tmp_path):
        pcm = b"\x00" * 4800
        gemini_http["_response"] = _ok_bytes_response(base64.b64encode(pcm).decode())
        config = {
            "gemini": {
                "model": "models/gemini-3.1-flash-tts-preview",
                "voice": "Puck",
                "base_url": "https://proxy.example/v1beta",
            }
        }
        tts_tool._generate_gemini_tts("Hi", str(tmp_path / "out.wav"), config)
        assert "models/gemini-3.1-flash-tts-preview" in gemini_http["url"]
        fake_payload = gemini_http["json"]
        voice_cfg = fake_payload["generationConfig"]["speechConfig"]["voiceConfig"]
        assert voice_cfg["prebuiltVoiceConfig"]["voiceName"] == "Puck"
        # non-google base_url -> no X-Goog client header
        assert "X-Goog-Api-Client" not in gemini_http["headers"]

    def test_hermes_cli_import_failure_uses_zero_version(
        self, gemini_http, fake_keys, tmp_path, monkeypatch
    ):
        pcm = b"\x00" * 4800
        gemini_http["_response"] = _ok_bytes_response(base64.b64encode(pcm).decode())
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "hermes_cli":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        tts_tool._generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})
        assert gemini_http["headers"]["X-Goog-Api-Client"] == "hermes-agent/0.0.0"

    def test_audio_tag_rewrite_composes_rewritten_text(
        self, gemini_http, fake_keys, tmp_path
    ):
        pcm = b"\x00" * 4800
        gemini_http["_response"] = _ok_bytes_response(base64.b64encode(pcm).decode())
        config = {
            "gemini": {
                "model": "gemini-3.1-flash-tts",
                "audio_tags": True,
            }
        }
        fenced = "```\n[whispers] Hi there.\n```"
        response = SimpleNamespace(
            choices=[SimpleNamespace(message={"content": fenced})]
        )
        with patch("agent.auxiliary_client.call_llm", return_value=response):
            tts_tool._generate_gemini_tts(
                "Hi there.", str(tmp_path / "out.wav"), config
            )
        prompt_text = gemini_http["json"]["contents"][0]["parts"][0]["text"]
        # rewritten (fence-stripped) text flows into the prompt
        assert "[whispers] Hi there." in prompt_text

    def test_audio_tags_gated_off_for_unsupported_model(
        self, gemini_http, fake_keys, tmp_path, caplog
    ):
        pcm = b"\x00" * 4800
        gemini_http["_response"] = _ok_bytes_response(base64.b64encode(pcm).decode())
        config = {
            "gemini": {
                "model": "gemini-2.5-flash-preview-tts",
                "audio_tags": True,
            }
        }
        with patch("agent.auxiliary_client.call_llm") as mock_call:
            with caplog.at_level("WARNING"):
                tts_tool._generate_gemini_tts(
                    "Hi there.", str(tmp_path / "out.wav"), config
                )
        mock_call.assert_not_called()
        assert "not known to support" in caplog.text
        prompt_text = gemini_http["json"]["contents"][0]["parts"][0]["text"]
        assert prompt_text == "Hi there."

    def test_prompt_exceeds_max_length_raises(
        self, gemini_http, fake_keys, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "tools.tts_tool._resolve_max_text_length", lambda provider, cfg: 5
        )
        with pytest.raises(ValueError, match="exceeds the provider request limit"):
            tts_tool._generate_gemini_tts(
                "This is definitely longer than five characters",
                str(tmp_path / "out.wav"),
                {},
            )

    def test_http_error_bytes_body(self, gemini_http, fake_keys, tmp_path):
        gemini_http["_response"] = _error_bytes_response(
            400, b'{"error":{"message":"bad request"}}'
        )
        with pytest.raises(RuntimeError, match="HTTP 400.*bad request"):
            tts_tool._generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})

    def test_http_error_json_fallback(self, gemini_http, fake_keys, tmp_path):
        resp = MagicMock()
        resp.status_code = 429
        resp.json.return_value = {"error": {"message": "rate limited"}}
        gemini_http["_response"] = resp
        with pytest.raises(RuntimeError, match="HTTP 429.*rate limited"):
            tts_tool._generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})

    def test_http_error_unparseable_body(self, gemini_http, fake_keys, tmp_path):
        gemini_http["_response"] = _error_bytes_response(500, b"not-json")
        with pytest.raises(RuntimeError, match="HTTP 500.*not-json"):
            tts_tool._generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})

    def test_http_error_no_body_reaches_else_branch(
        self, gemini_http, fake_keys, tmp_path
    ):
        # Empty body, no iter_content, no .json() -> the error dict stays {}.
        gemini_http["_response"] = _error_bytes_response(500, b"")
        with pytest.raises(RuntimeError, match="HTTP 500"):
            tts_tool._generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})

    def test_no_audio_part_raises(self, gemini_http, fake_keys, tmp_path):
        payload = {"candidates": [{"content": {"parts": [{"foo": "bar"}]}}]}
        gemini_http["_response"] = SimpleNamespace(
            status_code=200, content=json.dumps(payload).encode("utf-8")
        )
        with pytest.raises(RuntimeError, match="no audio data"):
            tts_tool._generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})

    def test_malformed_response_raises(self, gemini_http, fake_keys, tmp_path):
        # Missing "parts" key -> KeyError inside the try -> "malformed"
        gemini_http["_response"] = _error_bytes_response(
            200, b'{"candidates":[{"content":{}}]}'
        )
        with pytest.raises(RuntimeError, match="malformed"):
            tts_tool._generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})

    def test_empty_audio_data_raises(self, gemini_http, fake_keys, tmp_path):
        gemini_http["_response"] = _ok_bytes_response("")
        with pytest.raises(RuntimeError, match="empty audio data"):
            tts_tool._generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})

    def test_mp3_output_ffmpeg_missing_copies_wav(
        self, gemini_http, fake_keys, tmp_path, monkeypatch
    ):
        pcm = b"\x00" * 4800
        gemini_http["_response"] = _ok_bytes_response(base64.b64encode(pcm).decode())
        monkeypatch.setattr("shutil.which", lambda name: None)
        # os.remove raise is swallowed by the finally-block's except OSError.
        monkeypatch.setattr(
            "os.remove",
            lambda p: (_ for _ in ()).throw(OSError("in use")),
        )
        out = tmp_path / "out.mp3"
        result = tts_tool._generate_gemini_tts(
            "Hi", str(out), {"gemini": {"base_url": "https://proxy.example/v1beta"}}
        )
        assert result == str(out)
        # ffmpeg absent -> raw WAV copied with the (misleading) .mp3 name
        assert out.read_bytes()[:4] == b"RIFF"
        assert out.read_bytes()[44:] == pcm

    def test_mp3_output_ffmpeg_present_success(
        self, gemini_http, fake_keys, tmp_path, monkeypatch
    ):
        pcm = b"\x00" * 4800
        gemini_http["_response"] = _ok_bytes_response(base64.b64encode(pcm).decode())
        monkeypatch.setattr("shutil.which", lambda name: "ffmpeg")
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0, stderr=b"")

        monkeypatch.setattr("subprocess.run", fake_run)
        out = tmp_path / "out.mp3"
        result = tts_tool._generate_gemini_tts(
            "Hi", str(out), {"gemini": {"base_url": "https://proxy.example/v1beta"}}
        )
        assert result == str(out)
        # basic mp3 ffmpeg cmd (no libopus)
        cmd = captured["cmd"]
        assert "ffmpeg" in cmd[0]
        assert "-acodec" not in cmd

    def test_ogg_output_uses_libopus(self, gemini_http, fake_keys, tmp_path,
                                     monkeypatch):
        pcm = b"\x00" * 4800
        gemini_http["_response"] = _ok_bytes_response(base64.b64encode(pcm).decode())
        monkeypatch.setattr("shutil.which", lambda name: "ffmpeg")
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0, stderr=b"")

        monkeypatch.setattr("subprocess.run", fake_run)
        tts_tool._generate_gemini_tts(
            "Hi", str(tmp_path / "out.ogg"),
            {"gemini": {"base_url": "https://proxy.example/v1beta"}},
        )
        cmd = captured["cmd"]
        assert "-acodec" in cmd
        assert "libopus" in cmd

    def test_mp3_output_ffmpeg_failure_raises(
        self, gemini_http, fake_keys, tmp_path, monkeypatch
    ):
        pcm = b"\x00" * 4800
        gemini_http["_response"] = _ok_bytes_response(base64.b64encode(pcm).decode())
        monkeypatch.setattr("shutil.which", lambda name: "ffmpeg")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: SimpleNamespace(returncode=1, stderr=b"boom"),
        )
        with pytest.raises(RuntimeError, match="ffmpeg conversion failed"):
            tts_tool._generate_gemini_tts(
                "Hi", str(tmp_path / "out.mp3"),
                {"gemini": {"base_url": "https://proxy.example/v1beta"}},
            )
