"""Cloud-provider TTS generator coverage: DeepInfra, xAI, MiniMax, and the
uncovered Mistral branches.

Each block targets the *deep* branches of ``_generate_<provider>_tts`` that the
other focused test files leave uncovered:

* ``test_tts_deepinfra.py``            — no-model fallback + requirements
* ``test_tts_xai_speech_tags.py``      — speech-tag rewriting + payload shape
* ``test_tts_minimax_region.py``       — region/credential/endpoint selection
* ``test_tts_mistral.py``              — missing-key/success/format/error

This file adds the remaining generator branches: credential-missing raises,
model resolution fallbacks, xAI speed / streaming-latency / output-format /
text-normalization knobs, MiniMax t2a_v2 + flat text_to_speech error paths,
GroupId attachment and the mm-config coalesce, and Mistral's base_url /
ValueError re-raise paths.  No network — every upstream call is faked.
"""

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _sanitize_env(monkeypatch):
    """Drop every provider key so the tests control creds deterministically."""
    for key in (
        "DEEPINFRA_API_KEY",
        "XAI_API_KEY",
        "MINIMAX_API_KEY",
        "MINIMAX_CN_API_KEY",
        "MISTRAL_API_KEY",
        "XAI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_keys(monkeypatch):
    """Controlled provider-key resolver shared across providers."""
    keys: dict[str, str] = {}

    def _resolve(env_var, provider_id):
        return keys.get(env_var, "")

    monkeypatch.setattr("tools.tts_tool._resolve_provider_key", _resolve)
    return keys


def _fake_post_response(*, content, content_type="application/json", status_error=None):
    """Build a requests-shaped stub consumed by the tts_tool helpers."""

    def raise_for_status():
        if status_error is not None:
            raise status_error

    return SimpleNamespace(
        content=content,
        headers={"Content-Type": content_type},
        raise_for_status=raise_for_status,
    )


# ===========================================================================
# DeepInfra — _generate_deepinfra_tts
# ===========================================================================
class TestGenerateDeepinfraTts:
    def test_missing_key_raises(self, fake_keys, tmp_path):
        from tools.tts_tool import _generate_deepinfra_tts

        with pytest.raises(ValueError, match="DEEPINFRA_API_KEY"):
            _generate_deepinfra_tts("hello", str(tmp_path / "out.mp3"), {})

    def test_happy_path_delegates_to_openai_handler(self, fake_keys, tmp_path):
        from tools.tts_tool import _generate_deepinfra_tts

        fake_keys["DEEPINFRA_API_KEY"] = "test-deepinfra-key"
        output_path = str(tmp_path / "out.mp3")
        config = {
            "deepinfra": {
                "model": "mock-tts-model",
                "base_url": "https://api.deepinfra.com/v1",
                "voice": "ara",
                "speed": 2.0,
            }
        }

        with patch("tools.tts_tool._generate_openai_tts", return_value=output_path) as mock_openai:
            result = _generate_deepinfra_tts("hello", output_path, config)

        assert result == output_path
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args
        # position args: text, output_path, tts_config
        assert call_kwargs[0][0] == "hello"
        assert call_kwargs[0][1] == output_path
        assert call_kwargs[0][2] == config
        assert call_kwargs[1]["api_key"] == "test-deepinfra-key"
        assert call_kwargs[1]["base_url"] == "https://api.deepinfra.com/v1"
        assert call_kwargs[1]["model"] == "mock-tts-model"
        assert call_kwargs[1]["voice"] == "ara"
        assert call_kwargs[1]["speed"] == 2.0

    def test_model_resolves_from_catalog_when_unset(self, fake_keys, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_deepinfra_tts

        fake_keys["DEEPINFRA_API_KEY"] = "test-deepinfra-key"
        output_path = str(tmp_path / "out.mp3")

        monkeypatch.setattr(
            "hermes_cli.models.deepinfra_model_ids",
            lambda tag, force_refresh=False: ["first-tts-model", "second-tts-model"],
        )

        with patch("tools.tts_tool._generate_openai_tts", return_value=output_path) as mock_openai:
            _generate_deepinfra_tts("hello", output_path, {})

        assert mock_openai.call_args[1]["model"] == "first-tts-model"
        assert mock_openai.call_args[1]["voice"] == "default"
        # speed falls back to global 1.0 when neither provider nor global set it
        assert mock_openai.call_args[1]["speed"] == 1.0


# ===========================================================================
# xAI — _generate_xai_tts (remaining knobs)
# ===========================================================================
class TestGenerateXaiTts:
    @pytest.fixture(autouse=True)
    def _pin_credentials_and_http(self, monkeypatch):
        """Provide an API-key credential and capture requests.post payloads."""
        captured = {}

        monkeypatch.setattr(
            "tools.xai_http.resolve_xai_http_credentials",
            lambda **kwargs: {
                "provider": "xai-api",
                "api_key": "test-xai-key",
                "base_url": "https://api.x.ai/v1",
            },
        )

        def fake_post(url, headers, json, timeout, stream=False):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _fake_post_response(content=b"fake-audio")

        monkeypatch.setattr("requests.post", fake_post)
        self.captured = captured

    def _call(self, tmp_path, text="hello", config=None, extension=".mp3"):
        from tools.tts_tool import _generate_xai_tts

        return _generate_xai_tts(
            text,
            str(tmp_path / f"out{extension}"),
            config or {"xai": {"auto_speech_tags": False}},
        )

    def test_no_credentials_raises(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_xai_tts

        monkeypatch.setattr(
            "tools.xai_http.resolve_xai_http_credentials", lambda **kwargs: {}
        )
        with pytest.raises(ValueError, match="No xAI credentials"):
            _generate_xai_tts("hello", str(tmp_path / "out.mp3"), {})

    def test_speed_clamped_to_max_and_attached(self, tmp_path):
        self._call(
            tmp_path,
            config={"xai": {"auto_speech_tags": False, "speed": 5.0}},
        )
        assert self.captured["json"]["speed"] == 1.5

    def test_invalid_speed_becomes_none_and_omitted(self, tmp_path):
        self._call(
            tmp_path,
            config={"xai": {"auto_speech_tags": False, "speed": "not-a-number"}},
        )
        assert "speed" not in self.captured["json"]

    def test_invalid_optimize_streaming_latency_becomes_none(self, tmp_path):
        self._call(
            tmp_path,
            config={"xai": {"auto_speech_tags": False, "optimize_streaming_latency": "abc"}},
        )
        assert "optimize_streaming_latency" not in self.captured["json"]

    def test_optimize_streaming_latency_clamped_and_attached(self, tmp_path):
        self._call(
            tmp_path,
            config={"xai": {"auto_speech_tags": False, "optimize_streaming_latency": 5}},
        )
        assert self.captured["json"]["optimize_streaming_latency"] == 2

    def test_wav_output_forces_output_format(self, tmp_path):
        self._call(tmp_path, extension=".wav")
        assert self.captured["json"]["output_format"]["codec"] == "wav"
        assert self.captured["json"]["output_format"]["sample_rate"] == 24000
        # bit_rate is only added for the mp3 codec
        assert "bit_rate" not in self.captured["json"]["output_format"]

    def test_sample_rate_and_bit_rate_override_builds_output_format(self, tmp_path):
        self._call(
            tmp_path,
            config={
                "xai": {
                    "auto_speech_tags": False,
                    "sample_rate": 16000,
                    "bit_rate": 96000,
                }
            },
        )
        output_format = self.captured["json"]["output_format"]
        assert output_format["codec"] == "mp3"
        assert output_format["sample_rate"] == 16000
        assert output_format["bit_rate"] == 96000

    def test_speed_below_min_clamped_and_attached(self, tmp_path):
        self._call(
            tmp_path,
            config={"xai": {"auto_speech_tags": False, "speed": 0.2}},
        )
        assert self.captured["json"]["speed"] == 0.7

    def test_text_normalization_sent_when_true(self, tmp_path):
        self._call(
            tmp_path,
            config={"xai": {"auto_speech_tags": False, "text_normalization": True}},
        )
        assert self.captured["json"]["text_normalization"] is True

    def test_http_error_propagates_and_hides_key(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_xai_tts

        class HttpError(Exception):
            pass

        monkeypatch.setattr(
            "tools.xai_http.resolve_xai_http_credentials",
            lambda **kwargs: {
                "provider": "xai-api",
                "api_key": "secret-xyz",
                "base_url": "https://api.x.ai/v1",
            },
        )
        monkeypatch.setattr(
            "requests.post",
            lambda *a, **kw: _fake_post_response(
                content=b"", status_error=HttpError("boom")
            ),
        )

        with pytest.raises(HttpError) as exc_info:
            _generate_xai_tts("hello", str(tmp_path / "out.mp3"), {"xai": {}})
        assert "secret-xyz" not in str(exc_info.value)


# ===========================================================================
# MiniMax — _generate_minimax_tts (error paths + config coalesce)
# ===========================================================================
class TestGenerateMinimaxTts:
    @pytest.fixture(autouse=True)
    def _pin_credentials_and_http(self, fake_keys, monkeypatch):
        fake_keys["MINIMAX_API_KEY"] = "test-minimax-key"
        captured = {}

        def fake_post(url, json, headers, timeout, stream=False):
            captured["url"] = url
            captured["json"] = json
            return _fake_post_response(
                content=b'{"base_resp":{"status_code":0},"data":{"audio":"68656c6c6f"}}',
                content_type="application/json",
            )

        monkeypatch.setattr("requests.post", fake_post)
        self.captured = captured

    def _call(self, tmp_path, config=None, text="hello", base_path="out.mp3"):
        from tools.tts_tool import _generate_minimax_tts

        return _generate_minimax_tts(text, str(tmp_path / base_path), config or {})

    def test_t2a_v2_happy_path_writes_hex_audio(self, tmp_path):
        out = tmp_path / "out.mp3"
        self._call(tmp_path)
        # "hello" in hex
        assert out.read_bytes() == b"hello"

    def test_t2a_v2_empty_audio_raises(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setattr(
            "requests.post",
            lambda *a, **kw: _fake_post_response(
                content=b'{"base_resp":{"status_code":0},"data":{"audio":""}}',
                content_type="application/json",
            ),
        )
        with pytest.raises(RuntimeError, match="empty audio"):
            self._call(tmp_path)

    def test_group_id_appended_to_endpoint(self, tmp_path):
        # Default runtime endpoint is t2a_v2 with no GroupId → query is appended.
        self._call(tmp_path, config={"minimax": {"group_id": "grp-123"}})
        assert self.captured["url"] == (
            "https://api.minimax.io/v1/t2a_v2?GroupId=grp-123"
        )

    def test_none_minimax_config_coalesces_to_empty(self, tmp_path):
        # tts.minimax: null in YAML yields None — must coalesce, not crash.
        self._call(tmp_path, config={"minimax": None})
        assert (tmp_path / "out.mp3").read_bytes() == b"hello"

    def test_flat_endpoint_json_api_error_sanitized(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setattr(
            "requests.post",
            lambda *a, **kw: _fake_post_response(
                content=b'{"base_resp":{"status_code":1001,"status_msg":"rate limited"}}',
                content_type="application/json",
            ),
        )
        with pytest.raises(RuntimeError, match="code 1001"):
            _generate_minimax_tts(
                "hello",
                str(tmp_path / "out.mp3"),
                {"minimax": {"base_url": "https://api.minimax.io/v1/text_to_speech"}},
            )

    def test_flat_endpoint_non_json_body_raises_unexpected_content_type(
        self, tmp_path, monkeypatch
    ):
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setattr(
            "requests.post",
            lambda *a, **kw: _fake_post_response(
                content=b"not-json-at-all",
                content_type="text/plain",
            ),
        )
        with pytest.raises(RuntimeError, match="unexpected Content-Type"):
            _generate_minimax_tts(
                "hello",
                str(tmp_path / "out.mp3"),
                {"minimax": {"base_url": "https://api.minimax.io/v1/text_to_speech"}},
            )

    def test_flat_endpoint_json_ok_but_no_audio_raises(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_minimax_tts

        monkeypatch.setattr(
            "requests.post",
            lambda *a, **kw: _fake_post_response(
                content=b'{"base_resp":{"status_code":0}}',
                content_type="application/json",
            ),
        )
        with pytest.raises(RuntimeError, match="no audio data"):
            _generate_minimax_tts(
                "hello",
                str(tmp_path / "out.mp3"),
                {"minimax": {"base_url": "https://api.minimax.io/v1/text_to_speech"}},
            )


# ===========================================================================
# Mistral — uncovered branches (base_url / ValueError re-raise)
# ===========================================================================
@pytest.fixture
def mock_mistral_module():
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_mistral_cls = MagicMock(return_value=mock_client)
    fake_module = MagicMock()
    fake_module.Mistral = mock_mistral_cls
    with patch.dict("sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}):
        yield SimpleNamespace(client=mock_client, cls=mock_mistral_cls)


class TestGenerateMistralTtsBranches:
    def test_base_url_maps_to_server_url(self, fake_keys, mock_mistral_module, tmp_path):
        from tools.tts_tool import _generate_mistral_tts

        fake_keys["MISTRAL_API_KEY"] = "test-mistral-key"
        mock_mistral_module.client.audio.speech.complete.return_value = MagicMock(
            audio_data=base64.b64encode(b"audio").decode()
        )

        config = {"mistral": {"base_url": "https://proxy.example/v1"}}
        result = _generate_mistral_tts("Hi", str(tmp_path / "out.mp3"), config)

        assert result == str(tmp_path / "out.mp3")
        client_kwargs = mock_mistral_module.cls.call_args[1]
        assert client_kwargs["api_key"] == "test-mistral-key"
        assert client_kwargs["server_url"] == "https://proxy.example/v1"

    def test_base64_decode_value_error_is_rerraised(self, fake_keys, mock_mistral_module, tmp_path):
        from tools.tts_tool import _generate_mistral_tts

        fake_keys["MISTRAL_API_KEY"] = "test-mistral-key"
        # Invalid base64 raises binascii.Error, a ValueError subclass — the
        # function re-raises it verbatim (never wraps it as a generic error).
        mock_mistral_module.client.audio.speech.complete.return_value = MagicMock(
            audio_data="a"
        )

        with pytest.raises(ValueError):
            _generate_mistral_tts("Hi", str(tmp_path / "out.mp3"), {})
