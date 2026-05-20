"""Tests for TTS speed configuration across providers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY",
        "MINIMAX_API_KEY",
        "MINIMAX_GROUP_ID",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Edge TTS speed
# ---------------------------------------------------------------------------

class TestEdgeTtsSpeed:
    def _run(self, tts_config, tmp_path):
        mock_comm = MagicMock()
        mock_comm.save = AsyncMock()
        mock_edge = MagicMock()
        mock_edge.Communicate = MagicMock(return_value=mock_comm)

        with patch("tools.tts_tool._import_edge_tts", return_value=mock_edge):
            from tools.tts_tool import _generate_edge_tts
            asyncio.run(_generate_edge_tts("Hello", str(tmp_path / "out.mp3"), tts_config))
        return mock_edge.Communicate

    def test_default_no_rate_kwarg(self, tmp_path):
        """No speed config => no rate kwarg passed to Communicate."""
        comm_cls = self._run({}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert "rate" not in kwargs

    def test_global_speed_applied(self, tmp_path):
        """Global tts.speed used as fallback."""
        comm_cls = self._run({"speed": 1.5}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert kwargs["rate"] == "+50%"

    def test_provider_speed_overrides_global(self, tmp_path):
        """tts.edge.speed takes precedence over tts.speed."""
        comm_cls = self._run({"speed": 1.5, "edge": {"speed": 2.0}}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert kwargs["rate"] == "+100%"

    def test_speed_below_one(self, tmp_path):
        """Speed < 1.0 produces a negative rate string."""
        comm_cls = self._run({"speed": 0.5}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert kwargs["rate"] == "-50%"

    def test_speed_exactly_one_no_rate(self, tmp_path):
        """Explicit speed=1.0 should not pass rate kwarg."""
        comm_cls = self._run({"speed": 1.0}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert "rate" not in kwargs


# ---------------------------------------------------------------------------
# OpenAI TTS speed
# ---------------------------------------------------------------------------

class TestOpenaiTtsSpeed:
    def _run(self, tts_config, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_client = MagicMock()
        mock_client.audio.speech.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool._resolve_openai_audio_client_config",
                   return_value=("test-key", None)):
            from tools.tts_tool import _generate_openai_tts
            _generate_openai_tts("Hello", str(tmp_path / "out.mp3"), tts_config)
        return mock_client.audio.speech.create

    def test_default_no_speed_kwarg(self, tmp_path, monkeypatch):
        """No speed config => no speed kwarg in create call."""
        create = self._run({}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert "speed" not in kwargs

    def test_global_speed_applied(self, tmp_path, monkeypatch):
        """Global tts.speed used as fallback."""
        create = self._run({"speed": 1.5}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert kwargs["speed"] == 1.5

    def test_provider_speed_overrides_global(self, tmp_path, monkeypatch):
        """tts.openai.speed takes precedence over tts.speed."""
        create = self._run({"speed": 1.5, "openai": {"speed": 2.0}}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert kwargs["speed"] == 2.0

    def test_speed_clamped_low(self, tmp_path, monkeypatch):
        """Speed below 0.25 is clamped to 0.25."""
        create = self._run({"speed": 0.1}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert kwargs["speed"] == 0.25

    def test_speed_clamped_high(self, tmp_path, monkeypatch):
        """Speed above 4.0 is clamped to 4.0."""
        create = self._run({"speed": 10.0}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert kwargs["speed"] == 4.0


# ---------------------------------------------------------------------------
# MiniMax TTS (t2a_v2 endpoint: nested voice_setting/audio_setting,
# JSON response with hex-encoded audio.  Falls back to the legacy
# text_to_speech endpoint shape when the base_url points at it.)
# ---------------------------------------------------------------------------


def _hex_response(payload_audio: bytes = b"\x00\x01\x02\x03"):
    """Build a mock response shaped like a successful t2a_v2 reply."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "data": {"audio": payload_audio.hex(), "status": 2},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    return mock_response


class TestMinimaxTtsT2aV2:
    """Default path: base_url contains 't2a_v2'."""

    def _run(self, tts_config, tmp_path, monkeypatch, response=None):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        resp = response if response is not None else _hex_response()
        with patch("requests.post", return_value=resp) as mock_post:
            from tools.tts_tool import _generate_minimax_tts
            output = _generate_minimax_tts("Hello", str(tmp_path / "out.mp3"), tts_config)
        return mock_post, output

    def test_nested_payload(self, tmp_path, monkeypatch):
        """Default endpoint uses nested voice_setting / audio_setting."""
        mock_post, _ = self._run({}, tmp_path, monkeypatch)
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "speech-02-hd"
        assert payload["text"] == "Hello"
        assert "voice_setting" in payload
        assert payload["voice_setting"]["voice_id"] == "English_expressive_narrator"
        assert "audio_setting" in payload
        assert payload["audio_setting"]["format"] == "mp3"
        # Don't send flat top-level voice_id alongside nested voice_setting.
        assert "voice_id" not in payload

    def test_decodes_hex_audio(self, tmp_path, monkeypatch):
        """t2a_v2 hex-encoded audio is decoded and written verbatim."""
        _, output = self._run({}, tmp_path, monkeypatch)
        with open(output, "rb") as f:
            assert f.read() == b"\x00\x01\x02\x03"

    def test_default_url_is_t2a_v2(self, tmp_path, monkeypatch):
        """Default base URL points at the live t2a_v2 endpoint."""
        mock_post, _ = self._run({}, tmp_path, monkeypatch)
        url = mock_post.call_args[0][0]
        assert "t2a_v2" in url
        assert "api.minimax.io" in url

    def test_group_id_from_config(self, tmp_path, monkeypatch):
        """group_id from config attaches as ?GroupId=<id>."""
        mock_post, _ = self._run({"minimax": {"group_id": "G123"}}, tmp_path, monkeypatch)
        url = mock_post.call_args[0][0]
        assert "GroupId=G123" in url

    def test_group_id_from_env(self, tmp_path, monkeypatch):
        """MINIMAX_GROUP_ID env var attaches as ?GroupId=<id>."""
        monkeypatch.setenv("MINIMAX_GROUP_ID", "G456")
        mock_post, _ = self._run({}, tmp_path, monkeypatch)
        url = mock_post.call_args[0][0]
        assert "GroupId=G456" in url

    def test_group_id_already_in_url_left_alone(self, tmp_path, monkeypatch):
        """If user already set GroupId in base_url, don't double-append it."""
        cfg = {"minimax": {
            "base_url": "https://api.minimax.io/v1/t2a_v2?GroupId=PRESET",
            "group_id": "IGNORED",
        }}
        mock_post, _ = self._run(cfg, tmp_path, monkeypatch)
        url = mock_post.call_args[0][0]
        assert url.count("GroupId=") == 1
        assert "GroupId=PRESET" in url

    def test_api_error_raises(self, tmp_path, monkeypatch):
        """Non-zero base_resp.status_code surfaces as RuntimeError."""
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "application/json"}
        resp.json.return_value = {
            "data": {"audio": "", "status": 1},
            "base_resp": {"status_code": 2013, "status_msg": "invalid voice"},
        }
        with pytest.raises(RuntimeError, match="2013"):
            self._run({}, tmp_path, monkeypatch, response=resp)


class TestMinimaxTtsLegacyTextToSpeech:
    """Legacy path: caller pins base_url to the old text_to_speech endpoint."""

    LEGACY_URL = "https://api.minimax.chat/v1/text_to_speech"

    def _run(self, tts_config, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        cfg = dict(tts_config)
        cfg.setdefault("minimax", {})["base_url"] = self.LEGACY_URL
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "audio/mpeg"}
        mock_response.content = b"\x00\x01\x02\x03"
        with patch("requests.post", return_value=mock_response) as mock_post:
            from tools.tts_tool import _generate_minimax_tts
            output = _generate_minimax_tts("Hello", str(tmp_path / "out.mp3"), cfg)
        return mock_post, output

    def test_flat_payload(self, tmp_path, monkeypatch):
        """Legacy endpoint keeps the flat {model, text, voice_id} shape."""
        mock_post, _ = self._run({}, tmp_path, monkeypatch)
        payload = mock_post.call_args[1]["json"]
        assert "voice_id" in payload
        assert "voice_setting" not in payload
        assert "audio_setting" not in payload

    def test_writes_raw_audio(self, tmp_path, monkeypatch):
        """Legacy endpoint returns raw bytes written directly to file."""
        _, output = self._run({}, tmp_path, monkeypatch)
        with open(output, "rb") as f:
            assert f.read() == b"\x00\x01\x02\x03"
