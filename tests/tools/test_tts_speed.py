"""Tests for TTS speed configuration across providers."""

import asyncio
from pathlib import Path
from subprocess import CompletedProcess
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
             patch("tools.tts_tool_openai._resolve_openai_audio_client_config",
                   return_value=("test-key", None, False)):
            from tools.tts_tool import _generate_openai_tts
            _generate_openai_tts("Hello", str(tmp_path / "out.mp3"), tts_config)
        return mock_client.audio.speech.create

    def test_default_no_speed_kwarg(self, tmp_path, monkeypatch):
        """No speed config => no speed kwarg in create call."""
        create = self._run({}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert "speed" not in kwargs


    def test_speed_clamped_high(self, tmp_path, monkeypatch):
        """Speed above 4.0 is clamped to 4.0."""
        create = self._run({"speed": 10.0}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert kwargs["speed"] == 4.0

    def test_local_mode_synthesizes_at_normal_speed_then_processes(self, tmp_path, monkeypatch):
        """Local mode omits endpoint speed and post-processes the completed file."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        output = tmp_path / "out.mp3"
        mock_response = MagicMock()
        mock_response.stream_to_file.side_effect = lambda path: Path(path).write_bytes(b"audio")
        mock_client = MagicMock()
        mock_client.audio.speech.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool_openai._resolve_openai_audio_client_config",
                   return_value=("test-key", None, False)), \
             patch("tools.tts_tool._apply_local_tempo", return_value=str(output)) as apply_tempo:
            from tools.tts_tool import _generate_openai_tts
            result = _generate_openai_tts(
                "Hello", str(output), {"openai": {"speed": 1.5, "speed_mode": "local"}})

        assert result == str(output)
        assert "speed" not in mock_client.audio.speech.create.call_args.kwargs
        apply_tempo.assert_called_once_with(str(output), 1.5)


class TestLocalOpenaiTempo:
    @pytest.mark.parametrize(
        "speed,expected",
        [
            (0.25, "atempo=0.5,atempo=0.5"),
            (0.75, "atempo=0.75"),
            (1.5, "atempo=1.5"),
            (2.5, "atempo=2,atempo=1.25"),
            (4.0, "atempo=2,atempo=2"),
        ],
    )
    def test_filter_stages_stay_in_portable_range(self, speed, expected):
        from tools.tts_tool_delivery import _build_atempo_filter
        assert _build_atempo_filter(speed) == expected

    def test_atomic_replace_and_failure_cleanup(self, tmp_path):
        from tools.tts_tool_delivery import _apply_local_tempo

        output = tmp_path / "out.mp3"
        output.write_bytes(b"original")

        with patch("tools.tts_tool_delivery.shutil.which", return_value=None), \
             pytest.raises(RuntimeError, match="requires ffmpeg"):
            _apply_local_tempo(str(output), 1.5)
        assert output.read_bytes() == b"original"

        def successful_run(_ffmpeg, args, **_kwargs):
            Path(args[-1]).write_bytes(b"processed")
            return CompletedProcess(args, 0, b"", b"")

        with patch("tools.tts_tool_delivery.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("tools.tts_tool_delivery._ffmpeg_run", side_effect=successful_run):
            assert _apply_local_tempo(str(output), 1.5) == str(output)
        assert output.read_bytes() == b"processed"
        assert not list(tmp_path.glob(".*.atempo.mp3"))

        output.write_bytes(b"original")

        def failed_run(_ffmpeg, args, **_kwargs):
            Path(args[-1]).write_bytes(b"partial")
            return CompletedProcess(args, 1, b"", b"encoder failed")

        with patch("tools.tts_tool_delivery.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("tools.tts_tool_delivery._ffmpeg_run", side_effect=failed_run), \
             pytest.raises(RuntimeError, match="encoder failed"):
            _apply_local_tempo(str(output), 1.5)
        assert output.read_bytes() == b"original"
        assert not list(tmp_path.glob(".*.atempo.mp3"))


# ---------------------------------------------------------------------------
# OpenAI TTS language (lang_code for OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------

class TestOpenaiTtsLangCode:
    def _run(self, tts_config, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_client = MagicMock()
        mock_client.audio.speech.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool_openai._resolve_openai_audio_client_config",
                   return_value=("test-key", None, False)):
            from tools.tts_tool import _generate_openai_tts
            _generate_openai_tts("Hola", str(tmp_path / "out.mp3"), tts_config)
        return mock_client.audio.speech.create

    def test_default_no_extra_body(self, tmp_path, monkeypatch):
        """No language config => no extra_body kwarg in create call."""
        create = self._run({}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert "extra_body" not in kwargs


    def test_language_coexists_with_speed(self, tmp_path, monkeypatch):
        """language and speed are forwarded independently."""
        create = self._run({"openai": {"language": "es", "speed": 2.0}},
                           tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert kwargs["extra_body"] == {"lang_code": "es"}
        assert kwargs["speed"] == 2.0


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


# ---------------------------------------------------------------------------
# Tool-level speed parameter (text_to_speech_tool speed injection)
# ---------------------------------------------------------------------------

class TestToolLevelSpeed:
    """Verify that the speed parameter on text_to_speech_tool injects into config."""

    def test_speed_injected_into_config(self, tmp_path, monkeypatch):
        """When speed is passed to the tool, it overrides config speed."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_client = MagicMock()
        mock_client.audio.speech.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool_openai._resolve_openai_audio_client_config",
                   return_value=("test-key", None, False)), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "openai", "openai": {}}), \
             patch("tools.tts_tool._get_provider", return_value="openai"), \
             patch("tools.tts_tool._resolve_command_provider_config", return_value=None), \
             patch("tools.tts_tool._resolve_max_text_length", return_value=4096), \
             patch("tools.tts_tool._generate_openai_tts") as mock_gen, \
             patch("gateway.session_context.get_session_env", return_value=""):
            from tools.tts_tool import text_to_speech_tool
            text_to_speech_tool("Hello", str(tmp_path / "out.mp3"), speed=0.7)

        # Verify the tts_config passed to the generator has speed=0.7
        call_args = mock_gen.call_args
        config_passed = call_args[0][2]  # (text, output_path, tts_config)
        assert config_passed["speed"] == 0.7

    def test_speed_clamped_range(self, tmp_path, monkeypatch):
        """Speed values outside 0.25-4.0 are clamped."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "openai", "openai": {}}), \
             patch("tools.tts_tool._get_provider", return_value="openai"), \
             patch("tools.tts_tool._resolve_command_provider_config", return_value=None), \
             patch("tools.tts_tool._resolve_max_text_length", return_value=4096), \
             patch("tools.tts_tool._generate_openai_tts") as mock_gen, \
             patch("gateway.session_context.get_session_env", return_value=""):
            from tools.tts_tool import text_to_speech_tool
            text_to_speech_tool("Hello", str(tmp_path / "out.mp3"), speed=10.0)

        config_passed = mock_gen.call_args[0][2]
        assert config_passed["speed"] == 4.0

    def test_no_speed_preserves_config(self, tmp_path, monkeypatch):
        """When speed is None, config is not mutated."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        original_config = {"provider": "openai", "openai": {}, "speed": 1.5}

        with patch("tools.tts_tool._load_tts_config", return_value=original_config), \
             patch("tools.tts_tool._get_provider", return_value="openai"), \
             patch("tools.tts_tool._resolve_command_provider_config", return_value=None), \
             patch("tools.tts_tool._resolve_max_text_length", return_value=4096), \
             patch("tools.tts_tool._generate_openai_tts") as mock_gen, \
             patch("gateway.session_context.get_session_env", return_value=""):
            from tools.tts_tool import text_to_speech_tool
            text_to_speech_tool("Hello", str(tmp_path / "out.mp3"), speed=None)

        config_passed = mock_gen.call_args[0][2]
        assert config_passed.get("speed") == 1.5  # original config preserved
        assert original_config.get("speed") == 1.5  # original not mutated
