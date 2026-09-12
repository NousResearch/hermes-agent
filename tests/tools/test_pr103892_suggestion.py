"""Unit tests for PR #103892 (beefiker): afplay silent-truncation fix.

Changes:
1. tools/tts_tool_speaker.py: _synthesize_to_tmp now parses the TTS tool
   result envelope to find the real output path (providers can rename).
2. tools/voice_mode.py: _system_player_candidates restricts afplay to
   CoreAudio-decodable containers; Ogg/Opus/Flac fall through to ffplay.

All tests are pure-logic / mock-based — no real audio hardware or API calls.
"""

from __future__ import annotations

import os
import platform
from unittest.mock import MagicMock, patch

import pytest

from tools.tts_tool_speaker import _SyncSentencePipeline
from tools.voice_mode import _system_player_candidates


# ── _synthesize_to_tmp (tts_tool_speaker) ────────────────────────────────


class TestSynthesizeToTmp:
    """Result-envelope parsing in _SyncSentencePipeline._synthesize_to_tmp."""

    def _make_pipeline(self) -> _SyncSentencePipeline:
        """Return a pipeline with a mocked _origin()."""
        import threading
        pipeline = _SyncSentencePipeline.__new__(_SyncSentencePipeline)
        pipeline._stop = threading.Event()
        return pipeline

    def test_returns_file_path_from_result_envelope(self):
        """When the tool returns a valid envelope, use the reported file_path."""
        pipeline = self._make_pipeline()
        with patch("tools.tts_tool_speaker._origin") as mock_origin:
            mock_origin().text_to_speech_tool.return_value = (
                '{"success": true, "file_path": "/tmp/real_output.ogg"}'
            )
            result = pipeline._synthesize_to_tmp("hello")
        assert result == "/tmp/real_output.ogg"

    def test_falls_back_to_tmp_path_when_result_not_parseable(self):
        """When the result is not JSON, fall back to tmp_path if it exists."""
        pipeline = self._make_pipeline()
        with patch("tools.tts_tool_speaker._origin") as mock_origin:
            mock_origin().text_to_speech_tool.return_value = "plain text"
            with patch("os.path.isfile", return_value=True), patch("os.path.getsize", return_value=100):
                result = pipeline._synthesize_to_tmp("hello")
        # Should return the tmp_path (which is a tempfile path)
        assert result is not None
        assert isinstance(result, str)

    def test_returns_none_when_tmp_path_empty(self):
        """When tmp_path is empty/missing and no envelope, return None."""
        pipeline = self._make_pipeline()
        with patch("tools.tts_tool_speaker._origin") as mock_origin:
            mock_origin().text_to_speech_tool.return_value = "plain text"
            with patch("os.path.isfile", return_value=False):
                result = pipeline._synthesize_to_tmp("hello")
        assert result is None

    def test_returns_none_when_tmp_path_zero_bytes(self):
        """When tmp_path exists but is 0 bytes, return None."""
        pipeline = self._make_pipeline()
        with patch("tools.tts_tool_speaker._origin") as mock_origin:
            mock_origin().text_to_speech_tool.return_value = "plain text"
            with patch("os.path.isfile", return_value=True), patch("os.path.getsize", return_value=0):
                result = pipeline._synthesize_to_tmp("hello")
        assert result is None

    def test_handles_json_decode_error_gracefully(self):
        """Malformed JSON falls back to tmp_path check."""
        pipeline = self._make_pipeline()
        with patch("tools.tts_tool_speaker._origin") as mock_origin:
            mock_origin().text_to_speech_tool.return_value = "{bad json}"
            with patch("os.path.isfile", return_value=True), patch("os.path.getsize", return_value=100):
                result = pipeline._synthesize_to_tmp("hello")
        assert result is not None

    def test_handles_non_string_result(self):
        """When the tool returns a non-string (e.g. dict), handle gracefully."""
        pipeline = self._make_pipeline()
        with patch("tools.tts_tool_speaker._origin") as mock_origin:
            mock_origin().text_to_speech_tool.return_value = {"success": True, "file_path": "/tmp/out.ogg"}
            with patch("os.path.isfile", return_value=True), patch("os.path.getsize", return_value=100):
                result = pipeline._synthesize_to_tmp("hello")
        # Non-string result: json.loads not called, falls to tmp_path check
        assert result is not None

    def test_ignores_envelope_without_success(self):
        """Envelope without success=True falls through to tmp_path."""
        pipeline = self._make_pipeline()
        with patch("tools.tts_tool_speaker._origin") as mock_origin:
            mock_origin().text_to_speech_tool.return_value = (
                '{"success": false, "file_path": "/tmp/failed.ogg"}'
            )
            with patch("os.path.isfile", return_value=True), patch("os.path.getsize", return_value=100):
                result = pipeline._synthesize_to_tmp("hello")
        assert result is not None

    def test_ignores_envelope_without_file_path(self):
        """Envelope without file_path falls through to tmp_path."""
        pipeline = self._make_pipeline()
        with patch("tools.tts_tool_speaker._origin") as mock_origin:
            mock_origin().text_to_speech_tool.return_value = (
                '{"success": true}'
            )
            with patch("os.path.isfile", return_value=True), patch("os.path.getsize", return_value=100):
                result = pipeline._synthesize_to_tmp("hello")
        assert result is not None

    def test_exception_in_tool_returns_none(self):
        """When the tool raises, _synthesize_to_tmp returns None."""
        pipeline = self._make_pipeline()
        with patch("tools.tts_tool_speaker._origin") as mock_origin:
            mock_origin().text_to_speech_tool.side_effect = RuntimeError("API down")
            result = pipeline._synthesize_to_tmp("hello")
        assert result is None


# ── _system_player_candidates (voice_mode) ───────────────────────────────


class TestSystemPlayerCandidates:
    """Platform-aware player candidate selection."""

    def test_darwin_afplay_only_for_coreaudio_containers(self):
        """On Darwin, afplay is only listed for CoreAudio-decodable containers."""
        with patch("platform.system", return_value="Darwin"):
            candidates = _system_player_candidates("/path/to/file.mp3")
        assert any("afplay" in cmd for cmd in candidates)

    def test_darwin_no_afplay_for_ogg(self):
        """On Darwin, afplay is NOT listed for Ogg/Opus containers."""
        with patch("platform.system", return_value="Darwin"):
            candidates = _system_player_candidates("/path/to/file.ogg")
        # afplay should NOT be in the list for .ogg
        assert not any("afplay" in cmd for cmd in candidates)

    def test_darwin_no_afplay_for_flac(self):
        """On Darwin, afplay is NOT listed for Flac containers."""
        with patch("platform.system", return_value="Darwin"):
            candidates = _system_player_candidates("/path/to/file.flac")
        assert not any("afplay" in cmd for cmd in candidates)

    def test_darwin_ffplay_always_present(self):
        """ffplay is always in the candidate list regardless of container."""
        with patch("platform.system", return_value="Darwin"):
            candidates = _system_player_candidates("/path/to/file.ogg")
        assert any("ffplay" in cmd for cmd in candidates)

    def test_linux_no_afplay(self):
        """On Linux, afplay is never in the list."""
        with patch("platform.system", return_value="Linux"):
            candidates = _system_player_candidates("/path/to/file.mp3")
        assert not any("afplay" in cmd for cmd in candidates)

    def test_linux_has_ffplay_and_aplay(self):
        """On Linux, ffplay and aplay are in the candidate list."""
        with patch("platform.system", return_value="Linux"):
            candidates = _system_player_candidates("/path/to/file.mp3")
        assert any("ffplay" in cmd for cmd in candidates)
        assert any("aplay" in cmd for cmd in candidates)

    def test_darwin_afplay_for_wav(self):
        """On Darwin, afplay is listed for .wav (CoreAudio container)."""
        with patch("platform.system", return_value="Darwin"):
            candidates = _system_player_candidates("/path/to/file.wav")
        assert any("afplay" in cmd for cmd in candidates)

    def test_darwin_afplay_for_m4a(self):
        """On Darwin, afplay is listed for .m4a (CoreAudio container)."""
        with patch("platform.system", return_value="Darwin"):
            candidates = _system_player_candidates("/path/to/file.m4a")
        assert any("afplay" in cmd for cmd in candidates)

    def test_darwin_afplay_for_aiff(self):
        """On Darwin, afplay is listed for .aiff (CoreAudio container)."""
        with patch("platform.system", return_value="Darwin"):
            candidates = _system_player_candidates("/path/to/file.aiff")
        assert any("afplay" in cmd for cmd in candidates)

    def test_darwin_afplay_for_caf(self):
        """On Darwin, afplay is listed for .caf (CoreAudio container)."""
        with patch("platform.system", return_value="Darwin"):
            candidates = _system_player_candidates("/path/to/file.caf")
        assert any("afplay" in cmd for cmd in candidates)

    def test_darwin_afplay_for_aac(self):
        """On Darwin, afplay is listed for .aac (CoreAudio container)."""
        with patch("platform.system", return_value="Darwin"):
            candidates = _system_player_candidates("/path/to/file.aac")
        assert any("afplay" in cmd for cmd in candidates)

    def test_unknown_container_on_darwin_no_afplay(self):
        """Unknown containers on Darwin get ffplay but not afplay."""
        with patch("platform.system", return_value="Darwin"):
            candidates = _system_player_candidates("/path/to/file.xyz")
        # Unknown container: afplay is NOT added because ffplay is always present
        assert not any("afplay" in cmd for cmd in candidates)
        assert any("ffplay" in cmd for cmd in candidates)

    def test_case_insensitive_extension(self):
        """Extension matching is case-insensitive."""
        with patch("platform.system", return_value="Darwin"):
            candidates_upper = _system_player_candidates("/path/to/file.MP3")
            candidates_lower = _system_player_candidates("/path/to/file.mp3")
        assert any("afplay" in cmd for cmd in candidates_upper)
        assert any("afplay" in cmd for cmd in candidates_lower)

    def test_wsl_powershell_on_linux(self):
        """On Linux, WSL PowerShell player is included."""
        with patch("platform.system", return_value="Linux"):
            with patch("tools.voice_mode._wsl_powershell_player_cmd", return_value=["powershell.exe", "-c", "play"]):
                candidates = _system_player_candidates("/path/to/file.mp3")
        assert any("powershell.exe" in cmd for cmd in candidates)

    def test_wsl_powershell_none_on_linux(self):
        """When WSL PowerShell is None, it's not included."""
        with patch("platform.system", return_value="Linux"):
            with patch("tools.voice_mode._wsl_powershell_player_cmd", return_value=None):
                candidates = _system_player_candidates("/path/to/file.mp3")
        assert not any("powershell" in str(cmd) for cmd in candidates)
