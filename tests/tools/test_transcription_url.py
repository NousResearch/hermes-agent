"""Tests for remote URL transcription — _probe_audio_url, _download_audio,
_split_audio, _transcribe_chunks, and transcribe_url.

All external dependencies (HTTP, ffmpeg, Groq) are mocked.
"""

import json
import os
import struct
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_wav(tmp_path):
    """Create a minimal valid WAV file (1 second of silence at 16kHz)."""
    wav_path = tmp_path / "test.wav"
    n_frames = 16000
    silence = struct.pack(f"<{n_frames}h", *([0] * n_frames))

    import wave
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(silence)

    return str(wav_path)


@pytest.fixture
def sample_mp3(tmp_path):
    """Create a fake MP3 file for split tests."""
    mp3_path = tmp_path / "test.mp3"
    mp3_path.write_bytes(b"x" * 4096)
    return str(mp3_path)


# ============================================================================
# _probe_audio_url
# ============================================================================

def _mock_urlopen_success(
    final_url="https://cdn.example.com/audio.mp3",
    content_type="audio/mpeg",
    content_length=5000000,
):
    """Create a mock urllib.request.urlopen response for a successful HEAD."""
    resp = MagicMock()
    resp.url = final_url
    resp.headers = {"Content-Type": content_type, "Content-Length": str(content_length)}
    # Context manager support: __enter__ returns self
    resp.__enter__.return_value = resp
    return resp


class TestProbeAudioUrl:
    def test_successful_probe(self):
        """Happy path — HEAD request returns redirect-resolved URL with metadata."""
        mock_resp = _mock_urlopen_success()
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            from tools.transcription_tools import _probe_audio_url
            result = _probe_audio_url("https://example.com/audio.mp3")

        assert result["success"] is True
        assert result["url"] == "https://cdn.example.com/audio.mp3"
        assert result["content_type"] == "audio/mpeg"
        assert result["content_length"] == 5000000
        # Verify HEAD request
        req = mock_urlopen.call_args[0][0]
        assert req.method == "HEAD"

    def test_probe_no_content_length(self):
        """HEAD response without Content-Length."""
        mock_resp = MagicMock()
        mock_resp.url = "https://example.com/audio.ogg"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.headers = {"Content-Type": "audio/ogg"}  # no Content-Length
        with patch("urllib.request.urlopen", return_value=mock_resp):
            from tools.transcription_tools import _probe_audio_url
            result = _probe_audio_url("https://example.com/audio.ogg")

        assert result["success"] is True
        assert result["content_length"] is None

    def test_probe_http_404(self):
        """HTTP 404 returns error with status code."""
        import urllib.error
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "http://example.com/404", 404, "Not Found", {}, None
            ),
        ):
            from tools.transcription_tools import _probe_audio_url
            result = _probe_audio_url("http://example.com/404")

        assert result["success"] is False
        assert "404" in result["error"]

    def test_probe_connection_error(self):
        """Connection refused / DNS failure."""
        import urllib.error
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Name or service not known"),
        ):
            from tools.transcription_tools import _probe_audio_url
            result = _probe_audio_url("http://nonexistent.example/audio.mp3")

        assert result["success"] is False
        assert "Name or service" in result["error"]

    def test_probe_timeout_and_catchall(self):
        """Generic exception (e.g. timeout) caught by catch-all."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            from tools.transcription_tools import _probe_audio_url
            result = _probe_audio_url("http://example.com/audio.mp3")

        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_probe_adds_user_agent(self):
        """Every probe request includes a User-Agent header."""
        mock_resp = _mock_urlopen_success()
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            from tools.transcription_tools import _probe_audio_url
            _probe_audio_url("https://example.com/audio.mp3")

        req = mock_urlopen.call_args[0][0]
        import urllib.request
        assert isinstance(req, urllib.request.Request)
        # Use has_header which is a real Request method
        assert req.has_header("User-agent") or req.has_header("User-Agent")


# ============================================================================
# _download_audio
# ============================================================================

class TestDownloadAudio:
    def test_download_success(self):
        """Download MP3 audio from URL to a temp file."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"fake audio data"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.headers = {"Content-Type": "audio/mpeg"}
        mock_resp.url = "https://cdn.example.com/audio.mp3"

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from tools.transcription_tools import _download_audio
            result = _download_audio("https://example.com/audio.mp3")

        assert result["success"] is True
        assert "file_path" in result
        assert os.path.isfile(result["file_path"])
        with open(result["file_path"], "rb") as f:
            assert f.read() == b"fake audio data"
        # Clean up
        os.unlink(result["file_path"])

    def test_download_ogg_content_type(self):
        """Download with ogg content type gets .ogg extension."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"ogg data"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.headers = {"Content-Type": "audio/ogg"}
        mock_resp.url = "https://example.com/audio.ogg"

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from tools.transcription_tools import _download_audio
            result = _download_audio("https://example.com/audio.ogg")

        assert result["success"] is True
        assert result["file_path"].endswith(".ogg")
        os.unlink(result["file_path"])

    def test_download_wav_content_type(self):
        """WAV audio gets .wav extension."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"wav data"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.headers = {"Content-Type": "audio/wave"}
        mock_resp.url = "https://example.com/audio.wav"

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from tools.transcription_tools import _download_audio
            result = _download_audio("https://example.com/audio.wav")

        assert result["success"] is True
        assert result["file_path"].endswith(".wav")
        os.unlink(result["file_path"])

    def test_download_default_extension(self):
        """Unknown content type defaults to .mp3."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"data"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.headers = {"Content-Type": "application/octet-stream"}
        mock_resp.url = "https://example.com/audio"

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from tools.transcription_tools import _download_audio
            result = _download_audio("https://example.com/audio")

        assert result["success"] is True
        assert result["file_path"].endswith(".mp3")
        os.unlink(result["file_path"])

    def test_download_http_error(self):
        """HTTP error during download."""
        import urllib.error
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "http://example.com/403", 403, "Forbidden", {}, None
            ),
        ):
            from tools.transcription_tools import _download_audio
            result = _download_audio("http://example.com/403")

        assert result["success"] is False
        assert "403" in result["error"]

    def test_download_connection_error(self):
        """Connection error during download."""
        import urllib.error
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            from tools.transcription_tools import _download_audio
            result = _download_audio("http://example.com/audio.mp3")

        assert result["success"] is False
        assert "Connection refused" in result["error"]

    def test_download_adds_user_agent(self):
        """Download request includes User-Agent."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"data"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.headers = {"Content-Type": "audio/mpeg"}
        mock_resp.url = "https://example.com/audio.mp3"

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            from tools.transcription_tools import _download_audio
            _download_audio("https://example.com/audio.mp3")

        req = mock_urlopen.call_args[0][0]
        import urllib.request
        assert isinstance(req, urllib.request.Request)
        assert req.has_header("User-agent") or req.has_header("User-Agent")

    def test_download_timeout(self):
        """Timeout during download."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            from tools.transcription_tools import _download_audio
            result = _download_audio("http://example.com/audio.mp3")

        assert result["success"] is False
        assert "timed out" in result["error"]


# ============================================================================
# _split_audio
# ============================================================================

def _make_ffprobe_output(duration=60.0, size=12000000):
    """Create a mock ffprobe JSON output."""
    return json.dumps({
        "format": {
            "duration": str(duration),
            "size": str(size),
        }
    })


class TestSplitAudio:
    def test_split_success(self, sample_mp3, tmp_path):
        """Happy path — ffprobe + ffmpeg succeed, chunks collected."""
        fake_ffmpeg = shutil_which_ffmpeg() or "/usr/bin/ffmpeg"

        def fake_run(cmd, *args, **kwargs):
            if "ffprobe" in str(cmd):
                return MagicMock(
                    stdout=_make_ffprobe_output(duration=60.0, size=12000000),
                    returncode=0,
                )
            # ffmpeg split: create fake chunk files
            base = os.path.splitext(os.path.basename(sample_mp3))[0]
            out_dir = str(tmp_path)
            # The pattern uses the output_dir directly
            for i in range(3):
                chunk_path = os.path.join(out_dir, f"{base}_chunk_00{i}.mp3")
                with open(chunk_path, "wb") as f:
                    f.write(b"x" * 1000)
            return MagicMock(stdout="", stderr="")

        with patch("tools.transcription_tools._find_ffmpeg_binary", return_value=fake_ffmpeg), \
             patch("tools.transcription_tools.subprocess.run", side_effect=fake_run):
            from tools.transcription_tools import _split_audio
            result = _split_audio(sample_mp3, str(tmp_path))

        assert result["success"] is True
        assert len(result["chunks"]) == 3
        assert all(c.startswith(str(tmp_path)) for c in result["chunks"])

    def test_split_no_ffmpeg(self, sample_mp3, tmp_path):
        """ffmpeg not found returns error."""
        with patch("tools.transcription_tools._find_ffmpeg_binary", return_value=None):
            from tools.transcription_tools import _split_audio
            result = _split_audio(sample_mp3, str(tmp_path))

        assert result["success"] is False
        assert "ffmpeg not found" in result["error"]

    def test_split_no_duration(self, sample_mp3, tmp_path):
        """ffprobe returns no duration."""
        def fake_run(cmd, *args, **kwargs):
            return MagicMock(
                stdout=json.dumps({"format": {}}),
                returncode=0,
            )

        fake_ffmpeg = shutil_which_ffmpeg() or "/usr/bin/ffmpeg"
        with patch("tools.transcription_tools._find_ffmpeg_binary", return_value=fake_ffmpeg), \
             patch("tools.transcription_tools.subprocess.run", side_effect=fake_run):
            from tools.transcription_tools import _split_audio
            result = _split_audio(sample_mp3, str(tmp_path))

        assert result["success"] is False
        assert "Could not determine" in result.get("error", "")

    def test_split_no_chunks(self, sample_mp3, tmp_path):
        """ffmpeg runs but produces no chunk files."""
        def fake_run(cmd, *args, **kwargs):
            if "ffprobe" in str(cmd):
                return MagicMock(
                    stdout=_make_ffprobe_output(duration=60.0, size=12000000),
                    returncode=0,
                )
            return MagicMock(stdout="", stderr="")

        fake_ffmpeg = shutil_which_ffmpeg() or "/usr/bin/ffmpeg"
        with patch("tools.transcription_tools._find_ffmpeg_binary", return_value=fake_ffmpeg), \
             patch("tools.transcription_tools.subprocess.run", side_effect=fake_run):
            from tools.transcription_tools import _split_audio
            result = _split_audio(sample_mp3, str(tmp_path))

        assert result["success"] is False
        assert "no chunk files" in result.get("error", "")

    def test_split_ffmpeg_fails(self, sample_mp3, tmp_path):
        """ffmpeg split command fails."""
        def fake_run(cmd, *args, **kwargs):
            if "ffprobe" in str(cmd):
                return MagicMock(
                    stdout=_make_ffprobe_output(duration=60.0, size=12000000),
                    returncode=0,
                )
            import subprocess
            raise subprocess.CalledProcessError(1, cmd, stderr="invalid data")

        fake_ffmpeg = shutil_which_ffmpeg() or "/usr/bin/ffmpeg"
        with patch("tools.transcription_tools._find_ffmpeg_binary", return_value=fake_ffmpeg), \
             patch("tools.transcription_tools.subprocess.run", side_effect=fake_run):
            from tools.transcription_tools import _split_audio
            result = _split_audio(sample_mp3, str(tmp_path))

        assert result["success"] is False
        assert "invalid data" in result.get("error", "")


def shutil_which_ffmpeg():
    """Find ffmpeg on this system (for tests where it's required as path)."""
    import shutil
    return shutil.which("ffmpeg")


# ============================================================================
# _transcribe_chunks
# ============================================================================

class TestTranscribeChunks:
    def test_transcribe_chunks_all_succeed(self, tmp_path):
        """Multiple chunks all transcribe successfully, merged with markers."""
        chunks = []
        for i in range(3):
            p = os.path.join(str(tmp_path), f"chunk_{i}.ogg")
            with open(p, "w") as f:
                f.write("fake")
            chunks.append(p)

        def fake_groq(*args, **kwargs):
            idx = args[0]  # file_path
            return {"success": True, "transcript": f"Content of chunk {os.path.basename(idx)}", "provider": "groq"}

        with patch("tools.transcription_tools._transcribe_groq", side_effect=fake_groq):
            from tools.transcription_tools import _transcribe_chunks
            result = _transcribe_chunks(chunks, "whisper-large-v3-turbo")

        assert result["success"] is True
        assert "Content of chunk" in result["transcript"]
        assert result["transcript"].count("[") == 3  # Three segment markers
        assert result["provider"] == "groq"
        # Chunks should be cleaned up
        for c in chunks:
            assert not os.path.isfile(c)

    def test_transcribe_chunks_partial_failure(self, tmp_path):
        """Some chunks fail — error markers inserted, others preserved."""
        chunks = []
        for i in range(3):
            p = os.path.join(str(tmp_path), f"chunk_{i}.ogg")
            with open(p, "w") as f:
                f.write("fake")
            chunks.append(p)

        side_effects = [
            {"success": True, "transcript": "Part one", "provider": "groq"},
            {"success": False, "transcript": "", "error": "API error"},
            {"success": True, "transcript": "Part three", "provider": "groq"},
        ]

        with patch("tools.transcription_tools._transcribe_groq", side_effect=side_effects):
            from tools.transcription_tools import _transcribe_chunks
            result = _transcribe_chunks(chunks, "whisper-large-v3-turbo")

        assert result["success"] is True
        assert "Part one" in result["transcript"]
        assert "TRANSCRIPTION ERROR" in result["transcript"]
        assert "Part three" in result["transcript"]

    def test_transcribe_chunks_single_chunk(self, tmp_path):
        """Single chunk works without rate-limit delay."""
        p = os.path.join(str(tmp_path), "chunk_0.ogg")
        with open(p, "w") as f:
            f.write("fake")

        with patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "Hello", "provider": "groq"}):
            from tools.transcription_tools import _transcribe_chunks
            result = _transcribe_chunks([p], "whisper-large-v3-turbo")

        assert result["success"] is True
        assert "Hello" in result["transcript"]
        assert result["provider"] == "groq"


# ============================================================================
# transcribe_url (public API)
# ============================================================================

class TestTranscribeUrlPublicApi:
    def test_disabled_stt(self):
        """STT disabled returns error immediately."""
        with patch("tools.transcription_tools._load_stt_config", return_value={"enabled": False}):
            from tools.transcription_tools import transcribe_url
            result = transcribe_url("http://example.com/audio.mp3")
        assert result["success"] is False
        assert "disabled" in result["error"].lower()

    def test_non_groq_provider_rejected(self):
        """Only Groq provider is supported for URL transcription."""
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openai"}), \
             patch("tools.transcription_tools._get_provider", return_value="openai"):
            from tools.transcription_tools import transcribe_url
            result = transcribe_url("http://example.com/audio.mp3")
        assert result["success"] is False
        assert "Groq" in result["error"]

    def test_no_groq_key(self):
        """No GROQ_API_KEY set."""
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools.get_env_value", return_value=None):
            from tools.transcription_tools import transcribe_url
            result = transcribe_url("http://example.com/audio.mp3")
        assert result["success"] is False
        assert "GROQ_API_KEY" in result["error"]

    def test_missing_openai_package(self):
        """openai package not installed."""
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools.get_env_value", return_value="gsk-test"), \
             patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import transcribe_url
            result = transcribe_url("http://example.com/audio.mp3")
        assert result["success"] is False
        assert "openai package" in result["error"]

    def test_probe_failure(self):
        """URL probe fails, error propagated."""
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools.get_env_value", return_value="gsk-test"), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._probe_audio_url",
                   return_value={"success": False, "url": "http://bad.example/", "error": "HTTP 404: Not Found"}):
            from tools.transcription_tools import transcribe_url
            result = transcribe_url("http://bad.example/audio.mp3")
        assert result["success"] is False
        assert "HTTP 404" in result["error"]

    def test_direct_url_within_limit(self):
        """Small audio within MAX_FILE_SIZE — passes URL directly to Groq."""
        from tools.transcription_tools import MAX_FILE_SIZE
        small_size = MAX_FILE_SIZE  # exactly at limit

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools.get_env_value", return_value="gsk-test"), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._probe_audio_url",
                   return_value={
                       "success": True,
                       "url": "https://cdn.example.com/audio.mp3",
                       "content_type": "audio/mpeg",
                       "content_length": small_size,
                   }), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "Hello world", "provider": "groq"}) as mock_groq:
            from tools.transcription_tools import transcribe_url
            result = transcribe_url("http://example.com/audio.mp3")

        assert result["success"] is True
        assert result["transcript"] == "Hello world"
        # Verify Groq was called with audio_url
        call_kwargs = mock_groq.call_args
        assert call_kwargs[1].get("audio_url") == "https://cdn.example.com/audio.mp3"

    def test_large_url_downloads_and_chunks(self, tmp_path, monkeypatch):
        """Large audio over MAX_FILE_SIZE — downloads, splits, transcribes."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        # Setup: probe says it's large
        size_over_limit = 26 * 1024 * 1024  # > 25 MB

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools.get_env_value", return_value="gsk-test"), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._probe_audio_url",
                   return_value={
                       "success": True,
                       "url": "https://cdn.example.com/podcast.mp3",
                       "content_type": "audio/mpeg",
                       "content_length": size_over_limit,
                   }), \
             patch("tools.transcription_tools._download_audio",
                   return_value={"success": True, "file_path": str(tmp_path / "downloaded.mp3")}), \
             patch("tools.transcription_tools._split_audio",
                   return_value={
                       "success": True,
                       "chunks": [
                           str(tmp_path / "chunk_001.mp3"),
                           str(tmp_path / "chunk_002.mp3"),
                       ],
                   }), \
             patch("tools.transcription_tools._transcribe_chunks",
                   return_value={
                       "success": True,
                       "transcript": "[00:00] Intro content\n\n[00:30] Main content",
                       "provider": "groq",
                   }) as mock_chunks:
            from tools.transcription_tools import transcribe_url
            result = transcribe_url("http://example.com/podcast.mp3")

        assert result["success"] is True
        assert "[00:00]" in result["transcript"]
        mock_chunks.assert_called_once()

    def test_large_url_download_fails(self, tmp_path):
        """Download failure propagated."""
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools.get_env_value", return_value="gsk-test"), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._probe_audio_url",
                   return_value={
                       "success": True,
                       "url": "https://cdn.example.com/podcast.mp3",
                       "content_type": "audio/mpeg",
                       "content_length": 50 * 1024 * 1024,
                   }), \
             patch("tools.transcription_tools._download_audio",
                   return_value={"success": False, "error": "HTTP 403: Forbidden"}):
            from tools.transcription_tools import transcribe_url
            result = transcribe_url("http://example.com/podcast.mp3")
        assert result["success"] is False
        assert "Forbidden" in result["error"]

    def test_large_url_split_fails(self, tmp_path):
        """Split failure propagated."""
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools.get_env_value", return_value="gsk-test"), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._probe_audio_url",
                   return_value={
                       "success": True,
                       "url": "https://cdn.example.com/podcast.mp3",
                       "content_type": "audio/mpeg",
                       "content_length": 50 * 1024 * 1024,
                   }), \
             patch("tools.transcription_tools._download_audio",
                   return_value={"success": True, "file_path": str(tmp_path / "downloaded.mp3")}), \
             patch("tools.transcription_tools._split_audio",
                   return_value={"success": False, "error": "ffmpeg not found"}):
            from tools.transcription_tools import transcribe_url
            result = transcribe_url("http://example.com/podcast.mp3")
        assert result["success"] is False
        assert "ffmpeg" in result["error"]

    def test_url_with_no_content_length(self, monkeypatch):
        """URL returns no Content-Length — treated as over limit, download path used."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools.get_env_value", return_value="gsk-test"), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._probe_audio_url",
                   return_value={
                       "success": True,
                       "url": "https://cdn.example.com/stream",
                       "content_type": "audio/mpeg",
                       "content_length": None,  # unknown size → goes to download path
                   }), \
             patch("tools.transcription_tools._download_audio",
                   return_value={"success": True, "file_path": "/tmp/hermes-url-audio-test.mp3"}), \
             patch("tools.transcription_tools._split_audio",
                   return_value={"success": True, "chunks": ["/tmp/chunk.mp3"]}), \
             patch("tools.transcription_tools._transcribe_chunks",
                   return_value={"success": True, "transcript": "Stream content", "provider": "groq"}):
            from tools.transcription_tools import transcribe_url
            result = transcribe_url("https://stream.example/audio")

        assert result["success"] is True
        assert "Stream content" in result["transcript"]