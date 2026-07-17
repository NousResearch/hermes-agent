"""Tests for streaming TTS providers (_stream_edge, _stream_openai).

These implementations exist for future real-time consumers (voice-mode
playback, gateway chunk forwarding). The dispatcher's default
text_to_speech_tool path still uses _generate_* — these tests pin the
streaming contract so the dead code doesn't rot silently.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("OPENAI_API_KEY", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Edge TTS streaming
# ---------------------------------------------------------------------------


def _mock_edge_tts(events):
    """Build a mock edge_tts module whose Communicate(...).stream() yields events."""
    async def _aiter():
        for ev in events:
            yield ev

    mock_comm = MagicMock()
    mock_comm.stream.return_value = _aiter()
    mock_comm.aclose = MagicMock()
    mock_edge = MagicMock()
    mock_edge.Communicate.return_value = mock_comm
    return mock_edge


class TestStreamEdge:
    def test_yields_only_audio_data(self):
        """Non-audio events (WordBoundary, end) are filtered out; empty data skipped."""
        events = [
            {"type": "audio", "data": b"chunk1"},
            {"type": "WordBoundary", "offset": 100, "duration": 50},
            {"type": "audio", "data": b"chunk2"},
            {"type": "SentenceBoundary", "offset": 200},
            {"type": "audio", "data": b""},
            {"type": "end"},
        ]
        with patch("tools.tts_tool._import_edge_tts",
                   return_value=_mock_edge_tts(events)):
            from tools.tts_tool import _stream_edge
            chunks = list(_stream_edge("hello", tts_config={}))
        assert chunks == [b"chunk1", b"chunk2"]

    def test_is_synchronous_iterator(self):
        """ABC contract: stream() returns a sync Iterator[bytes], not async."""
        with patch("tools.tts_tool._import_edge_tts",
                   return_value=_mock_edge_tts([{"type": "audio", "data": b"x"}])):
            from tools.tts_tool import _stream_edge
            gen = _stream_edge("hi", tts_config={})
            # Sync iterator protocol — must not be an async generator
            assert hasattr(gen, "__next__")
            assert not hasattr(gen, "__anext__")
            assert next(gen) == b"x"

    def test_speed_translates_to_rate_percent(self):
        """tts.edge.speed=2.0 → Communicate(rate='+100%')."""
        mock_edge = _mock_edge_tts([{"type": "audio", "data": b"x"}])
        with patch("tools.tts_tool._import_edge_tts", return_value=mock_edge):
            from tools.tts_tool import _stream_edge
            list(_stream_edge("hi", tts_config={"edge": {"speed": 2.0}}))
        kwargs = mock_edge.Communicate.call_args[1]
        assert kwargs["rate"] == "+100%"


# ---------------------------------------------------------------------------
# OpenAI TTS streaming
# ---------------------------------------------------------------------------


class TestStreamOpenai:
    def _run(self, tts_config, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.iter_bytes.return_value = iter([b"a", b"b", b"c"])
        mock_client = MagicMock()
        mock_client.audio.speech.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool._resolve_openai_audio_client_config",
                   return_value=("test-key", None, False)):
            from tools.tts_tool import _stream_openai
            chunks = list(_stream_openai("hello", tts_config=tts_config))
        return chunks, mock_client.audio.speech.create

    def test_yields_chunks_from_iter_bytes(self, monkeypatch):
        chunks, _ = self._run({}, monkeypatch)
        assert chunks == [b"a", b"b", b"c"]

    def test_does_not_pass_stream_kwarg(self, monkeypatch):
        """OpenAI SDK's audio.speech.create() has no ``stream`` parameter.

        The HttpxBinaryResponseContent returned by create() is already
        stream-capable via iter_bytes(); passing stream=True is invalid
        on the SDK and was a leftover from an incorrect port.
        """
        _, create = self._run({}, monkeypatch)
        kwargs = create.call_args[1]
        assert "stream" not in kwargs, (
            f"stream must not be passed to audio.speech.create(); got kwargs={kwargs}"
        )

    def test_format_opus_maps_to_response_format(self, monkeypatch):
        """ABC format='opus' (default for voice bubbles) → SDK response_format='opus'."""
        _, create = self._run({}, monkeypatch)
        assert create.call_args[1]["response_format"] == "opus"

    def test_speed_clamped(self, monkeypatch):
        """Speed is clamped to [0.25, 4.0] like _generate_openai_tts."""
        _, create = self._run({"openai": {"speed": 10.0}}, monkeypatch)
        assert create.call_args[1]["speed"] == 4.0
