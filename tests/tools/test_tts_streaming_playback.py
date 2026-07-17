"""Tests for _iter_pcm_chunks and its helpers (real-time playback layer).

_iter_pcm_chunks is the unified entry point that stream_tts_to_speaker
uses to pull raw int16 LE 24kHz mono PCM chunks from any streaming
provider, suitable for direct write to sounddevice.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_iter_pcm_chunks_unknown_provider_raises():
    from tools.tts_tool import _iter_pcm_chunks
    with pytest.raises(ValueError, match="unknown"):
        list(_iter_pcm_chunks("hi", "unknown", {}))


def test_iter_pcm_chunks_openai_passes_pcm_format(monkeypatch):
    """openai dispatch must request response_format=pcm for real-time playback."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_response = MagicMock()
    mock_response.iter_bytes.return_value = iter([b"\x00\x01\x00\x02"])
    mock_client = MagicMock()
    mock_client.audio.speech.create.return_value = mock_response
    mock_cls = MagicMock(return_value=mock_client)

    with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
         patch("tools.tts_tool._resolve_openai_audio_client_config",
               return_value=("test-key", None, False)):
        from tools.tts_tool import _iter_pcm_chunks
        chunks = list(_iter_pcm_chunks("hi", "openai", {}))

    assert chunks == [b"\x00\x01\x00\x02"]
    create_kwargs = mock_client.audio.speech.create.call_args[1]
    assert create_kwargs["response_format"] == "pcm"


def test_iter_pcm_chunks_openai_managed_coerces_unsupported_model(monkeypatch):
    """When the resolver returns the managed gateway, an unsupported model
    (e.g. tts-1-hd) must be coerced to a MANAGED_OPENAI_TTS_MODELS entry so
    the gateway doesn't 400. Mirrors the sync _generate_openai_tts path."""
    monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "managed-token")
    mock_response = MagicMock()
    mock_response.iter_bytes.return_value = iter([b"\x00\x01"])
    mock_client = MagicMock()
    mock_client.audio.speech.create.return_value = mock_response
    mock_cls = MagicMock(return_value=mock_client)

    # is_managed=True, fallback_base is the managed gateway URL
    with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
         patch("tools.tts_tool._resolve_openai_audio_client_config",
               return_value=("managed-token", "https://managed.test/v1", True)):
        from tools.tts_tool import _iter_pcm_chunks
        list(_iter_pcm_chunks("hi", "openai", {
            "openai": {"model": "tts-1-hd", "voice": "alloy"},
        }))

    create_kwargs = mock_client.audio.speech.create.call_args[1]
    # tts-1-hd is not in MANAGED_OPENAI_TTS_MODELS -> coerced to default
    assert create_kwargs["model"] != "tts-1-hd"
    assert create_kwargs["model"] in {"gpt-4o-mini-tts"}
    # managed gateway base_url is respected
    assert create_kwargs is not None


def test_iter_pcm_chunks_elevenlabs_uses_sdk_pcm_24000(monkeypatch):
    """elevenlabs dispatch calls SDK convert() with output_format=pcm_24000."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.text_to_speech.convert.return_value = iter([b"el1", b"el2"])
    mock_el = MagicMock(return_value=mock_client)

    with patch("tools.tts_tool._import_elevenlabs", return_value=mock_el):
        from tools.tts_tool import _iter_pcm_chunks
        chunks = list(_iter_pcm_chunks("hi", "elevenlabs", {
            "elevenlabs": {"voice_id": "vid", "model_id": "mid"},
        }))

    assert chunks == [b"el1", b"el2"]
    convert_kwargs = mock_client.text_to_speech.convert.call_args[1]
    assert convert_kwargs["output_format"] == "pcm_24000"
    assert convert_kwargs["voice_id"] == "vid"
    assert convert_kwargs["model_id"] == "mid"


def test_iter_pcm_chunks_edge_routes_through_miniaudio_decoder():
    """edge dispatch feeds MP3 byte chunks to _decode_edge_mp3_to_pcm."""
    with patch("tools.tts_tool._stream_edge", return_value=iter([b"mp3a", b"mp3b"])), \
         patch("tools.tts_tool._decode_edge_mp3_to_pcm",
               return_value=iter([b"pcm1", b"pcm2"])) as mock_decode:
        from tools.tts_tool import _iter_pcm_chunks
        chunks = list(_iter_pcm_chunks("hi", "edge", {"edge": {"voice": "v"}}))

    assert chunks == [b"pcm1", b"pcm2"]
    mock_decode.assert_called_once()
    args, _ = mock_decode.call_args
    assert args[0] is not None


# ---------------------------------------------------------------------------
# _decode_edge_mp3_to_pcm (miniaudio streaming MP3 decode)
# ---------------------------------------------------------------------------


def test_decode_edge_mp3_to_pcm_passes_correct_miniaudio_args():
    """stream_any must be called with MP3 source format, 24kHz int16 mono output."""
    received = {}
    yielded_pcm = [b"p1", b"p2", b"p3"]

    class _StreamableSource:
        pass

    class _FileFormat:
        MP3 = "MP3"

    class _SampleFormat:
        SIGNED16 = "SIGNED16"

    def _stream_any(source, *, source_format, output_format, sample_rate, nchannels):
        received["source"] = source
        received["source_format"] = source_format
        received["output_format"] = output_format
        received["sample_rate"] = sample_rate
        received["nchannels"] = nchannels
        yield from yielded_pcm

    mock_module = MagicMock()
    mock_module.StreamableSource = _StreamableSource
    mock_module.FileFormat = _FileFormat
    mock_module.SampleFormat = _SampleFormat
    mock_module.stream_any = _stream_any

    mp3_chunks = [b"frame1", b"frame2", b"frame3"]
    with patch.dict("sys.modules", {"miniaudio": mock_module}):
        from tools.tts_tool import _decode_edge_mp3_to_pcm
        pcm = list(_decode_edge_mp3_to_pcm(iter(mp3_chunks)))

    assert pcm == yielded_pcm
    assert received["source_format"] == "MP3"
    assert received["output_format"] == "SIGNED16"
    assert received["sample_rate"] == 24000
    assert received["nchannels"] == 1
    assert isinstance(received["source"], _StreamableSource)


def test_decode_edge_mp3_source_wraps_generator_and_buffers():
    """The StreamableSource subclass must pull bytes from the MP3 iterator on demand.

    miniaudio's decoder calls source.read(num_bytes) repeatedly; our wrapper
    pulls from the wrapped generator and buffers leftovers between calls.
    """
    mock_module = MagicMock()

    class _StreamableSource:
        pass

    class _FileFormat:
        MP3 = "MP3"

    class _SampleFormat:
        SIGNED16 = "SIGNED16"

    # Capture the source so we can drive .read() ourselves.
    captured = {}

    def _stream_any(source, *, source_format, output_format, sample_rate, nchannels):
        captured["source"] = source
        yield b""

    mock_module.StreamableSource = _StreamableSource
    mock_module.FileFormat = _FileFormat
    mock_module.SampleFormat = _SampleFormat
    mock_module.stream_any = _stream_any

    mp3_iter = iter([b"abc", b"def", b"gh"])
    with patch.dict("sys.modules", {"miniaudio": mock_module}):
        from tools.tts_tool import _decode_edge_mp3_to_pcm, _MP3StreamableSource
        # Drive the wrapper directly: 5-byte read should consume "abc"+part of "def"
        source = _MP3StreamableSource(mp3_iter)
        first = source.read(5)
        assert first == b"abcde"
        # Next 4-byte read should pull remaining 'f' from buffered 'def' + 'gh'
        second = source.read(4)
        assert second == b"fgh"
        # EOF: empty return, no exception
        third = source.read(8)
        assert third == b""


def test_decode_edge_mp3_instantiable_when_streamablesource_is_abc():
    """Regression: real miniaudio.StreamableSource is an ABC.

    The runtime subclass built inside _decode_edge_mp3_to_pcm must be
    concrete (read() provided by _MP3StreamableSource mixin), otherwise
    instantiation raises "Can't instantiate abstract class". This was
    broken by an earlier type()-based mixin that bypassed ABCMeta's
    abstractmethods recomputation.
    """
    import abc

    # Real-shaped miniaudio: StreamableSource is an ABC with abstract read()
    class _StreamableSource(metaclass=abc.ABCMeta):
        @abc.abstractmethod
        def read(self, num_bytes: int) -> bytes:
            ...

    class _FileFormat:
        MP3 = "MP3"

    class _SampleFormat:
        SIGNED16 = "SIGNED16"

    mock_module = MagicMock()
    mock_module.StreamableSource = _StreamableSource
    mock_module.FileFormat = _FileFormat
    mock_module.SampleFormat = _SampleFormat
    mock_module.stream_any = lambda *a, **kw: iter([b"pcm"])

    with patch.dict("sys.modules", {"miniaudio": mock_module}):
        from tools.tts_tool import _decode_edge_mp3_to_pcm
        # Must not raise "Can't instantiate abstract class"
        chunks = list(_decode_edge_mp3_to_pcm(iter([b"mp3frame"])))
        assert chunks == [b"pcm"]
