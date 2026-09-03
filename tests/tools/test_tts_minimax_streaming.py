"""Tests for the MiniMax chunked-PCM streamer (tools.tts_streaming.MiniMaxStreamer).

No network: ``requests.post`` and the region/credential resolver are mocked, so
these assert the SSE parsing contract rather than MiniMax's live behaviour.

The load-bearing case is ``test_summary_event_is_not_yielded``. MiniMax ends a
stream with a summary event whose ``data.audio`` repeats the WHOLE utterance
rather than carrying a final increment. Yielding it plays every sentence twice
at exactly 2x the bytes — and it survives a naive round-trip check, because
Whisper transcribes duplicated audio back to one clean sentence. Duration is
the only signal that catches it, which is what this test pins.
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.tts_streaming as ts


def _sse(events):
    """Encode *events* as the SSE ``data:`` lines ``iter_lines`` would hand back."""
    return [b"data: " + json.dumps(e).encode() for e in events]


class _Response:
    """Minimal stand-in for a streaming ``requests`` response.

    A real class, not SimpleNamespace: ``with requests.post(...)`` looks the
    context-manager dunders up on the type, so instance attributes won't do.
    """

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextmanager
def _minimax(events, lines=None):
    """Run MiniMaxStreamer against a canned SSE event list."""
    response = _Response(_sse(events) if lines is None else lines)

    class _Post:
        def __call__(self, *a, **kw):
            self.kwargs = kw
            return response

    post = _Post()
    runtime = SimpleNamespace(endpoint="https://api.minimax.io/v1/t2a_v2", api_key="k")
    with patch("requests.post", post), \
         patch("tools.tts_tool._resolve_minimax_tts_runtime", return_value=runtime):
        yield post


def _audio(hex_bytes, status=1, **extra):
    return {"data": {"audio": hex_bytes, "status": status}, **extra}


class TestMiniMaxStreamerSSE:
    def test_yields_decoded_pcm_for_each_incremental_chunk(self):
        with _minimax([_audio("0011"), _audio("2233")]):
            out = list(ts.MiniMaxStreamer({}, {}).stream("hello"))
        assert out == [b"\x00\x11", b"\x22\x33"]

    def test_summary_event_is_not_yielded(self):
        """status=2 repeats the entire utterance — yielding it doubles the audio."""
        whole = "0011" + "2233"
        with _minimax([
            _audio("0011"),
            _audio("2233"),
            _audio(whole, status=2, extra_info={"audio_length": 123}),
        ]):
            out = list(ts.MiniMaxStreamer({}, {}).stream("hello"))
        assert out == [b"\x00\x11", b"\x22\x33"]
        assert b"".join(out) == b"\x00\x11\x22\x33"  # 4 bytes, not 8

    def test_extra_info_alone_marks_a_summary_event(self):
        """Some responses carry extra_info without status=2; still a summary."""
        with _minimax([_audio("0011"), _audio("0011", extra_info={"usage": 1})]):
            out = list(ts.MiniMaxStreamer({}, {}).stream("hello"))
        assert out == [b"\x00\x11"]

    def test_api_error_raises_instead_of_yielding_silence(self):
        events = [{"base_resp": {"status_code": 1004, "status_msg": "bad key"}}]
        with _minimax(events):
            with pytest.raises(RuntimeError, match="1004"):
                list(ts.MiniMaxStreamer({}, {}).stream("hello"))

    def test_undecodable_chunk_is_skipped_not_fatal(self):
        with _minimax([_audio("zzzz"), _audio("0011")]):
            out = list(ts.MiniMaxStreamer({}, {}).stream("hello"))
        assert out == [b"\x00\x11"]

    def test_non_data_and_malformed_lines_are_ignored(self):
        lines = [
            b"",
            b"event: ping",
            b"data: {oops",
            b"data: " + json.dumps(_audio("0011")).encode(),
        ]
        with _minimax([], lines=lines):
            out = list(ts.MiniMaxStreamer({}, {}).stream("hello"))
        assert out == [b"\x00\x11"]


class TestMiniMaxStreamerRequest:
    def test_requests_pcm_at_the_declared_sample_rate(self):
        """The interface promises int16 PCM at ``sample_rate``; ask for exactly that."""
        with _minimax([_audio("0011")]) as post:
            list(ts.MiniMaxStreamer({}, {}).stream("hello"))
        audio = post.kwargs["json"]["audio_setting"]
        assert post.kwargs["json"]["stream"] is True
        assert audio["format"] == "pcm"
        assert audio["channel"] == 1
        assert audio["sample_rate"] == ts.MiniMaxStreamer.sample_rate

    def test_voice_and_model_come_from_the_tts_minimax_section(self):
        """Enabling streaming must not silently change the configured voice."""
        section = {"model": "speech-02-turbo", "voice_id": "Custom_voice"}
        with _minimax([_audio("0011")]) as post:
            list(ts.MiniMaxStreamer({"minimax": section}, section).stream("hi"))
        body = post.kwargs["json"]
        assert body["model"] == "speech-02-turbo"
        assert body["voice_setting"]["voice_id"] == "Custom_voice"

    def test_emotion_is_only_sent_when_configured(self):
        with _minimax([_audio("0011")]) as post:
            list(ts.MiniMaxStreamer({}, {}).stream("hi"))
        assert "emotion" not in post.kwargs["json"]["voice_setting"]

        with _minimax([_audio("0011")]) as post:
            list(ts.MiniMaxStreamer({}, {"emotion": "happy"}).stream("hi"))
        assert post.kwargs["json"]["voice_setting"]["emotion"] == "happy"


class TestMiniMaxStreamerResolution:
    def test_registered_under_minimax(self):
        assert ts._REGISTRY.get("minimax") is ts.MiniMaxStreamer

    def test_configured_minimax_provider_resolves_to_this_streamer(self):
        """tts.provider: minimax should stream, not fall back to whole-reply."""
        with patch.object(ts.MiniMaxStreamer, "available", staticmethod(lambda: True)):
            inst = ts.resolve_streaming_provider({"provider": "minimax"})
        assert isinstance(inst, ts.MiniMaxStreamer)
