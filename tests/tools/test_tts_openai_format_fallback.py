"""OpenAI-compatible TTS: fall back to mp3 when a backend rejects the format.

`_generate_openai_tts` derives ``response_format`` from the output extension,
so a ``.ogg`` target requests ``opus``. Self-hosted OpenAI-compatible speech
servers (Speaches/Kokoro, …) that implement only mp3/wav/flac/pcm reject that
request with an HTTP 400/422 *before any bytes exist*, so the post-synthesis
``_repair_ogg_container`` hook (which only rescues backends that silently
*ignore* the parameter) never runs and every voice reply fails. The fallback
retries once as mp3 and lets the existing repair step rebuild the Ogg/Opus
container. See issue #73470 (follow-up to #57184).
"""

from pathlib import Path

import pytest

from tools import tts_tool


class _RejectResponseFormat(Exception):
    """Mimics the OpenAI SDK's UnprocessableEntityError shape."""

    def __init__(self, status_code: int = 422):
        self.status_code = status_code
        super().__init__(
            "Error code: 422 - {'detail': [{'type': 'literal_error', "
            "'loc': ['body', 'response_format'], "
            "'msg': \"Input should be 'mp3', 'flac', 'wav' or 'pcm'\", "
            "'input': 'opus'}]}"
        )


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def stream_to_file(self, output_path: str) -> None:
        Path(output_path).write_bytes(self._payload)


class _FakeSpeech:
    """Rejects any non-mp3 response_format once, then succeeds on mp3."""

    def __init__(self, calls: list):
        self._calls = calls

    def create(self, **kwargs):
        self._calls.append(kwargs)
        if kwargs.get("response_format") != "mp3":
            raise _RejectResponseFormat()
        return _FakeResponse(b"ID3mp3-bytes")


class _FakeAudio:
    def __init__(self, calls: list):
        self.speech = _FakeSpeech(calls)


class _FakeClient:
    def __init__(self, calls: list):
        self.audio = _FakeAudio(calls)
        self.closed = False

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------
# _is_response_format_rejection
# --------------------------------------------------------------------------
class TestIsResponseFormatRejection:
    def test_422_naming_response_format_is_a_rejection(self):
        assert tts_tool._is_response_format_rejection(_RejectResponseFormat(422))

    def test_400_naming_response_format_is_a_rejection(self):
        assert tts_tool._is_response_format_rejection(_RejectResponseFormat(400))

    def test_other_4xx_is_not_a_rejection(self):
        err = _RejectResponseFormat(404)
        assert not tts_tool._is_response_format_rejection(err)

    def test_422_not_naming_response_format_is_not_a_rejection(self):
        class _Other(Exception):
            status_code = 422

        assert not tts_tool._is_response_format_rejection(_Other("bad voice id"))

    def test_error_without_status_code_is_not_a_rejection(self):
        assert not tts_tool._is_response_format_rejection(
            RuntimeError("response_format is wrong but no status")
        )


# --------------------------------------------------------------------------
# _openai_speech_create_with_format_fallback
# --------------------------------------------------------------------------
class TestSpeechCreateFallback:
    def test_opus_rejection_retries_as_mp3(self):
        calls: list = []
        client = _FakeClient(calls)
        create_kwargs = {
            "model": "kokoro",
            "voice": "af_heart",
            "input": "hi",
            "response_format": "opus",
            "extra_headers": {"x-idempotency-key": "original"},
        }

        resp = tts_tool._openai_speech_create_with_format_fallback(client, create_kwargs)

        assert isinstance(resp, _FakeResponse)
        assert [c["response_format"] for c in calls] == ["opus", "mp3"]
        # Retry is a distinct request → a fresh idempotency key.
        assert calls[1]["extra_headers"]["x-idempotency-key"] != "original"
        # Non-format kwargs are carried through unchanged.
        assert calls[1]["voice"] == "af_heart"
        assert calls[1]["input"] == "hi"

    def test_native_opus_support_makes_no_extra_call(self):
        calls: list = []

        class _OkSpeech:
            def create(self, **kwargs):
                calls.append(kwargs)
                return _FakeResponse(b"OggS-opus")

        client = _FakeClient(calls)
        client.audio.speech = _OkSpeech()

        resp = tts_tool._openai_speech_create_with_format_fallback(
            client, {"response_format": "opus"}
        )

        assert isinstance(resp, _FakeResponse)
        assert len(calls) == 1

    def test_unrelated_error_propagates_without_retry(self):
        calls: list = []

        class _BoomSpeech:
            def create(self, **kwargs):
                calls.append(kwargs)
                raise RuntimeError("connection reset")

        client = _FakeClient(calls)
        client.audio.speech = _BoomSpeech()

        with pytest.raises(RuntimeError, match="connection reset"):
            tts_tool._openai_speech_create_with_format_fallback(
                client, {"response_format": "opus"}
            )
        assert len(calls) == 1

    def test_mp3_rejection_is_not_retried(self):
        calls: list = []

        class _RejectSpeech:
            def create(self, **kwargs):
                calls.append(kwargs)
                raise _RejectResponseFormat()

        client = _FakeClient(calls)
        client.audio.speech = _RejectSpeech()

        # Already mp3 → nothing safer to fall back to; propagate.
        with pytest.raises(_RejectResponseFormat):
            tts_tool._openai_speech_create_with_format_fallback(
                client, {"response_format": "mp3"}
            )
        assert len(calls) == 1

    @pytest.mark.parametrize("requested", ["wav", "flac", "pcm"])
    def test_non_opus_rejection_is_not_retried_as_mp3(self, requested):
        """Only opus has a post-synthesis container repair. Retrying a rejected
        wav/flac/pcm request as mp3 would write mp3 bytes under a .wav/.flac/.pcm
        path, so those rejections must propagate untouched (#73470 review)."""
        calls: list = []

        class _RejectSpeech:
            def create(self, **kwargs):
                calls.append(kwargs)
                raise _RejectResponseFormat()

        client = _FakeClient(calls)
        client.audio.speech = _RejectSpeech()

        with pytest.raises(_RejectResponseFormat):
            tts_tool._openai_speech_create_with_format_fallback(
                client, {"response_format": requested}
            )
        # No mp3 retry — the single failed call is the only one.
        assert [c["response_format"] for c in calls] == [requested]


# --------------------------------------------------------------------------
# End-to-end through _generate_openai_tts (opus .ogg target recovers to mp3)
# --------------------------------------------------------------------------
def test_generate_openai_tts_recovers_from_opus_rejection(tmp_path, monkeypatch):
    calls: list = []
    client = _FakeClient(calls)
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: (lambda **kw: client))

    out = tmp_path / "reply.ogg"
    result = tts_tool._generate_openai_tts(
        "testing",
        str(out),
        {},
        api_key="sk-test",
        base_url="http://speaches:8100/v1",
        model="speaches-ai/Kokoro-82M-v1.0-ONNX",
        voice="af_heart",
    )

    assert result == str(out)
    # opus requested first (from the .ogg extension), then mp3 after rejection.
    assert [c["response_format"] for c in calls] == ["opus", "mp3"]
    assert out.read_bytes() == b"ID3mp3-bytes"
    assert client.closed is True
