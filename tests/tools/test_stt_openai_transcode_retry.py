import types

import pytest

import tools.transcription_tools as tt


class _FakeAPIError(Exception):
    def __init__(self, message: str, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class _FakeBadRequestError(_FakeAPIError):
    pass


def _install_fake_openai(monkeypatch, calls, first_error):
    class FakeTranscriptions:
        def create(self, **kwargs):
            calls.append(kwargs["file"].name)
            if len(calls) == 1:
                raise first_error
            return {"text": "retry ok"}

    class FakeAudio:
        def __init__(self):
            self.transcriptions = FakeTranscriptions()

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.audio = FakeAudio()
            self.closed = False

        def close(self):
            self.closed = True

    fake_openai = types.SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIError=_FakeAPIError,
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
        APITimeoutError=type("APITimeoutError", (Exception,), {}),
        BadRequestError=_FakeBadRequestError,
    )
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)
    monkeypatch.setattr(tt, "_HAS_OPENAI", True)


def test_openai_stt_retries_5xx_webm_after_transcoding(monkeypatch, tmp_path):
    source = tmp_path / "voice.webm"
    source.write_bytes(b"webm-ish")
    calls = []
    _install_fake_openai(monkeypatch, calls, _FakeAPIError("upstream decoder failed", status_code=500))

    def fake_transcode(file_path, work_dir):
        converted = tmp_path / "voice.m4a"
        converted.write_bytes(b"m4a-ish")
        return str(converted), None

    monkeypatch.setattr(tt, "_transcode_audio_for_stt", fake_transcode)
    monkeypatch.setattr(tt, "_resolve_stt_language", lambda _provider: None)

    result = tt._transcribe_openai(str(source), "gpt-4o-transcribe", api_key="test")

    assert result == {"success": True, "transcript": "retry ok", "provider": "openai"}
    assert calls == [str(source), str(tmp_path / "voice.m4a")]


def test_openai_stt_does_not_retry_unrelated_5xx_for_plain_wav(monkeypatch, tmp_path):
    source = tmp_path / "voice.wav"
    source.write_bytes(b"wav-ish")
    calls = []
    _install_fake_openai(monkeypatch, calls, _FakeAPIError("temporary server outage", status_code=500))
    transcode = pytest.fail
    monkeypatch.setattr(tt, "_transcode_audio_for_stt", transcode)
    monkeypatch.setattr(tt, "_resolve_stt_language", lambda _provider: None)

    result = tt._transcribe_openai(str(source), "gpt-4o-transcribe", api_key="test")

    assert result["success"] is False
    assert result["error"].startswith("API error:")
    assert calls == [str(source)]
