"""Tests for the provider-agnostic streaming TTS backend (tools.tts_streaming)
and its dispatch through tools.tts_tool.stream_tts_to_speaker.

No live audio or network: the ElevenLabs/OpenAI SDKs, sounddevice, and the sync
synth path are all mocked. Covers the registry/resolver, provider availability,
the chunked-streamer playback path, and the universal per-sentence sync fallback.
"""

import json
import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tools.tts_streaming as ts
from tools import tts_tool

pytest.importorskip("numpy")


# ── SentenceChunker ──────────────────────────────────────────────────────


class TestSentenceChunker:
    def test_cuts_sentence_the_moment_its_boundary_arrives(self):
        c = ts.SentenceChunker()
        assert c.feed("This is the first full") == []
        assert c.feed(" sentence of it all. And") == ["This is the first full sentence of it all. "]
        assert c.flush() == ["And"]


    def test_think_blocks_are_stripped_even_across_deltas(self):
        c = ts.SentenceChunker()
        assert c.feed("<think>secret reason") == []
        assert c.feed("ing</think>The actual spoken answer. ") == ["The actual spoken answer. "]


    def test_paragraph_break_is_a_boundary(self):
        c = ts.SentenceChunker()
        assert c.feed("A paragraph without punctuation\n\nnext one") == [
            "A paragraph without punctuation\n\n"
        ]


class TestSplitTextForStreamingTTS:
    def test_short_prefix_precedes_every_piece_of_over_cap_tail(self):
        text = "Short. " + "x" * 23

        pieces = ts.split_text_for_tts(text, 10)

        assert pieces == ["Short. ", "x" * 10, "x" * 10, "x" * 3]
        assert all(piece and len(piece) <= 10 for piece in pieces)
        assert "".join(pieces) == text


# ── Interruption latch ───────────────────────────────────────────────────


class TestSpeechInterruptedLatch:
    def test_take_pops_and_reports_recent_barge(self):
        ts.mark_speech_interrupted()
        assert ts.take_speech_interrupted() is True
        assert ts.take_speech_interrupted() is False  # one-shot


    def test_stale_barge_expires(self, monkeypatch):
        ts.mark_speech_interrupted()
        at = ts._interrupted_at
        monkeypatch.setattr(ts.time, "monotonic", lambda: at + ts._INTERRUPT_TTL_S + 1)
        assert ts.take_speech_interrupted() is False


# ── Registry + resolver ──────────────────────────────────────────────────


def _register_fake(monkeypatch, name, available=True, chunks=(b"\x00\x00",)):
    class _Fake(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return available

        def stream(self, text):
            yield from chunks

    monkeypatch.setitem(ts._REGISTRY, name, _Fake)
    return _Fake


def test_resolve_returns_configured_streamer(monkeypatch):
    _register_fake(monkeypatch, "faketts")
    prov = ts.resolve_streaming_provider({"provider": "faketts"})
    assert isinstance(prov, ts.StreamingTTSProvider)


def test_explicit_streaming_pin_exposes_normalized_resolved_identity(monkeypatch):
    _register_fake(monkeypatch, "stream-b")

    prov = ts.resolve_streaming_provider(
        {
            "provider": "sync-a",
            "streaming": {"provider": " STREAM-B "},
        }
    )

    assert prov is not None
    assert prov.provider_id == "stream-b"


def test_never_swaps_provider_for_streaming(monkeypatch):
    # A registered streamer must NOT be substituted when the user picked another
    # (non-streaming) provider — that would silently change their voice.
    _register_fake(monkeypatch, "elevenlabs")
    assert ts.resolve_streaming_provider({"provider": "edge"}) is None


# ── Built-in provider availability ───────────────────────────────────────


def test_elevenlabs_available_reflects_key(monkeypatch):
    # Key lookups now route through the provider-secret resolver
    # (config > env/.env > credential pool), not bare get_env_value.
    monkeypatch.setattr(ts, "_resolve_key", lambda env, pid: "key" if env == "ELEVENLABS_API_KEY" else "")
    assert ts.ElevenLabsStreamer.available() is True
    monkeypatch.setattr(ts, "_resolve_key", lambda env, pid: "")
    assert ts.ElevenLabsStreamer.available() is False


def test_openai_available_reflects_audio_key_resolution(monkeypatch):
    monkeypatch.setattr(ts, "_openai_config_api_key", lambda: "")
    monkeypatch.setattr(ts, "resolve_openai_audio_api_key", lambda: "voice-key")
    assert ts.OpenAIStreamer.available() is True
    monkeypatch.setattr(ts, "resolve_openai_audio_api_key", lambda: "")
    assert ts.OpenAIStreamer.available() is False
    # tts.openai.api_key from config.yaml counts too
    monkeypatch.setattr(ts, "_openai_config_api_key", lambda: "cfg-key")
    assert ts.OpenAIStreamer.available() is True


def test_openai_streamer_prefers_configured_api_key(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_bytes(self):
            yield b"\x01\x00"

    class _StreamingCreate:
        @staticmethod
        def create(**kwargs):
            return _Response()

    class _OpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.audio = MagicMock()
            self.audio.speech.with_streaming_response = _StreamingCreate()

    monkeypatch.setattr(ts, "resolve_openai_audio_api_key", lambda: "env-key")
    monkeypatch.setattr(ts, "get_env_value", lambda key, *args: None)
    monkeypatch.setattr("openai.OpenAI", _OpenAI)

    config = {
        "provider": "openai",
        "openai": {"api_key": "cfg-key", "base_url": "http://local-tts.example/v1"},
    }
    streamer = ts.resolve_streaming_provider(config)

    assert streamer is not None
    assert list(streamer.stream("Streaming test.")) == [b"\x01\x00"]
    assert captured["client"]["api_key"] == "cfg-key"


_MISSING = object()


@pytest.mark.parametrize(
    ("streaming_model_id", "expected_model", "expected_cap"),
    [
        pytest.param(_MISSING, "eleven_multilingual_v2", 10000, id="missing"),
        pytest.param(None, "eleven_multilingual_v2", 10000, id="null"),
        pytest.param("", "eleven_multilingual_v2", 10000, id="empty"),
        pytest.param("   ", "eleven_multilingual_v2", 10000, id="whitespace"),
        pytest.param("eleven_v3", "eleven_v3", 5000, id="explicit"),
    ],
)
def test_elevenlabs_stream_model_and_cap_share_nonempty_precedence(
    monkeypatch,
    streaming_model_id,
    expected_model,
    expected_cap,
):
    captured_models = []

    class _TextToSpeech:
        @staticmethod
        def convert(**kwargs):
            captured_models.append(kwargs["model_id"])
            return iter([b"\x00\x00"])

    class _ElevenLabs:
        def __init__(self, **_kwargs):
            self.text_to_speech = _TextToSpeech()

    section = {"model_id": "eleven_multilingual_v2"}
    if streaming_model_id is not _MISSING:
        section["streaming_model_id"] = streaming_model_id
    config = {"provider": "elevenlabs", "elevenlabs": section}

    monkeypatch.setattr(ts, "_resolve_key", lambda *_args: "test-key")
    monkeypatch.setattr("tools.tts_tool._import_elevenlabs", lambda: _ElevenLabs)

    streamer = ts.resolve_streaming_provider(config)

    assert streamer is not None
    assert list(streamer.stream("Model and cap contract.")) == [b"\x00\x00"]
    assert ts.resolve_streaming_text_limit(streamer, config) == expected_cap
    assert captured_models == [expected_model]


# ── Dispatch: chunked streamer path ──────────────────────────────────────


def _drain_queue(sentences):
    q = queue.Queue()
    for s in sentences:
        q.put(s)
    q.put(None)
    return q


def _sd_mock():
    sd = MagicMock()
    out = MagicMock()
    sd.OutputStream.return_value = out
    return sd, out


class _RecordingStreamer:
    provider_id = "stream-b"
    sample_rate = 24000
    channels = 1

    def __init__(self, *, fail_on_request=None, stop_event=None, stop_after_request=None):
        self.fail_on_request = fail_on_request
        self.stop_event = stop_event
        self.stop_after_request = stop_after_request
        self.requests: list[str] = []

    def stream(self, text):
        self.requests.append(text)
        request_number = len(self.requests)
        if request_number == self.fail_on_request:
            raise RuntimeError(f"request {request_number} failed")
        yield bytes((request_number, 0))
        if request_number == self.stop_after_request and self.stop_event is not None:
            self.stop_event.set()


class _PreAudioFailureStreamer:
    provider_id = "stream-b"
    sample_rate = 24000
    channels = 1

    def __init__(self, mode, *, stop_event=None):
        self.mode = mode
        self.stop_event = stop_event
        self.requests: list[str] = []

    def stream(self, text):
        self.requests.append(text)
        if self.mode == "stop":
            assert self.stop_event is not None
            self.stop_event.set()
        if self.mode == "raise":
            raise RuntimeError("provider failed before yielding audio")
        return
        yield b"unreachable"


def _patch_cli_streamer(monkeypatch, streamer, config, *, platform_name="Linux"):
    resolution_calls = []

    def _resolve(resolved_config, preferred=None):
        resolution_calls.append((resolved_config, preferred))
        return streamer

    monkeypatch.setattr(ts, "resolve_streaming_provider", _resolve)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: config)
    monkeypatch.setattr(tts_tool, "_strip_markdown_for_tts", lambda text: text)
    monkeypatch.setattr(tts_tool.platform, "system", lambda: platform_name)
    return resolution_calls


def test_cli_streaming_uses_active_cap_losslessly_with_one_output_and_activity_interval(
    monkeypatch,
):
    text = "x" * 40
    config = {
        "provider": "sync-a",
        "sync-a": {"max_text_length": 50},
        "stream-b": {"max_text_length": 17},
        "streaming": {"provider": "auto"},
    }
    streamer = _RecordingStreamer()
    resolution_calls = _patch_cli_streamer(monkeypatch, streamer, config)
    sd, output = _sd_mock()
    monkeypatch.setattr(tts_tool, "_import_sounddevice", lambda: sd)
    activity = []
    monkeypatch.setattr("tools.voice_mode.mark_audio_output_active", activity.append)
    displayed = []
    stop_event = threading.Event()
    done_event = threading.Event()

    tts_tool.stream_tts_to_speaker(
        _drain_queue([text]),
        stop_event,
        done_event,
        display_callback=displayed.append,
        provider="preferred-a",
    )

    assert resolution_calls == [(config, "preferred-a")]
    assert [len(request) for request in streamer.requests] == [17, 17, 6]
    assert "".join(streamer.requests) == text
    assert displayed == [text]
    assert sd.OutputStream.call_count == 1
    output.start.assert_called_once_with()
    assert output.write.call_count == 3
    output.stop.assert_called_once_with()
    output.close.assert_called_once_with()
    assert activity == [True, False]
    assert done_event.is_set()


@pytest.mark.parametrize(
    ("failure_mode", "platform_name"),
    [
        pytest.param("zero-yield", "Linux", id="zero-yield-device"),
        pytest.param("zero-yield", "Darwin", id="zero-yield-tempfile"),
        pytest.param("raise", "Linux", id="provider-error"),
    ],
)
def test_cli_streaming_preaudio_failure_retries_losslessly_via_real_sync_dispatcher(
    monkeypatch,
    failure_mode,
    platform_name,
):
    text = "p" * 40
    config = {
        "provider": "edge",
        "edge": {"max_text_length": 17},
        "stream-b": {"max_text_length": 17},
    }
    streamer = _PreAudioFailureStreamer(failure_mode)
    _patch_cli_streamer(monkeypatch, streamer, config, platform_name=platform_name)
    sd, output = _sd_mock()
    monkeypatch.setattr(tts_tool, "_import_sounddevice", lambda: sd)
    monkeypatch.setattr("tools.voice_mode.mark_audio_output_active", lambda _active: None)
    sync_requests = []
    played_audio = []

    async def _fake_edge_generate(text, output_path, _tts_config):
        sync_requests.append(text)
        Path(output_path).write_bytes(text.encode())
        return output_path

    def _capture_playback(path):
        played_audio.append(Path(path).read_bytes())
        return True

    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", _fake_edge_generate)
    monkeypatch.setattr("tools.voice_mode.play_audio_file", _capture_playback)
    done_event = threading.Event()

    played = tts_tool.stream_tts_to_speaker(
        _drain_queue([text]),
        threading.Event(),
        done_event,
    )

    assert [len(request) for request in streamer.requests] == [17]
    assert output.write.call_count == 0
    assert [len(request) for request in sync_requests] == [17, 17, 6]
    assert "".join(sync_requests) == text
    assert b"".join(played_audio) == text.encode()
    assert played is True
    assert done_event.is_set()


@pytest.mark.parametrize(
    "provider_override",
    [
        pytest.param(_MISSING, id="missing"),
        pytest.param(None, id="none"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
    ],
)
@pytest.mark.parametrize(
    "configured_provider",
    [
        pytest.param("edge", id="configured-edge"),
        pytest.param("   ", id="configured-whitespace"),
    ],
)
def test_sync_dispatch_normalizes_empty_provider_before_cap_resolution(
    monkeypatch,
    provider_override,
    configured_provider,
):
    text = "n" * 40
    config = {
        "provider": configured_provider,
        "edge": {"max_text_length": 17},
    }
    resolution_calls = _patch_cli_streamer(monkeypatch, None, config)
    generated: list[str] = []
    played: list[bytes] = []

    async def _fake_edge_generate(text, output_path, _tts_config):
        generated.append(text)
        Path(output_path).write_bytes(text.encode())
        return output_path

    def _capture_playback(path):
        played.append(Path(path).read_bytes())
        return True

    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", _fake_edge_generate)
    monkeypatch.setattr("tools.voice_mode.play_audio_file", _capture_playback)
    done_event = threading.Event()
    kwargs = {}
    if provider_override is not _MISSING:
        kwargs["provider"] = provider_override

    played_audio = tts_tool.stream_tts_to_speaker(
        _drain_queue([text]),
        threading.Event(),
        done_event,
        **kwargs,
    )

    assert resolution_calls == [(config, None)]
    assert [len(piece) for piece in generated] == [17, 17, 6]
    assert "".join(generated) == text
    assert b"".join(played) == text.encode()
    assert played_audio is True
    assert done_event.is_set()


def test_sync_fallback_plays_returned_artifact_and_cleans_every_owned_path(
    monkeypatch,
    tmp_path,
):
    config = {"provider": "edge", "edge": {"max_text_length": 4000}}
    _patch_cli_streamer(monkeypatch, None, config)
    returned_path = tmp_path / "provider-output.wav"
    requested_paths = []
    providers = []
    played_paths = []

    def _alternate_output(*, text, output_path, provider):
        requested_paths.append(Path(output_path))
        providers.append(provider)
        returned_path.write_bytes(text.encode())
        return json.dumps(
            {"success": True, "file_path": str(returned_path)},
        )

    def _play(path):
        played_paths.append(Path(path))
        return True

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _alternate_output)
    monkeypatch.setattr("tools.voice_mode.play_audio_file", _play)
    done_event = threading.Event()

    played = tts_tool.stream_tts_to_speaker(
        _drain_queue(["Play the provider-returned artifact."]),
        threading.Event(),
        done_event,
    )

    assert played is True
    assert providers == ["edge"]
    assert played_paths == [returned_path]
    assert requested_paths and all(not path.exists() for path in requested_paths)
    assert not returned_path.exists()
    assert done_event.is_set()


def test_sync_fallback_prefers_valid_requested_mp3_over_returned_ogg(
    monkeypatch,
    tmp_path,
):
    config = {"provider": "edge", "edge": {"max_text_length": 4000}}
    _patch_cli_streamer(monkeypatch, None, config)
    returned_path = tmp_path / "converted-output.ogg"
    requested_paths = []
    playback_attempts = []

    def _dual_output(*, text, output_path, provider):
        requested_path = Path(output_path)
        requested_paths.append(requested_path)
        requested_path.write_bytes(b"valid requested mp3")
        returned_path.write_bytes(b"valid returned ogg")
        return json.dumps({"success": True, "file_path": str(returned_path)})

    def _play(path):
        audio_path = Path(path)
        playback_attempts.append(audio_path)
        return audio_path.suffix == ".mp3"

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _dual_output)
    monkeypatch.setattr("tools.voice_mode.play_audio_file", _play)
    done_event = threading.Event()

    played = tts_tool.stream_tts_to_speaker(
        _drain_queue(["Prefer the voice-compatible requested MP3."]),
        threading.Event(),
        done_event,
    )

    assert played is True
    assert playback_attempts == requested_paths
    assert requested_paths and all(not path.exists() for path in requested_paths)
    assert not returned_path.exists()
    assert done_event.is_set()


@pytest.mark.parametrize("requested_failure", ["false", "raise"])
def test_sync_fallback_does_not_retry_after_ambiguous_playback_failure(
    monkeypatch,
    tmp_path,
    requested_failure,
):
    config = {"provider": "edge", "edge": {"max_text_length": 4000}}
    _patch_cli_streamer(monkeypatch, None, config)
    returned_path = tmp_path / "converted-output.ogg"
    requested_paths = []
    playback_attempts = []

    def _dual_output(*, text, output_path, provider):
        requested_path = Path(output_path)
        requested_paths.append(requested_path)
        requested_path.write_bytes(b"unplayable requested mp3")
        returned_path.write_bytes(b"valid returned ogg")
        return json.dumps({"success": True, "file_path": str(returned_path)})

    def _play(path):
        audio_path = Path(path)
        playback_attempts.append(audio_path)
        if audio_path.suffix == ".mp3":
            if requested_failure == "raise":
                raise RuntimeError("requested artifact cannot play")
            return False
        return True

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _dual_output)
    monkeypatch.setattr("tools.voice_mode.play_audio_file", _play)
    done_event = threading.Event()

    played = tts_tool.stream_tts_to_speaker(
        _drain_queue(["Retry the provider-returned artifact."]),
        threading.Event(),
        done_event,
    )

    assert played is False
    assert playback_attempts == [requested_paths[0]]
    assert requested_paths and all(not path.exists() for path in requested_paths)
    assert not returned_path.exists()
    assert done_event.is_set()


def test_sync_fallback_rejects_tool_failure_even_if_partial_file_exists(monkeypatch):
    config = {"provider": "edge", "edge": {"max_text_length": 4000}}
    _patch_cli_streamer(monkeypatch, None, config)
    requested_paths = []
    played_paths = []

    def _failed_output(*, text, output_path, provider):
        path = Path(output_path)
        requested_paths.append(path)
        path.write_bytes(b"partial audio")
        return json.dumps(
            {"success": False, "file_path": str(path), "error": "failed"},
        )

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _failed_output)
    monkeypatch.setattr(
        "tools.voice_mode.play_audio_file",
        lambda path: played_paths.append(Path(path)) or True,
    )
    done_event = threading.Event()

    played = tts_tool.stream_tts_to_speaker(
        _drain_queue(["Reject partial failed output."]),
        threading.Event(),
        done_event,
    )

    assert played is False
    assert played_paths == []
    assert requested_paths and all(not path.exists() for path in requested_paths)
    assert done_event.is_set()


def test_sync_fallback_reports_false_when_generated_audio_cannot_play(monkeypatch):
    config = {"provider": "edge", "edge": {"max_text_length": 4000}}
    _patch_cli_streamer(monkeypatch, None, config)
    requested_paths = []

    def _successful_output(*, text, output_path, provider):
        path = Path(output_path)
        requested_paths.append(path)
        path.write_bytes(b"generated audio")
        return json.dumps({"success": True, "file_path": str(path)})

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _successful_output)
    monkeypatch.setattr("tools.voice_mode.play_audio_file", lambda _path: False)
    done_event = threading.Event()

    played = tts_tool.stream_tts_to_speaker(
        _drain_queue(["Playback fails after generation."]),
        threading.Event(),
        done_event,
    )

    assert played is False
    assert requested_paths and all(not path.exists() for path in requested_paths)
    assert done_event.is_set()


def test_cli_streaming_stop_before_audio_does_not_fallback(monkeypatch):
    text = "Stop before audio."
    stop_event = threading.Event()
    config = {"provider": "stream-b", "stream-b": {"max_text_length": 17}}
    streamer = _PreAudioFailureStreamer("stop", stop_event=stop_event)
    _patch_cli_streamer(monkeypatch, streamer, config)
    sd, output = _sd_mock()
    monkeypatch.setattr(tts_tool, "_import_sounddevice", lambda: sd)
    monkeypatch.setattr("tools.voice_mode.mark_audio_output_active", lambda _active: None)
    sync_requests = []
    monkeypatch.setattr(
        tts_tool,
        "text_to_speech_tool",
        lambda **kwargs: sync_requests.append(kwargs["text"]),
    )
    done_event = threading.Event()

    played = tts_tool.stream_tts_to_speaker(
        _drain_queue([text]),
        stop_event,
        done_event,
    )

    assert len(streamer.requests) == 1
    assert streamer.requests[0]
    assert output.write.call_count == 0
    assert sync_requests == []
    assert played is True
    assert done_event.is_set()


def test_cli_streaming_provider_error_stops_later_pieces_and_releases_ownership(
    monkeypatch,
):
    text = "y" * 40
    config = {
        "provider": "sync-a",
        "sync-a": {"max_text_length": 50},
        "stream-b": {"max_text_length": 17},
    }
    streamer = _RecordingStreamer(fail_on_request=2)
    _patch_cli_streamer(monkeypatch, streamer, config)
    sd, output = _sd_mock()
    monkeypatch.setattr(tts_tool, "_import_sounddevice", lambda: sd)
    activity = []
    monkeypatch.setattr("tools.voice_mode.mark_audio_output_active", activity.append)
    sync_requests = []
    monkeypatch.setattr(
        tts_tool,
        "text_to_speech_tool",
        lambda **kwargs: sync_requests.append(kwargs["text"]),
    )
    done_event = threading.Event()

    tts_tool.stream_tts_to_speaker(
        _drain_queue([text]),
        threading.Event(),
        done_event,
    )

    assert streamer.requests == [text[:17], text[17:34]]
    assert output.write.call_count == 1
    assert sync_requests == []
    assert activity == [True, False]
    output.stop.assert_called_once_with()
    output.close.assert_called_once_with()
    assert done_event.is_set()


def test_cli_streaming_stop_between_pieces_prevents_later_requests(monkeypatch):
    text = "z" * 40
    config = {"provider": "stream-b", "stream-b": {"max_text_length": 17}}
    stop_event = threading.Event()
    streamer = _RecordingStreamer(
        stop_event=stop_event,
        stop_after_request=1,
    )
    _patch_cli_streamer(monkeypatch, streamer, config)
    sd, output = _sd_mock()
    monkeypatch.setattr(tts_tool, "_import_sounddevice", lambda: sd)
    monkeypatch.setattr("tools.voice_mode.mark_audio_output_active", lambda _active: None)
    done_event = threading.Event()

    tts_tool.stream_tts_to_speaker(
        _drain_queue([text]),
        stop_event,
        done_event,
    )

    assert streamer.requests == [text[:17]]
    assert output.write.call_count == 1
    assert done_event.is_set()


def test_cli_tempfile_fallback_chains_all_capped_pieces_into_one_playback(
    monkeypatch,
):
    import wave

    text = "q" * 40
    config = {
        "provider": "sync-a",
        "sync-a": {"max_text_length": 50},
        "stream-b": {"max_text_length": 17},
    }
    streamer = _RecordingStreamer()
    _patch_cli_streamer(monkeypatch, streamer, config, platform_name="Darwin")
    played_audio = []

    def _capture_playback(path):
        with wave.open(path, "rb") as wav_file:
            played_audio.append(wav_file.readframes(wav_file.getnframes()))
        return True

    monkeypatch.setattr("tools.voice_mode.play_audio_file", _capture_playback)
    done_event = threading.Event()

    tts_tool.stream_tts_to_speaker(
        _drain_queue([text]),
        threading.Event(),
        done_event,
    )

    assert [len(request) for request in streamer.requests] == [17, 17, 6]
    assert "".join(streamer.requests) == text
    assert played_audio == [b"\x01\x00\x02\x00\x03\x00"]
    assert done_event.is_set()


@pytest.mark.parametrize("playback_failure", ["false", "raise"])
def test_cli_tempfile_playback_failure_does_not_replay_sentence(
    monkeypatch,
    playback_failure,
):
    text = "Retry playback."
    config = {"provider": "stream-b", "stream-b": {"max_text_length": 4000}}
    streamer = _RecordingStreamer()
    _patch_cli_streamer(monkeypatch, streamer, config, platform_name="Darwin")
    playback_suffixes = []
    sync_requests = []

    def _capture_playback(path):
        playback_suffixes.append(Path(path).suffix)
        if len(playback_suffixes) == 1:
            if playback_failure == "raise":
                raise RuntimeError("player failed after an ambiguous terminal outcome")
            return False
        return True

    def _sync_fallback(*, text, output_path, provider):
        sync_requests.append(text)
        Path(output_path).write_bytes(b"sync audio")
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr("tools.voice_mode.play_audio_file", _capture_playback)
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _sync_fallback)
    done_event = threading.Event()

    played = tts_tool.stream_tts_to_speaker(
        _drain_queue([text]),
        threading.Event(),
        done_event,
    )

    assert played is False
    assert streamer.requests == [text]
    assert sync_requests == []
    assert playback_suffixes == [".wav"]
    assert done_event.is_set()


# ── Dispatch: universal per-sentence sync fallback ───────────────────────


# ── tts.streaming.provider config knob (salvaged from PR #47588) ─────────


# ── Credential routing: resolve_provider_secret, never bare env ──────────


def test_elevenlabs_available_routes_through_secret_resolver(monkeypatch):
    calls = []

    def _fake_resolve(env_var, provider_id):
        calls.append((env_var, provider_id))
        return "pool-key"

    monkeypatch.setattr(ts, "_resolve_key", _fake_resolve)
    assert ts.ElevenLabsStreamer.available() is True
    assert ("ELEVENLABS_API_KEY", "elevenlabs") in calls


def test_xai_available_uses_oauth_credential_resolver(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("tools.xai_http")
    fake.resolve_xai_http_credentials = lambda: {"api_key": "xai-key"}
    monkeypatch.setitem(sys.modules, "tools.xai_http", fake)
    assert ts.XAIStreamer.available() is True
    fake.resolve_xai_http_credentials = lambda: {"api_key": ""}
    assert ts.XAIStreamer.available() is False


# ── Gemini SSE parsing ────────────────────────────────────────────────────


# ── xAI WebSocket bridge ─────────────────────────────────────────────────


# ── 16 MiB per-sentence stream cap ───────────────────────────────────────


def test_stream_cap_truncates_runaway_upstream(monkeypatch):
    monkeypatch.setattr(ts, "_STREAM_SENTENCE_BYTE_CAP", 100)

    def _endless():
        while True:
            yield b"\x00" * 64

    out = list(ts._capped(_endless(), "test"))
    assert len(out) == 1  # 64 ok, 128 > cap → stop
    assert sum(len(c) for c in out) <= 100
