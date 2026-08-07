"""Contract tests for provider-neutral streaming speech-to-text."""

import asyncio
import base64
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from tools.transcription_tools import (
    _resolve_openai_audio_client_config,
    resolve_stt_provider_config,
)
from tools.stt_streaming import (
    ElevenLabsStreamingSTTProvider,
    ElevenLabsStreamingSTTSession,
    OpenAIStreamingSTTProvider,
    OpenAIStreamingSTTSession,
    StreamingSTTCoordinator,
    StreamingSTTProvider,
    StreamingSTTSession,
    StreamingSTTTurnController,
    StreamingTranscriptEvent,
    _PCM16StreamResampler,
    resolve_streaming_stt_provider,
)


class _FakeWebSocket:
    def __init__(self, incoming=None):
        self.incoming = asyncio.Queue()
        for message in incoming or []:
            self.incoming.put_nowait(json.dumps(message))
        self.sent = []
        self.closed = False

    async def send(self, payload):
        self.sent.append(payload)

    async def recv(self):
        return await self.incoming.get()

    async def close(self):
        self.closed = True


def _event(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _event(item) for key, item in value.items()})
    return value


class _FakeOpenAIConnection:
    def __init__(self, incoming=None):
        self.incoming = asyncio.Queue()
        for event in incoming or []:
            self.incoming.put_nowait(_event(event))
        self.input_audio_buffer = SimpleNamespace(
            append=AsyncMock(),
            commit=AsyncMock(),
        )
        self.session = SimpleNamespace(update=AsyncMock())
        self.close = AsyncMock()

    async def recv(self):
        return await self.incoming.get()


class _FakeOpenAIClient:
    def __init__(self, connection):
        manager = SimpleNamespace(enter=AsyncMock(return_value=connection))
        self.realtime = SimpleNamespace(connect=MagicMock(return_value=manager))
        self.close = AsyncMock()


@pytest.fixture
def provider_wire_events():
    """Sanitized provider events validated against the live APIs."""
    return {
        "openai": [
            {"type": "session.updated"},
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "delta": "Hermes streaming",
                "event_id": "evt-openai-delta",
                "item_id": "item-openai",
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "Hermes streaming speech test alpha seven.",
                "event_id": "evt-openai-completed",
                "item_id": "item-openai",
            },
        ],
        "elevenlabs": [
            {"message_type": "session_started", "session_id": "session-elevenlabs"},
            {"message_type": "partial_transcript", "text": "Hermes streaming"},
            {
                "message_type": "committed_transcript",
                "text": "Hermes streaming speech test alpha seven.",
            },
        ],
    }


class TestProviderResolution:
    @pytest.mark.parametrize(
        "provider_name",
        ["local", "groq", "mistral", "xai", "plugin-provider"],
    )
    def test_unsupported_providers_stay_batch(self, provider_name):
        assert (
            resolve_streaming_stt_provider({
                "provider": provider_name,
            })
            is None
        )

    def test_xai_stays_batch_when_credentialed(self):
        assert (
            resolve_streaming_stt_provider({
                "provider": "xai",
                "xai": {"api_key": "configured-xai-key"},
            })
            is None
        )

    def test_disabled_streaming_stays_batch(self):
        assert (
            resolve_streaming_stt_provider({
                "provider": "openai",
                "openai": {"streaming": False},
            })
            is None
        )

    def test_master_stt_switch_disables_streaming(self):
        assert (
            resolve_streaming_stt_provider({
                "enabled": False,
                "provider": "openai",
                "openai": {
                    "api_key": "key",
                    "streaming_model_id": "configured-model",
                },
            })
            is None
        )

    def test_switching_to_unsupported_provider_disables_streaming(self):
        openai_config = {
            "provider": "openai",
            "openai": {
                "api_key": "key",
                "streaming_model_id": "configured-model",
            },
        }
        local_config = {**openai_config, "provider": "local"}

        assert isinstance(
            resolve_streaming_stt_provider(openai_config),
            OpenAIStreamingSTTProvider,
        )
        assert resolve_streaming_stt_provider(local_config) is None

    def test_openai_batch_and_streaming_share_resolved_connection(self):
        config = {
            "provider": "openai",
            "openai": {
                "api_key": "shared-key",
                "base_url": "https://compatible.example/v1",
                "model": "batch-model",
                "streaming_model_id": "streaming-model",
            },
        }

        resolved = resolve_stt_provider_config("openai", config)
        streaming = resolve_streaming_stt_provider(config)
        with patch("tools.transcription_tools._load_stt_config", return_value=config):
            batch_key, batch_url = _resolve_openai_audio_client_config()

        assert resolved is not None
        assert isinstance(streaming, OpenAIStreamingSTTProvider)
        assert (batch_key, batch_url) == (resolved.api_key, resolved.base_url)
        assert (streaming.api_key, streaming.base_url) == (
            resolved.api_key,
            resolved.base_url,
        )

    def test_openai_requires_direct_profile_scoped_key(self):
        config = {
            "provider": "openai",
            "openai": {"streaming_model_id": "gpt-4o-mini-transcribe"},
        }
        with patch("tools.transcription_tools.resolve_openai_audio_api_key", return_value=""):
            assert resolve_streaming_stt_provider(config) is None
        with patch(
            "tools.transcription_tools.resolve_openai_audio_api_key",
            return_value="profile-key",
        ):
            provider = resolve_streaming_stt_provider(config)
        assert isinstance(provider, OpenAIStreamingSTTProvider)
        assert provider.api_key == "profile-key"

    @pytest.mark.parametrize(
        "config, env_url",
        [
            (
                {
                    "provider": "openai",
                    "openai": {"base_url": "https://compatible.example/v1"},
                },
                "",
            ),
            (
                {"provider": "openai"},
                "https://compatible.example/v1",
            ),
        ],
    )
    def test_custom_openai_endpoint_is_passed_to_realtime_client(self, config, env_url):
        config.setdefault("openai", {})["streaming_model_id"] = "configured-model"
        with (
            patch(
                "tools.transcription_tools.get_env_value",
                return_value=env_url,
            ),
            patch(
                "tools.transcription_tools.resolve_openai_audio_api_key",
                return_value="key",
            ),
        ):
            provider = resolve_streaming_stt_provider(config)
        assert isinstance(provider, OpenAIStreamingSTTProvider)
        assert provider.base_url == "https://compatible.example/v1"
        assert provider.model == "configured-model"

    def test_openai_accepts_profile_config_key_for_official_realtime(self):
        with patch("tools.transcription_tools.get_env_value", return_value=""):
            provider = resolve_streaming_stt_provider({
                "provider": "openai",
                "openai": {
                    "api_key": "profile-config-key",
                    "streaming_model_id": "configured-model",
                },
            })
        assert isinstance(provider, OpenAIStreamingSTTProvider)
        assert provider.api_key == "profile-config-key"

    def test_elevenlabs_uses_config_key_and_custom_endpoint(self):
        config = {
            "provider": "elevenlabs",
            "elevenlabs": {
                "api_key": "config-key",
                "base_url": "https://compatible.example/v1",
                "streaming_model_id": "configured-model",
            },
        }
        provider = resolve_streaming_stt_provider(config)
        assert isinstance(provider, ElevenLabsStreamingSTTProvider)
        assert provider.api_key == "config-key"
        assert provider.base_url == "https://compatible.example/v1"

    def test_provider_language_uses_shared_stt_precedence(self):
        openai_config = {
            "provider": "openai",
            "language": "fr",
            "openai": {
                "language": "",
                "streaming_model_id": "configured-model",
            },
        }
        with patch(
            "tools.transcription_tools.resolve_openai_audio_api_key", return_value="key"
        ):
            openai = resolve_streaming_stt_provider(openai_config)

        elevenlabs_config = {
            "provider": "elevenlabs",
            "language": "fr",
            "elevenlabs": {
                "language_code": "eng",
                "streaming_model_id": "configured-model",
            },
        }
        with patch(
            "tools.transcription_tools._resolve_provider_key", return_value="key"
        ):
            elevenlabs = resolve_streaming_stt_provider(elevenlabs_config)

        assert openai.language == "fr"
        assert elevenlabs.language == "eng"

    def test_real_profile_config_selects_streaming_provider(
        self, tmp_path, monkeypatch
    ):
        hermes_home = tmp_path / "profile"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "stt:\n"
            "  provider: openai\n"
            "  openai:\n"
            "    streaming_model_id: gpt-4o-mini-transcribe\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch(
            "tools.transcription_tools.resolve_openai_audio_api_key",
            return_value="profile-key",
        ):
            provider = resolve_streaming_stt_provider()
        assert isinstance(provider, OpenAIStreamingSTTProvider)

    @pytest.mark.parametrize(
        "provider_name,key_name,key_value,provider_type",
        [
            (
                "openai",
                "VOICE_TOOLS_OPENAI_KEY",
                "profile-openai-key",
                OpenAIStreamingSTTProvider,
            ),
            (
                "elevenlabs",
                "ELEVENLABS_API_KEY",
                "profile-elevenlabs-key",
                ElevenLabsStreamingSTTProvider,
            ),
        ],
    )
    def test_profile_dotenv_selects_each_supported_provider(
        self,
        tmp_path,
        monkeypatch,
        provider_name,
        key_name,
        key_value,
        provider_type,
    ):
        hermes_home = tmp_path / provider_name
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            f"stt:\n  provider: {provider_name}\n",
            encoding="utf-8",
        )
        (hermes_home / ".env").write_text(
            f"{key_name}={key_value}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        provider = resolve_streaming_stt_provider()

        assert isinstance(provider, provider_type)
        assert provider.api_key == key_value


class TestProviderProtocols:
    def test_openai_configures_transcription_only_semantic_vad(
        self, provider_wire_events
    ):
        async def _test():
            connection = _FakeOpenAIConnection(provider_wire_events["openai"][:1])
            client = _FakeOpenAIClient(connection)
            provider = OpenAIStreamingSTTProvider(
                "key",
                "https://openai-compatible.example/v1",
                "configured-model",
                "en",
            )
            with patch("openai.AsyncOpenAI", return_value=client) as openai_client:
                session = await provider.open_session()
            openai_client.assert_called_once_with(
                api_key="key",
                base_url="https://openai-compatible.example/v1",
            )
            client.realtime.connect.assert_called_once_with(
                extra_query={"intent": "transcription"},
            )
            update = connection.session.update.await_args.kwargs["session"]
            assert update["type"] == "transcription"
            audio = update["audio"]["input"]
            assert audio["format"] == {"type": "audio/pcm", "rate": 24000}
            assert audio["turn_detection"] == {
                "type": "semantic_vad",
                "eagerness": "auto",
            }
            assert audio["transcription"]["model"] == "configured-model"
            assert audio["transcription"]["language"] == "en"
            assert isinstance(session, OpenAIStreamingSTTSession)

        asyncio.run(_test())

    def test_openai_normalizes_delta_and_completed_events(self, provider_wire_events):
        async def _test():
            connection = _FakeOpenAIConnection(provider_wire_events["openai"][1:])
            client = _FakeOpenAIClient(connection)
            session = OpenAIStreamingSTTSession(client, connection)
            partial = await session.receive_event()
            final = await session.receive_event()
            assert partial == StreamingTranscriptEvent(
                "Hermes streaming",
                False,
                "evt-openai-delta",
                "item-openai",
            )
            assert final == StreamingTranscriptEvent(
                "Hermes streaming speech test alpha seven.",
                True,
                "evt-openai-completed",
                "item-openai",
            )

        asyncio.run(_test())

    def test_openai_transcription_failure_is_terminal(self):
        async def _test():
            connection = _FakeOpenAIConnection([
                {
                    "type": "conversation.item.input_audio_transcription.failed",
                    "event_id": "evt-failed",
                    "item_id": "item-failed",
                    "error": {"message": "audio could not be transcribed"},
                }
            ])
            session = OpenAIStreamingSTTSession(
                _FakeOpenAIClient(connection),
                connection,
            )
            with pytest.raises(RuntimeError, match="could not be transcribed"):
                await session.receive_event()

        asyncio.run(_test())

    def test_openai_empty_manual_commit_error_keeps_final_in_flight(self):
        async def _test():
            connection = _FakeOpenAIConnection([
                {
                    "type": "error",
                    "error": {
                        "message": (
                            "Error committing input audio buffer: buffer too small. "
                            "Expected at least 100ms of audio, but buffer only has "
                            "0.00ms of audio."
                        )
                    },
                },
                {
                    "type": (
                        "conversation.item.input_audio_transcription.completed"
                    ),
                    "event_id": "evt-final",
                    "item_id": "item-final",
                    "transcript": "already committed by semantic VAD",
                },
            ])
            session = OpenAIStreamingSTTSession(
                _FakeOpenAIClient(connection),
                connection,
            )

            assert await session.receive_event() is None
            assert await session.receive_event() == StreamingTranscriptEvent(
                "already committed by semantic VAD",
                True,
                "evt-final",
                "item-final",
            )

        asyncio.run(_test())

    def test_openai_uses_sdk_audio_buffer_and_closes_client(self):
        async def _test():
            connection = _FakeOpenAIConnection()
            client = _FakeOpenAIClient(connection)
            session = OpenAIStreamingSTTSession(client, connection)
            await session.send_audio(b"\x01\x02\x03\x04")
            await session.commit()
            await session.close()
            connection.input_audio_buffer.append.assert_awaited_once_with(
                audio=base64.b64encode(b"\x01\x02\x03\x04").decode("ascii")
            )
            connection.input_audio_buffer.commit.assert_awaited_once_with()
            connection.close.assert_awaited_once_with()
            client.close.assert_awaited_once_with()

        asyncio.run(_test())

    def test_elevenlabs_marks_commits_for_vad_silence_confirmation(
        self, provider_wire_events
    ):
        async def _test():
            websocket = _FakeWebSocket(provider_wire_events["elevenlabs"])
            provider = ElevenLabsStreamingSTTProvider(
                "key",
                "https://elevenlabs-compatible.example/v1",
                {
                    "streaming_model_id": "configured-model",
                    "vad_threshold": 0.4,
                    "vad_silence_threshold_secs": 1.5,
                },
                "eng",
            )
            with patch(
                "tools.stt_streaming._connect", AsyncMock(return_value=websocket)
            ) as connect:
                session = await provider.open_session()
            url = urlsplit(connect.await_args.args[0])
            assert url.scheme == "wss"
            assert url.netloc == "elevenlabs-compatible.example"
            assert url.path == "/v1/speech-to-text/realtime"
            assert parse_qs(url.query) == {
                "model_id": ["configured-model"],
                "audio_format": ["pcm_16000"],
                "commit_strategy": ["vad"],
                "language_code": ["eng"],
                "vad_threshold": ["0.4"],
                "vad_silence_threshold_secs": ["1.5"],
            }
            assert connect.await_args.args[1] == {"xi-api-key": "key"}
            assert await session.receive_event() == StreamingTranscriptEvent(
                "Hermes streaming", False, None, "elevenlabs-0"
            )
            assert await session.receive_event() == StreamingTranscriptEvent(
                "Hermes streaming speech test alpha seven.",
                True,
                "elevenlabs-0",
                "elevenlabs-0",
            )

        asyncio.run(_test())

    @pytest.mark.parametrize(
        "message_type",
        [
            "auth_error",
            "chunk_size_exceeded",
            "commit_throttled",
            "input_error",
            "insufficient_audio_activity",
            "resource_exhausted",
            "session_time_limit_exceeded",
            "transcriber_error",
            "unaccepted_terms",
        ],
    )
    def test_elevenlabs_documented_terminal_events_raise(self, message_type):
        async def _test():
            session = ElevenLabsStreamingSTTSession(
                _FakeWebSocket([
                    {"message_type": message_type, "error": "terminal failure"}
                ])
            )
            with pytest.raises(RuntimeError, match="terminal failure"):
                await session.receive_event()

        asyncio.run(_test())

    @pytest.mark.parametrize(
        "session_type,sample_rate",
        [
            (ElevenLabsStreamingSTTSession, 16000),
        ],
    )
    def test_json_audio_sessions_send_base64_pcm(self, session_type, sample_rate):
        async def _test():
            websocket = _FakeWebSocket()
            session = session_type(websocket)
            assert session.sample_rate == sample_rate
            await session.send_audio(b"\x01\x02\x03\x04")
            message = json.loads(websocket.sent[0])
            assert (
                base64.b64decode(message.get("audio") or message.get("audio_base_64"))
                == b"\x01\x02\x03\x04"
            )

        asyncio.run(_test())

    def test_manual_commits_do_not_close_persistent_sessions(self):
        async def _test():
            openai_connection = _FakeOpenAIConnection()
            openai_client = _FakeOpenAIClient(openai_connection)
            elevenlabs_socket = _FakeWebSocket()
            await OpenAIStreamingSTTSession(openai_client, openai_connection).commit()
            await ElevenLabsStreamingSTTSession(elevenlabs_socket).commit()
            openai_connection.input_audio_buffer.commit.assert_awaited_once_with()
            elevenlabs_commit = json.loads(elevenlabs_socket.sent[0])
            assert elevenlabs_commit["message_type"] == "input_audio_chunk"
            assert elevenlabs_commit["commit"] is True
            assert base64.b64decode(elevenlabs_commit["audio_base_64"]) == b"\0" * 640
            openai_connection.close.assert_not_awaited()
            openai_client.close.assert_not_awaited()
            assert not elevenlabs_socket.closed

        asyncio.run(_test())

class _CoordinatorSession(StreamingSTTSession):
    sample_rate = 16000

    def __init__(self):
        self.events = asyncio.Queue()
        self.audio = []
        self.operations = []
        self.commits = 0
        self.closes = 0

    async def send_audio(self, pcm16):
        self.audio.append(pcm16)
        self.operations.append("audio")

    async def commit(self):
        self.commits += 1
        self.operations.append("commit")

    async def receive_event(self):
        event = await self.events.get()
        if isinstance(event, Exception):
            raise event
        return event

    async def close(self):
        self.closes += 1


class _CoordinatorProvider(StreamingSTTProvider):
    name = "fake"

    def __init__(self):
        self.sessions = []

    async def open_session(self):
        session = _CoordinatorSession()
        self.sessions.append(session)
        return session


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestStreamingSTTCoordinator:
    def test_one_session_serves_two_turns_and_delivers_partials(self):
        np = pytest.importorskip("numpy")
        provider = _CoordinatorProvider()
        finals = []
        errors = []
        coordinator = StreamingSTTCoordinator(
            provider,
            lambda generation, event: finals.append((generation, event.text)),
            lambda generation, error: errors.append((generation, error)),
        )
        assert coordinator.start_utterance(1)
        session = provider.sessions[0]
        coordinator.enqueue_audio(np.ones((480, 1), dtype=np.int16), 48000)
        assert _wait_until(lambda: bool(session.audio))
        session.events.put_nowait(StreamingTranscriptEvent("hel", False))
        assert _wait_until(lambda: finals == [(1, "hel")])
        session.events.put_nowait(StreamingTranscriptEvent("hello", True))
        assert _wait_until(lambda: finals[-1] == (1, "hello"))

        assert coordinator.start_utterance(2)
        assert len(provider.sessions) == 1
        session.events.put_nowait(StreamingTranscriptEvent("stale", True))
        time.sleep(0.05)
        assert finals == [(1, "hel"), (1, "hello")]
        coordinator.enqueue_audio(np.ones((480, 1), dtype=np.int16), 48000)
        assert _wait_until(lambda: len(session.audio) >= 2)
        session.events.put_nowait(StreamingTranscriptEvent("again", True))
        assert _wait_until(lambda: finals[-1] == (2, "again"))
        coordinator.close()
        assert session.closes == 1
        assert errors == []

    def test_empty_committed_final_ends_turn(self):
        np = pytest.importorskip("numpy")
        provider = _CoordinatorProvider()
        finals = []
        coordinator = StreamingSTTCoordinator(
            provider,
            lambda generation, event: finals.append((generation, event.text)),
            lambda generation, error: None,
        )
        assert coordinator.start_utterance(3)
        session = provider.sessions[0]
        assert coordinator.enqueue_audio(
            np.ones((480, 1), dtype=np.int16),
            48000,
        )
        assert _wait_until(lambda: bool(session.audio))
        session.events.put_nowait(StreamingTranscriptEvent("", True))
        assert _wait_until(lambda: finals == [(3, "")])
        coordinator.close()

    def test_abandoned_turn_resets_session_before_next_turn(self):
        np = pytest.importorskip("numpy")
        provider = _CoordinatorProvider()
        finals = []
        coordinator = StreamingSTTCoordinator(
            provider,
            lambda generation, event: finals.append((generation, event.text)),
            lambda generation, error: None,
        )
        assert coordinator.start_utterance(4)
        first_session = provider.sessions[0]
        assert coordinator.enqueue_audio(
            np.ones((480, 1), dtype=np.int16),
            48000,
        )
        assert _wait_until(lambda: bool(first_session.audio))

        coordinator.pause(4)
        assert coordinator.start_utterance(5)
        assert len(provider.sessions) == 2
        second_session = provider.sessions[1]
        first_session.events.put_nowait(StreamingTranscriptEvent("late", True))
        assert coordinator.enqueue_audio(
            np.ones((480, 1), dtype=np.int16),
            48000,
        )
        assert _wait_until(lambda: bool(second_session.audio))
        second_session.events.put_nowait(StreamingTranscriptEvent("current", True))
        assert _wait_until(lambda: finals == [(5, "current")])
        coordinator.close()

    def test_audio_buffer_overflow_reports_once_without_blocking(self):
        provider = _CoordinatorProvider()
        errors = []
        coordinator = StreamingSTTCoordinator(
            provider,
            lambda generation, event: None,
            lambda generation, error: errors.append((generation, str(error))),
        )
        assert coordinator.start_utterance(7)
        oversized_frame = [0] * (16000 * 2 + 1)
        started = time.monotonic()
        assert coordinator.enqueue_audio(oversized_frame, 16000) is False
        assert time.monotonic() - started < 0.1
        assert _wait_until(lambda: len(errors) == 1)
        assert "overflow" in errors[0][1]
        assert coordinator.start_utterance(8)
        assert len(provider.sessions) == 2
        coordinator.close()

    def test_commit_flushes_already_queued_audio_first(self):
        np = pytest.importorskip("numpy")
        provider = _CoordinatorProvider()
        coordinator = StreamingSTTCoordinator(
            provider,
            lambda generation, event: None,
            lambda generation, error: None,
        )
        assert coordinator.start_utterance(9)
        session = provider.sessions[0]
        coordinator.enqueue_audio(np.ones((480, 1), dtype=np.int16), 48000)
        assert coordinator.commit(9)
        assert coordinator.enqueue_audio(
            np.ones((480, 1), dtype=np.int16),
            48000,
        ) is False
        assert _wait_until(lambda: session.commits == 1)
        assert session.operations[-1] == "commit"
        assert "audio" in session.operations[:-1]
        coordinator.close()

    def test_disconnect_falls_back_current_turn_and_reconnects_next_turn(self):
        provider = _CoordinatorProvider()
        errors = []
        coordinator = StreamingSTTCoordinator(
            provider,
            lambda generation, event: None,
            lambda generation, error: errors.append(generation),
        )
        assert coordinator.start_utterance(11)
        provider.sessions[0].events.put_nowait(RuntimeError("expired"))
        assert _wait_until(lambda: errors == [11])
        assert coordinator.start_utterance(12)
        assert len(provider.sessions) == 2
        coordinator.close()

    def test_aged_session_recycles_between_turns(self):
        np = pytest.importorskip("numpy")
        provider = _CoordinatorProvider()
        provider.session_recycle_seconds = 0.01
        finals = []
        coordinator = StreamingSTTCoordinator(
            provider,
            lambda generation, event: finals.append((generation, event.text)),
            lambda generation, error: None,
        )
        assert coordinator.start_utterance(13)
        first_session = provider.sessions[0]
        coordinator.enqueue_audio(
            np.ones((480, 1), dtype=np.int16),
            48000,
        )
        assert _wait_until(lambda: bool(first_session.audio))
        first_session.events.put_nowait(StreamingTranscriptEvent("first", True))
        assert _wait_until(lambda: finals == [(13, "first")])

        with coordinator._lock:
            coordinator._session_opened_at = time.monotonic() - 1.0
        assert coordinator.start_utterance(14)
        assert len(provider.sessions) == 2
        coordinator.close()

    def test_queued_final_wins_when_provider_closes_after_commit(self):
        np = pytest.importorskip("numpy")
        provider = _CoordinatorProvider()
        finals = []
        errors = []
        coordinator = StreamingSTTCoordinator(
            provider,
            lambda generation, event: finals.append((generation, event.text)),
            lambda generation, error: errors.append(generation),
        )
        assert coordinator.start_utterance(15)
        session = provider.sessions[0]
        coordinator.enqueue_audio(
            np.ones((480, 1), dtype=np.int16),
            48000,
        )
        assert _wait_until(lambda: bool(session.audio))
        session.events.put_nowait(StreamingTranscriptEvent("complete", True))
        session.events.put_nowait(RuntimeError("closed after final"))

        assert _wait_until(lambda: finals == [(15, "complete")])
        assert errors == []
        coordinator.close()

    def test_stateful_resampler_preserves_duration_across_chunks(self):
        np = pytest.importorskip("numpy")
        resampler = _PCM16StreamResampler(48000, 16000)
        first = resampler.convert(np.arange(480, dtype=np.int16))
        second = resampler.convert(np.arange(480, 960, dtype=np.int16))
        samples = len(first + second) // 2
        assert 318 <= samples <= 321


class TestStreamingSTTTurnController:
    def test_two_turns_reuse_session_without_exposing_generation(self):
        np = pytest.importorskip("numpy")
        provider = _CoordinatorProvider()
        finals = []
        errors = []
        controller = StreamingSTTTurnController(
            provider,
            lambda event: finals.append(event.text),
            lambda error: errors.append(str(error)),
        )

        assert controller.provider_name == "fake"
        assert controller.start_utterance()
        session = provider.sessions[0]
        controller.enqueue_audio(np.ones((480, 1), dtype=np.int16), 48000)
        assert _wait_until(lambda: bool(session.audio))
        session.events.put_nowait(StreamingTranscriptEvent("first", True))
        assert _wait_until(lambda: finals == ["first"])
        assert controller.active is False

        assert controller.start_utterance()
        assert len(provider.sessions) == 1
        controller.enqueue_audio(np.ones((480, 1), dtype=np.int16), 48000)
        assert _wait_until(lambda: len(session.audio) == 2)
        session.events.put_nowait(StreamingTranscriptEvent("second", True))
        assert _wait_until(lambda: finals == ["first", "second"])
        controller.close()

        assert errors == []
        assert session.closes == 1

    def test_commit_timeout_fails_active_turn_once(self):
        np = pytest.importorskip("numpy")
        provider = _CoordinatorProvider()
        errors = []
        controller = StreamingSTTTurnController(
            provider,
            lambda event: None,
            lambda error: errors.append(str(error)),
            commit_timeout=0.02,
        )

        assert controller.start_utterance()
        controller.enqueue_audio(np.ones((480, 1), dtype=np.int16), 48000)
        assert controller.commit()
        assert _wait_until(lambda: len(errors) == 1)
        assert "timed out" in errors[0]
        assert controller.active is False
        controller.close()

    def test_pause_revokes_a_blocked_session_start(self):
        provider = _CoordinatorProvider()
        entered = threading.Event()
        release = threading.Event()
        original_open = provider.open_session

        async def _blocked_open():
            entered.set()
            await asyncio.get_running_loop().run_in_executor(None, release.wait)
            return await original_open()

        provider.open_session = _blocked_open
        controller = StreamingSTTTurnController(
            provider,
            lambda event: None,
            lambda error: None,
        )
        result = []
        starter = threading.Thread(
            target=lambda: result.append(controller.start_utterance(timeout=1.0))
        )
        starter.start()
        assert entered.wait(1.0)
        controller.pause()
        release.set()
        starter.join(1.0)

        assert result == [False]
        assert controller.active is False
        controller.close()


class TestContinuousStreamingSTTTurnController:
    def test_audio_and_events_continue_across_provider_finals(self):
        np = pytest.importorskip("numpy")
        provider = _CoordinatorProvider()
        events = []
        errors = []
        controller = StreamingSTTTurnController(
            provider,
            lambda event: events.append((event.text, event.is_final)),
            lambda error: errors.append(str(error)),
            continuous=True,
        )

        assert controller.start_utterance()
        session = provider.sessions[0]
        controller.enqueue_audio(np.ones((480, 1), dtype=np.int16), 48000)
        assert _wait_until(lambda: len(session.audio) == 1)
        session.events.put_nowait(StreamingTranscriptEvent("first par", False))
        session.events.put_nowait(StreamingTranscriptEvent("first", True))
        assert _wait_until(
            lambda: events == [("first par", False), ("first", True)]
        )

        controller.enqueue_audio(np.ones((480, 1), dtype=np.int16), 48000)
        assert _wait_until(lambda: len(session.audio) == 2)
        session.events.put_nowait(StreamingTranscriptEvent("second", True))
        assert _wait_until(lambda: events[-1] == ("second", True))

        assert controller.active is True
        assert len(provider.sessions) == 1
        controller.close()
        assert errors == []
        assert session.closes == 1

    def test_manual_commit_preserves_continuous_audio_order(self):
        np = pytest.importorskip("numpy")
        provider = _CoordinatorProvider()
        events = []
        controller = StreamingSTTTurnController(
            provider,
            events.append,
            lambda error: None,
            continuous=True,
        )

        assert controller.start_utterance()
        session = provider.sessions[0]
        controller.enqueue_audio(np.ones((480, 1), dtype=np.int16), 48000)
        assert controller.commit()
        controller.enqueue_audio(np.full((480, 1), 2, dtype=np.int16), 48000)

        assert _wait_until(lambda: len(session.operations) >= 3)
        assert session.operations[:3] == ["audio", "commit", "audio"]
        session.events.put_nowait(
            StreamingTranscriptEvent("still speaking", False, None, "segment-1")
        )
        assert _wait_until(lambda: len(events) == 1)
        assert controller.commit_pending
        session.events.put_nowait(StreamingTranscriptEvent("done", True))
        assert _wait_until(lambda: len(events) == 2)
        assert not controller.commit_pending
        controller.close()
