"""Tests for Yandex SpeechKit TTS provider."""

import json
import queue
import sys
import threading
from types import ModuleType
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clean_yandex_env(monkeypatch):
    for key in (
        "YANDEX_FOLDER_ID",
        "YANDEX_API_KEY",
        "YANDEX_IAM_TOKEN",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


def _yandex_config(**overrides):
    config = {
        "provider": "yandex",
        "yandex": {
            "folder_id": "folder-1",
            "api_key": "api-key-1",
            "iam_token": "iam-token-1",
            "voice": "kirill",
            "role": "neutral",
            "speed": 1.5,
            "audio_format": "PCM16(24000)",
            "timeout": 12,
            "stream_timeout": 34,
        },
    }
    config["yandex"].update(overrides)
    return config


def test_get_provider_accepts_yandex():
    from tools.tts_tool import _get_provider

    assert _get_provider({"provider": "yandex"}) == "yandex"


def test_yandex_import_uses_lazy_deps(monkeypatch):
    calls = []
    fake_sdk = ModuleType("yandex_ai_studio_sdk")

    class FakeAIStudio:
        pass

    fake_sdk.AIStudio = FakeAIStudio
    monkeypatch.setitem(sys.modules, "yandex_ai_studio_sdk", fake_sdk)

    def fake_ensure(feature, *, prompt=True):
        calls.append((feature, prompt))

    monkeypatch.setattr("tools.lazy_deps.ensure", fake_ensure)

    from tools.tts_tool import _import_yandex_ai_studio

    assert _import_yandex_ai_studio() is FakeAIStudio
    assert calls == [("tts.yandex", False)]


def test_yandex_auth_prefers_api_key_and_ignores_placeholders(monkeypatch):
    monkeypatch.setenv("YANDEX_FOLDER_ID", "env-folder")
    monkeypatch.setenv("YANDEX_API_KEY", "env-api-key")
    monkeypatch.setenv("YANDEX_IAM_TOKEN", "env-iam-token")

    from tools.tts_tool import _resolve_yandex_tts_config

    resolved = _resolve_yandex_tts_config(
        {
            "yandex": {
                "folder_id": "${YANDEX_FOLDER_ID}",
                "api_key": "",
                "iam_token": "",
            }
        }
    )

    assert resolved["folder_id"] == "env-folder"
    assert resolved["auth"] == "env-api-key"
    assert resolved["auth_type"] == "api_key"


def test_yandex_auth_error_does_not_leak_secret():
    from tools.tts_tool import _resolve_yandex_tts_config

    with pytest.raises(ValueError) as exc:
        _resolve_yandex_tts_config({"yandex": {"api_key": "super-secret"}})

    assert "super-secret" not in str(exc.value)


def test_yandex_file_fallback_writes_wav_and_passes_config(tmp_path, monkeypatch):
    output_path = tmp_path / "yandex.wav"
    captured = {}

    class FakeResult:
        data = b"\x01\x00\x02\x00"

    class FakeTTS:
        def run(self, text, timeout):
            captured["run"] = {"text": text, "timeout": timeout}
            return FakeResult()

    class FakeSpeechKit:
        def tts(self, **kwargs):
            captured["tts_kwargs"] = kwargs
            return FakeTTS()

    class FakeAIStudio:
        def __init__(self, folder_id, auth):
            captured["sdk"] = {"folder_id": folder_id, "auth": auth}
            self.speechkit = FakeSpeechKit()

    monkeypatch.setattr("tools.tts_tool._import_yandex_ai_studio", lambda: FakeAIStudio)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: _yandex_config())

    from tools.tts_tool import text_to_speech_tool

    result = json.loads(text_to_speech_tool("Привет", output_path=str(output_path)))

    assert result["success"] is True
    assert result["provider"] == "yandex"
    assert output_path.read_bytes().startswith(b"RIFF")
    assert captured["sdk"] == {"folder_id": "folder-1", "auth": "api-key-1"}
    assert captured["tts_kwargs"] == {
        "voice": "kirill",
        "role": "neutral",
        "speed": 1.5,
        "audio_format": "PCM16(24000)",
    }
    assert captured["run"] == {"text": "Привет", "timeout": 12.0}


def test_check_tts_requirements_requires_yandex_sdk_and_credentials(monkeypatch):
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: _yandex_config())

    with patch("tools.tts_tool._import_yandex_ai_studio", return_value=object):
        from tools.tts_tool import check_tts_requirements

        assert check_tts_requirements() is True

    monkeypatch.setattr(
        "tools.tts_tool._load_tts_config",
        lambda: {"provider": "yandex", "yandex": {"folder_id": "folder-1"}},
    )
    with patch("tools.tts_tool._import_yandex_ai_studio", return_value=object):
        assert check_tts_requirements() is False


def test_check_streaming_tts_available_accepts_yandex(monkeypatch):
    monkeypatch.setattr("tools.tts_tool._import_yandex_ai_studio", lambda: object)
    monkeypatch.setattr("tools.tts_tool._import_sounddevice", lambda: object())

    from tools.tts_tool import check_streaming_tts_available

    assert check_streaming_tts_available(_yandex_config()) is True


def test_yandex_streaming_writes_clean_sentences_and_audio(monkeypatch):
    text_queue = queue.Queue()
    text_queue.put("**Привет** [дом](https://example.com).")
    text_queue.put(None)
    stop_event = threading.Event()
    done_event = threading.Event()
    displayed = []

    class FakeChunk:
        data = b"\x01\x00\x02\x00"

    class FakeStream:
        def __init__(self):
            self.writes = []
            self.flushes = 0
            self.done = False
            self.responses = queue.Queue()

        def write(self, text):
            self.writes.append(text)
            self.responses.put(FakeChunk())

        def flush(self):
            self.flushes += 1

        def done_writing(self):
            self.done = True
            self.responses.put(None)

        def __iter__(self):
            return self

        def __next__(self):
            item = self.responses.get(timeout=1)
            if item is None:
                raise StopIteration
            return item

    fake_stream = FakeStream()

    class FakeTTS:
        def create_bistream(self, timeout):
            assert timeout == 34.0
            return fake_stream

    class FakeOutputStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.writes = []
            self.started = False
            self.closed = False

        def start(self):
            self.started = True

        def write(self, data):
            self.writes.append(data)

        def stop(self):
            pass

        def close(self):
            self.closed = True

    fake_output = FakeOutputStream
    output_instances = []

    class FakeSoundDevice:
        def RawOutputStream(self, **kwargs):
            stream = fake_output(**kwargs)
            output_instances.append(stream)
            return stream

    monkeypatch.setattr("tools.tts_tool._create_yandex_tts_client", lambda _cfg: (FakeTTS(), _cfg["yandex"]))
    monkeypatch.setattr("tools.tts_tool._import_sounddevice", lambda: FakeSoundDevice())

    from tools.tts_tool import _stream_yandex_tts_to_speaker

    monkeypatch.setattr("tools.tts_tool.time.sleep", lambda _seconds: None)

    _stream_yandex_tts_to_speaker(
        text_queue,
        stop_event,
        done_event,
        display_callback=displayed.append,
        tts_config=_yandex_config(),
    )

    assert done_event.is_set()
    assert fake_stream.writes == ["Привет дом."]
    assert fake_stream.flushes == 1
    assert fake_stream.done is True
    assert displayed == ["**Привет** [дом](https://example.com)."]
    assert output_instances[0].kwargs == {"samplerate": 24000, "channels": 1, "dtype": "int16"}
    assert output_instances[0].writes[0] == b"\x01\x00\x02\x00"
    assert output_instances[0].writes[1].strip(b"\x00") == b""
    assert len(output_instances[0].writes[1]) > 0
    assert output_instances[0].writes[2].strip(b"\x00") == b""
    assert len(output_instances[0].writes[2]) > len(output_instances[0].writes[1])
    assert output_instances[0].closed is True


def test_yandex_streaming_does_not_open_audio_before_text(monkeypatch):
    text_queue = queue.Queue()
    text_queue.put(None)
    done_event = threading.Event()

    def fail_create_client(_cfg):
        raise AssertionError("Yandex runtime should be lazy until there is text to speak")

    monkeypatch.setattr("tools.tts_tool._create_yandex_tts_client", fail_create_client)

    from tools.tts_tool import _stream_yandex_tts_to_speaker

    _stream_yandex_tts_to_speaker(
        text_queue,
        threading.Event(),
        done_event,
        tts_config=_yandex_config(),
    )

    assert done_event.is_set()


def test_yandex_streaming_reuses_audio_stream_between_sentences(monkeypatch):
    text_queue = queue.Queue()
    text_queue.put("Первое предложение готово. Второе предложение тоже готово.")
    text_queue.put(None)
    done_event = threading.Event()

    class FakeChunk:
        data = b"\x01\x00"

    class FakeStream:
        def __init__(self):
            self.responses = queue.Queue()

        def write(self, text):
            self.responses.put(FakeChunk())

        def flush(self):
            pass

        def done_writing(self):
            self.responses.put(None)

        def __iter__(self):
            return self

        def __next__(self):
            item = self.responses.get(timeout=1)
            if item is None:
                raise StopIteration
            return item

    class FakeTTS:
        def create_bistream(self, timeout):
            return FakeStream()

    output_instances = []

    class FakeOutputStream:
        def __init__(self, **kwargs):
            self.writes = []
            self.closed = False

        def start(self):
            pass

        def write(self, data):
            self.writes.append(data)

        def stop(self):
            pass

        def close(self):
            self.closed = True

    class FakeSoundDevice:
        def RawOutputStream(self, **kwargs):
            stream = FakeOutputStream(**kwargs)
            output_instances.append(stream)
            return stream

    monkeypatch.setattr("tools.tts_tool._create_yandex_tts_client", lambda _cfg: (FakeTTS(), _cfg["yandex"]))
    monkeypatch.setattr("tools.tts_tool._import_sounddevice", lambda: FakeSoundDevice())

    from tools.tts_tool import _stream_yandex_tts_to_speaker

    monkeypatch.setattr("tools.tts_tool.time.sleep", lambda _seconds: None)

    _stream_yandex_tts_to_speaker(
        text_queue,
        threading.Event(),
        done_event,
        tts_config=_yandex_config(),
    )

    assert done_event.is_set()
    assert len(output_instances) == 1
    assert output_instances[0].closed is True
    assert output_instances[0].writes[0] == b"\x01\x00"
    assert output_instances[0].writes[1].strip(b"\x00") == b""
    assert output_instances[0].writes[2] == b"\x01\x00"
    assert output_instances[0].writes[3].strip(b"\x00") == b""
    assert output_instances[0].writes[4].strip(b"\x00") == b""
    assert len(output_instances[0].writes[4]) > len(output_instances[0].writes[3])


def test_yandex_streaming_waits_for_reader_before_closing_output(monkeypatch):
    text_queue = queue.Queue()
    text_queue.put("Длинный ответ должен дождаться аудио целиком.")
    text_queue.put(None)
    stop_event = threading.Event()
    done_event = threading.Event()
    join_timeouts = []

    class FakeThread:
        def __init__(self, target, daemon=False):
            self.target = target
            self.daemon = daemon

        def start(self):
            pass

        def join(self, timeout=None):
            join_timeouts.append(timeout)

        def is_alive(self):
            return False

    class FakeStream:
        def __init__(self):
            self.writes = []
            self.done = False

        def write(self, text):
            self.writes.append(text)

        def flush(self):
            pass

        def done_writing(self):
            self.done = True

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    fake_stream = FakeStream()

    class FakeTTS:
        def create_bistream(self, timeout):
            return fake_stream

    class FakeOutputStream:
        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    class FakeSoundDevice:
        def RawOutputStream(self, **kwargs):
            return FakeOutputStream()

    monkeypatch.setattr("tools.tts_tool.threading.Thread", FakeThread)
    monkeypatch.setattr("tools.tts_tool._create_yandex_tts_client", lambda _cfg: (FakeTTS(), _cfg["yandex"]))
    monkeypatch.setattr("tools.tts_tool._import_sounddevice", lambda: FakeSoundDevice())

    from tools.tts_tool import _stream_yandex_tts_to_speaker

    _stream_yandex_tts_to_speaker(
        text_queue,
        stop_event,
        done_event,
        tts_config=_yandex_config(stream_timeout=34),
    )

    assert done_event.is_set()
    assert fake_stream.done is True
    assert join_timeouts == [34.0]


def test_yandex_streaming_sets_done_on_error(monkeypatch):
    text_queue = queue.Queue()
    text_queue.put("Привет.")
    text_queue.put(None)
    done_event = threading.Event()

    monkeypatch.setattr(
        "tools.tts_tool._create_yandex_tts_client",
        lambda _cfg: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    from tools.tts_tool import _stream_yandex_tts_to_speaker

    _stream_yandex_tts_to_speaker(
        text_queue,
        threading.Event(),
        done_event,
        tts_config=_yandex_config(),
    )

    assert done_event.is_set()
