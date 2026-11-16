"""Coverage for the cloud TTS provider generators: Edge TTS, ElevenLabs, OpenAI.

Exercises ``_generate_edge_tts`` (async), ``_generate_elevenlabs``, and
``_generate_openai_tts`` (plus the ``_tts_response_format_from_path`` and
``_elevenlabs_environment_kwargs`` helpers they depend on) entirely offline:
every cloud SDK is injected into ``sys.modules`` and the OpenAI auth resolver
is patched so no network or credential lookup ever happens.

The intended controller and secret never appear in any asserted message; where
a generator does its own error sanitisation (none of these three do, unlike the
Mistral one) we assert the real behaviour rather than a sanitised string that
the code does not actually produce.
"""

import asyncio
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest

from tools import tts_tool


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "ELEVENLABS_API_KEY",
        "OPENAI_API_KEY",
        "VOICE_TOOLS_OPENAI_KEY",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Edge TTS
# ---------------------------------------------------------------------------


class FakeEdgeCommunicate:
    """Records constructor kwargs and writes deterministic bytes on save()."""

    instances = []

    def __init__(self, text, **kwargs):
        self.text = text
        self.kwargs = kwargs
        self.saved_to = None
        FakeEdgeCommunicate.instances.append(self)

    async def save(self, output_path):
        self.saved_to = output_path
        with open(output_path, "wb") as f:
            f.write(b"edge-audio-bytes")


@pytest.fixture
def mock_edge_tts():
    FakeEdgeCommunicate.instances.clear()
    fake_module = SimpleNamespace(Communicate=FakeEdgeCommunicate)
    with patch.dict("sys.modules", {"edge_tts": fake_module}):
        yield


class TestGenerateEdgeTts:
    def test_happy_path_default_voice_mp3(self, tmp_path, mock_edge_tts):
        output_path = str(tmp_path / "out.mp3")
        result = asyncio.run(
            tts_tool._generate_edge_tts("Hello", output_path, {})
        )
        assert result == output_path
        assert (tmp_path / "out.mp3").read_bytes() == b"edge-audio-bytes"

        comm = FakeEdgeCommunicate.instances[-1]
        assert comm.text == "Hello"
        assert comm.kwargs["voice"] == tts_tool.DEFAULT_EDGE_VOICE
        # speed is 1.0 -> no rate override
        assert "rate" not in comm.kwargs
        assert comm.saved_to == output_path

    def test_voice_from_config_overrides_default(self, tmp_path, mock_edge_tts):
        output_path = str(tmp_path / "out.mp3")
        asyncio.run(
            tts_tool._generate_edge_tts(
                "Hi", output_path, {"edge": {"voice": "en-US-JennyNeural"}}
            )
        )
        comm = FakeEdgeCommunicate.instances[-1]
        assert comm.kwargs["voice"] == "en-US-JennyNeural"

    def test_speed_adds_rate_kwarg(self, tmp_path, mock_edge_tts):
        output_path = str(tmp_path / "out.mp3")
        # 1.1x -> +10%
        asyncio.run(
            tts_tool._generate_edge_tts(
                "Hi", output_path, {"edge": {"speed": 1.1}}
            )
        )
        comm = FakeEdgeCommunicate.instances[-1]
        assert comm.kwargs["rate"] == "+10%"

        # 0.5x -> -50%
        FakeEdgeCommunicate.instances.clear()
        asyncio.run(
            tts_tool._generate_edge_tts(
                "Hi", str(tmp_path / "out2.mp3"), {"speed": 0.5}
            )
        )
        assert FakeEdgeCommunicate.instances[-1].kwargs["rate"] == "-50%"

    def test_save_error_propagates(self, tmp_path, mock_edge_tts):
        output_path = str(tmp_path / "out.mp3")

        class Boom:
            async def save(self, output_path):
                raise RuntimeError("edge network down")

        # Make the injected Communicate return a failing communicator.
        fake_module = SimpleNamespace(
            Communicate=lambda text, **kwargs: Boom()
        )
        with patch.dict("sys.modules", {"edge_tts": fake_module}):
            with pytest.raises(RuntimeError, match="edge network down"):
                asyncio.run(
                    tts_tool._generate_edge_tts("Hi", output_path, {})
                )


# ---------------------------------------------------------------------------
# ElevenLabs
# ---------------------------------------------------------------------------


class FakeElevenLabsClient:
    instances = []

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.convert_kwargs = None
        self.chunks = iter([b"chunk-a", b"chunk-b"])
        FakeElevenLabsClient.instances.append(self)
        self.text_to_speech = SimpleNamespace(convert=self._convert)

    def _convert(self, **kwargs):
        self.convert_kwargs = kwargs
        return self.chunks


@pytest.fixture
def mock_elevenlabs():
    FakeElevenLabsClient.instances.clear()
    fake_client_mod = SimpleNamespace(ElevenLabs=FakeElevenLabsClient)
    fake_env_mod = SimpleNamespace(
        ElevenLabsEnvironment=lambda base, wss: {"base": base, "wss": wss}
    )
    with patch.dict(
        "sys.modules",
        {
            "elevenlabs": fake_client_mod,
            "elevenlabs.client": fake_client_mod,
            "elevenlabs.environment": fake_env_mod,
        },
    ):
        yield


class TestGenerateElevenlabs:
    def test_missing_api_key_raises_exact_message(self, tmp_path, mock_elevenlabs):
        with pytest.raises(ValueError) as exc:
            tts_tool._generate_elevenlabs("Hi", str(tmp_path / "out.mp3"), {})
        assert (
            str(exc.value)
            == "ELEVENLABS_API_KEY not set. Get one at https://elevenlabs.io/"
        )

    def test_happy_path_writes_chunks_mp3(
        self, tmp_path, mock_elevenlabs, monkeypatch
    ):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "secret-el-key")
        output_path = str(tmp_path / "out.mp3")
        result = tts_tool._generate_elevenlabs(
            "Hello world", output_path, {}
        )
        assert result == output_path
        assert (tmp_path / "out.mp3").read_bytes() == b"chunk-achunk-b"

        client = FakeElevenLabsClient.instances[-1]
        assert client.init_kwargs["api_key"] == "secret-el-key"
        # No base_url configured -> no environment kwarg.
        assert "environment" not in client.init_kwargs

        assert client.convert_kwargs["text"] == "Hello world"
        assert client.convert_kwargs["voice_id"] == tts_tool.DEFAULT_ELEVENLABS_VOICE_ID
        assert client.convert_kwargs["model_id"] == tts_tool.DEFAULT_ELEVENLABS_MODEL_ID
        assert client.convert_kwargs["output_format"] == "mp3_44100_128"

    def test_ogg_output_picks_opus_format(
        self, tmp_path, mock_elevenlabs, monkeypatch
    ):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "secret-el-key")
        output_path = str(tmp_path / "out.ogg")
        tts_tool._generate_elevenlabs("Hello", output_path, {})
        client = FakeElevenLabsClient.instances[-1]
        assert client.convert_kwargs["output_format"] == "opus_48000_64"

    def test_base_url_passes_environment_kwarg(
        self, tmp_path, mock_elevenlabs, monkeypatch
    ):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "secret-el-key")
        config = {"elevenlabs": {"base_url": "https://proxy.example.com/v1"}}
        tts_tool._generate_elevenlabs("Hi", str(tmp_path / "out.mp3"), config)
        client = FakeElevenLabsClient.instances[-1]
        env_kw = client.init_kwargs["environment"]
        assert env_kw == {
            "base": "https://proxy.example.com/v1",
            "wss": "wss://proxy.example.com/v1",
        }

    def test_wss_url_overrides_derived_scheme(
        self, tmp_path, mock_elevenlabs, monkeypatch
    ):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "secret-el-key")
        config = {
            "elevenlabs": {
                "base_url": "https://proxy.example.com/v1",
                "wss_url": "wss://custom.example.org/ws",
            }
        }
        tts_tool._generate_elevenlabs("Hi", str(tmp_path / "out.mp3"), config)
        client = FakeElevenLabsClient.instances[-1]
        # An explicit wss_url wins over the scheme derived from base_url
        # (_elevenlabs_environment_kwargs only builds an environment when
        # base_url is set).
        assert client.init_kwargs["environment"] == {
            "base": "https://proxy.example.com/v1",
            "wss": "wss://custom.example.org/ws",
        }

    def test_sdk_convert_error_propagates_and_no_partial_file(
        self, tmp_path, mock_elevenlabs, monkeypatch
    ):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "secret-el-key")

        class BoomConvert:
            def __init__(self):
                self.text_to_speech = SimpleNamespace(
                    convert=lambda **kwargs: (_ for _ in ()).throw(
                        RuntimeError("el service error")
                    )
                )

        fake_client_mod = SimpleNamespace(ElevenLabs=lambda *a, **k: BoomConvert())
        with patch.dict(
            "sys.modules",
            {
                "elevenlabs": fake_client_mod,
                "elevenlabs.client": fake_client_mod,
            },
        ):
            output_path = str(tmp_path / "out.mp3")
            with pytest.raises(RuntimeError, match="el service error"):
                tts_tool._generate_elevenlabs("Hi", output_path, {})
        assert not (tmp_path / "out.mp3").exists()


# ---------------------------------------------------------------------------
# OpenAI TTS
# ---------------------------------------------------------------------------


class FakeOpenAIClient:
    instances = []
    create_side_effect: Optional[BaseException] = None

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.create_kwargs = None
        self.close_called = False
        FakeOpenAIClient.instances.append(self)
        self.audio = SimpleNamespace(speech=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if FakeOpenAIClient.create_side_effect is not None:
            raise FakeOpenAIClient.create_side_effect
        self.create_kwargs = kwargs

        def stream_to_file(path):
            with open(path, "wb") as f:
                f.write(b"openai-audio-bytes")

        return SimpleNamespace(stream_to_file=stream_to_file)

    def close(self):
        self.close_called = True


@pytest.fixture
def mock_openai():
    FakeOpenAIClient.instances.clear()
    FakeOpenAIClient.create_side_effect = None
    fake_module = SimpleNamespace(OpenAI=FakeOpenAIClient)
    with patch.dict("sys.modules", {"openai": fake_module}):
        yield
    FakeOpenAIClient.create_side_effect = None


@pytest.fixture
def _patch_resolver():
    yield patch.object(
        tts_tool,
        "_resolve_openai_audio_client_config",
        return_value=("resolved-key", tts_tool.DEFAULT_OPENAI_BASE_URL, False),
    )


class TestGenerateOpenaiTts:
    def test_happy_path_default_mp3(self, tmp_path, mock_openai, _patch_resolver):
        output_path = str(tmp_path / "out.mp3")
        with _patch_resolver:
            result = tts_tool._generate_openai_tts("Hello", output_path, {})

        assert result == output_path
        assert (tmp_path / "out.mp3").read_bytes() == b"openai-audio-bytes"

        client = FakeOpenAIClient.instances[-1]
        assert client.init_kwargs["api_key"] == "resolved-key"
        assert client.init_kwargs["base_url"] == tts_tool.DEFAULT_OPENAI_BASE_URL
        assert client.close_called is True

        kw = client.create_kwargs
        assert kw["model"] == tts_tool.DEFAULT_OPENAI_MODEL
        assert kw["voice"] == tts_tool.DEFAULT_OPENAI_VOICE
        assert kw["input"] == "Hello"
        assert kw["response_format"] == "mp3"
        assert isinstance(kw["extra_headers"]["x-idempotency-key"], str)
        # Default speed == 1.0 -> no speed kwarg.
        assert "speed" not in kw
        assert "instructions" not in kw
        assert "extra_body" not in kw

    def test_explicit_api_key_skips_resolver(self, tmp_path, mock_openai):
        output_path = str(tmp_path / "out.mp3")
        with patch.object(
            tts_tool,
            "_resolve_openai_audio_client_config",
        ) as resolver:
            tts_tool._generate_openai_tts(
                "Hi",
                output_path,
                {},
                api_key="explicit-key",
                base_url="https://explicit.example/v1",
            )
        resolver.assert_not_called()
        client = FakeOpenAIClient.instances[-1]
        assert client.init_kwargs["api_key"] == "explicit-key"
        assert client.init_kwargs["base_url"] == "https://explicit.example/v1"

    @pytest.mark.parametrize(
        "extension, expected_format",
        [
            (".ogg", "opus"),
            (".wav", "wav"),
            (".flac", "flac"),
            (".mp3", "mp3"),
        ],
    )
    def test_response_format_from_extension(
        self, tmp_path, mock_openai, _patch_resolver, extension, expected_format
    ):
        with _patch_resolver:
            tts_tool._generate_openai_tts(
                "Hi", str(tmp_path / f"out{extension}"), {}
            )
        client = FakeOpenAIClient.instances[-1]
        assert client.create_kwargs["response_format"] == expected_format

    def test_response_format_helper_direct(
        self, _patch_resolver, tmp_path, mock_openai
    ):
        with _patch_resolver:
            assert (
                tts_tool._tts_response_format_from_path("/tmp/a.ogg") == "opus"
            )
            assert (
                tts_tool._tts_response_format_from_path("/tmp/a.wav") == "wav"
            )
            assert (
                tts_tool._tts_response_format_from_path("/tmp/a.flac") == "flac"
            )
            assert (
                tts_tool._tts_response_format_from_path("/tmp/a.mp3") == "mp3"
            )

    def test_speed_from_config_and_clamping(
        self, tmp_path, mock_openai, _patch_resolver
    ):
        with _patch_resolver:
            tts_tool._generate_openai_tts(
                "Hi", str(tmp_path / "out.mp3"), {"openai": {"speed": 2.0}}
            )
        assert (
            FakeOpenAIClient.instances[-1].create_kwargs["speed"] == 2.0
        )

        # Over-speed clamps to 4.0, under-speed clamps to 0.25.
        with _patch_resolver:
            tts_tool._generate_openai_tts(
                "Hi", str(tmp_path / "out2.mp3"), {"openai": {"speed": 5.0}}
            )
        assert (
            FakeOpenAIClient.instances[-1].create_kwargs["speed"] == 4.0
        )

        with _patch_resolver:
            tts_tool._generate_openai_tts(
                "Hi", str(tmp_path / "out3.mp3"), {"openai": {"speed": 0.1}}
            )
        assert (
            FakeOpenAIClient.instances[-1].create_kwargs["speed"] == 0.25
        )

    def test_instructions_added_only_when_provided(
        self, tmp_path, mock_openai, _patch_resolver
    ):
        with _patch_resolver:
            tts_tool._generate_openai_tts(
                "Hi",
                str(tmp_path / "out.mp3"),
                {},
                instructions="Speak cheerfully",
            )
        kw = FakeOpenAIClient.instances[-1].create_kwargs
        assert kw["instructions"] == "Speak cheerfully"

    def test_language_adds_extra_body(self, tmp_path, mock_openai, _patch_resolver):
        with _patch_resolver:
            tts_tool._generate_openai_tts(
                "Hi",
                str(tmp_path / "out.mp3"),
                {"openai": {"language": "es"}},
            )
        kw = FakeOpenAIClient.instances[-1].create_kwargs
        assert kw["extra_body"] == {"lang_code": "es"}

    def test_managed_gateway_coerces_unsupported_model(
        self, tmp_path, mock_openai, _patch_resolver
    ):
        # is_managed=True + a model outside MANAGED_OPENAI_TTS_MODELS +
        # no explicit/config base URL -> coerced back to the managed default.
        with patch.object(
            tts_tool,
            "_resolve_openai_audio_client_config",
            return_value=(
                "managed-token",
                "https://gateway.example/v1",
                True,
            ),
        ):
            with patch.object(tts_tool.logger, "warning") as warn:
                tts_tool._generate_openai_tts(
                    "Hi", str(tmp_path / "out.mp3"), {"openai": {"model": "tts-1-hd"}}
                )
        warn.assert_called_once()
        assert FakeOpenAIClient.instances[-1].create_kwargs["model"] == (
            tts_tool.DEFAULT_OPENAI_MODEL
        )

    def test_missing_key_propagates_resolver_error(
        self, tmp_path, mock_openai
    ):
        with patch.object(
            tts_tool,
            "_resolve_openai_audio_client_config",
            side_effect=ValueError(
                "Neither tts.openai.api_key in config nor "
                "VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
            ),
        ):
            with pytest.raises(ValueError, match="OPENAI_API_KEY") as exc:
                tts_tool._generate_openai_tts(
                    "Hi", str(tmp_path / "out.mp3"), {}
                )
        assert "OPENAI_API_KEY" in str(exc.value)

    def test_sdk_error_propagates_and_close_still_called(
        self, tmp_path, mock_openai, _patch_resolver
    ):
        FakeOpenAIClient.create_side_effect = RuntimeError("openai service failed")
        with _patch_resolver:
            with pytest.raises(RuntimeError, match="openai service failed"):
                tts_tool._generate_openai_tts(
                    "Hi", str(tmp_path / "out.mp3"), {}
                )
        client = FakeOpenAIClient.instances[-1]
        assert client.close_called is True
        assert not (tmp_path / "out.mp3").exists()
