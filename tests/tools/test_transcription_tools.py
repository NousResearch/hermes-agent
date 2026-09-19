"""Tests for tools.transcription_tools — three-provider STT pipeline.

Covers the full provider matrix (local, groq, openai), fallback chains,
model auto-correction, config loading, validation edge cases, and
end-to-end dispatch.  All external dependencies are mocked.
"""

import os
import sys
import struct
import subprocess
import types
import wave
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

if "faster_whisper" not in sys.modules:
    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = MagicMock(name="WhisperModel")
    # Set ``__spec__`` so ``importlib.util.find_spec("faster_whisper")``
    # doesn't raise ``ValueError: faster_whisper.__spec__ is None`` during
    # collection (used by skipif markers further down in this file).
    from importlib.machinery import ModuleSpec
    faster_whisper_stub.__spec__ = ModuleSpec("faster_whisper", loader=None)
    sys.modules["faster_whisper"] = faster_whisper_stub


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_wav(tmp_path):
    """Create a minimal valid WAV file (1 second of silence at 16kHz)."""
    wav_path = tmp_path / "test.wav"
    n_frames = 16000
    silence = struct.pack(f"<{n_frames}h", *([0] * n_frames))

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(silence)

    return str(wav_path)


@pytest.fixture
def sample_ogg(tmp_path):
    """Create a fake OGG file for validation tests."""
    ogg_path = tmp_path / "test.ogg"
    ogg_path.write_bytes(b"fake audio data")
    return str(ogg_path)

@pytest.fixture
def sample_silk(tmp_path):
    """Create a fake WeChat .silk file for preprocessing tests."""
    silk_path = tmp_path / "voice.silk"
    silk_path.write_bytes(b"\x02#!SILK_V3fake")
    return str(silk_path)


@pytest.fixture
def oversized_wav(tmp_path):
    """Create a sparse WAV-shaped file just above the remote upload cap."""
    from tools.transcription_common import MAX_FILE_SIZE

    wav_path = tmp_path / "oversized.wav"
    with wav_path.open("wb") as audio_file:
        audio_file.seek(MAX_FILE_SIZE)
        audio_file.write(b"\0")
    return str(wav_path)


pytestmark = pytest.mark.usefixtures("disable_lazy_stt_install")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no real API keys leak into tests."""
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)
    # Stub host binary discovery; individual codec tests supply their own fake.
    monkeypatch.setattr("tools.transcription_audio._find_ffmpeg_binary", lambda: None)


# ============================================================================
# _get_provider — full permutation matrix
# ============================================================================

class TestGetProviderGroq:
    """Groq-specific provider selection tests."""

    def test_groq_when_key_set(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "groq"}) == "groq"


class TestProcessErrorDetail:
    """#112582: a failed STT helper reports its real error even when
    CalledProcessError carries no captured output (stderr/stdout default to None)."""

    @staticmethod
    def _detail(*, stderr=None, stdout=None):
        from tools.transcription_common import _process_error_detail

        error = subprocess.CalledProcessError(
            1,
            ["ffmpeg"],
            output=stdout,
            stderr=stderr,
        )
        return _process_error_detail(error)

    def test_prefers_stderr_over_stdout(self):
        assert self._detail(stderr=" stderr detail \n", stdout="stdout detail") == "stderr detail"

    @pytest.mark.parametrize(
        ("stderr", "stdout", "expected"),
        [
            (None, None, "returned non-zero exit status 1"),
            (None, " stdout detail \n", "stdout detail"),
            (b" bad \xff output \n", None, "bad \ufffd output"),
        ],
    )
    def test_missing_or_byte_output_does_not_mask_the_failure(self, stderr, stdout, expected):
        assert expected in self._detail(stderr=stderr, stdout=stdout)


class TestGetProviderFallbackPriority:
    """Auto-detect fallback priority and explicit provider behaviour."""

    def test_auto_detect_prefers_local(self):
        """Auto-detect prefers local over any cloud provider."""
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "local"

    def test_unknown_provider_passed_through(self):
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "custom-endpoint"}) == "custom-endpoint"

# ============================================================================
# Explicit provider config respected  (GH-1774)
# ============================================================================

class TestExplicitProviderRespected:
    """When stt.provider is explicitly set, that choice is authoritative.
    No silent fallback to a different cloud provider."""

    def test_explicit_local_no_fallback_to_openai(self, monkeypatch):
        """GH-1774: provider=local must not silently fall back to openai
        even when an OpenAI API key is set."""
        monkeypatch.setenv("OPENAI_API_KEY", "***")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.tool_backend_helpers.read_selection", return_value="local"), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "local"})
            assert result == "none", f"Expected 'none' but got {result!r}"

    def test_seeded_local_without_stored_selection_autodetects(self, monkeypatch):
        """The DEFAULT_CONFIG-seeded stt.provider: local (no raw-config
        selection) is treated as never-configured: autodetect runs instead of
        hard-pinning to a missing local backend."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._try_lazy_install_stt", return_value=False), \
             patch("tools.tool_backend_helpers.read_selection", return_value=None), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "local"}) == "groq"

    def test_explicit_local_uses_local_command_fallback(self, monkeypatch):
        """Local-to-local_command fallback is fine — both are local."""
        monkeypatch.setenv(
            "HERMES_LOCAL_STT_COMMAND",
            "whisper {input_path} --output_dir {output_dir} --language {language}",
        )
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "local"})
            assert result == "local_command"


    def test_auto_detect_prefers_groq_over_openai(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({})
            assert result == "groq"


# ============================================================================
# _transcribe_groq
# ============================================================================

class TestTranscribeGroq:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from tools.transcription_tools import _transcribe_groq
        result = _transcribe_groq("/tmp/test.ogg", "whisper-large-v3-turbo")
        assert result["success"] is False
        assert "GROQ_API_KEY" in result["error"]

    def test_openai_package_not_installed(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq("/tmp/test.ogg", "whisper-large-v3-turbo")
        assert result["success"] is False
        assert "openai package" in result["error"]


    def test_null_groq_subsection_is_safe(self, monkeypatch, sample_wav):
        """`stt.groq: null` in YAML yields None; must not raise, auto-detect stays intact."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hi"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch(
                 "tools.transcription_tools._load_stt_config",
                 return_value={"groq": None},
             ):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["success"] is True
        kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert "language" not in kwargs


# ============================================================================
# _transcribe_openai — additional tests
# ============================================================================

@pytest.fixture
def context_wire(monkeypatch, sample_wav):
    """Real dispatch/hook merge/request builder; only SDK and discovery boundaries fake."""
    import copy
    from tools import transcription_tools as stt

    cfg = {
        "provider": "openai", "cloud_trim_silence": False, "prompt": "generic",
        "openai": {"api_key": "offline-key", "model": "gpt-transcribe",
                   "base_url": "https://api.openai.com/v1", "prompt": "native",
                   "keywords": ["Hermes"], "language": "de"},
    }
    hooks = []
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1/"
    client.audio.transcriptions.create.return_value = types.SimpleNamespace(text="offline transcript")
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr("openai.OpenAI", constructor)
    monkeypatch.setattr(stt, "_HAS_OPENAI", True)
    monkeypatch.setattr(stt, "_load_stt_config", lambda: cfg)
    monkeypatch.setattr("tools.tool_backend_helpers.read_selection", lambda *a: cfg["provider"])
    monkeypatch.setattr(stt, "_resolve_provider_key", lambda *a, **kw: "offline-key")
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: bool(hooks))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *a, **kw: list(hooks))

    def run(**kwargs):
        before = copy.deepcopy(cfg)
        result = stt.transcribe_audio(sample_wav, **kwargs)
        assert cfg == before
        return result

    return types.SimpleNamespace(cfg=cfg, hooks=hooks, client=client, constructor=constructor,
                                 run=run, audio=sample_wav)


class TestNativeTranscriptionContext:
    @pytest.mark.parametrize("base", ["https://api.openai.com/v1", "https://api.openai.com/v1/",
                                      "https://api.openai.com:443/v1/"])
    @pytest.mark.parametrize("keywords,expected", [
        ('["Hermes", "Nous Research"]', ["Hermes", "Nous Research"]),
        (["[Hermes]", " Nous Research "], ["[Hermes]", "Nous Research"]),
    ])
    def test_native_context_reaches_sdk(self, context_wire, base, keywords, expected):
        wire = context_wire
        wire.client.base_url = base
        wire.cfg["openai"]["keywords"] = keywords
        assert wire.run()["success"]
        kw = wire.client.audio.transcriptions.create.call_args.kwargs
        assert kw["prompt"] == "native"
        assert kw["extra_body"] == {"keywords": expected, "languages": ["de"]}
        assert "language" not in kw
        wire.client.close.assert_called_once()
        assert wire.constructor.call_args.kwargs["api_key"] == "offline-key"

    @pytest.mark.parametrize("start,finish,prompt", [
        ("whisper-1", "gpt-transcribe", "native"),
        ("gpt-transcribe", "whisper-1", "generic"),
        ("gpt-transcribe", "gpt-4o-transcribe", "generic"),
    ])
    def test_model_only_hooks_select_final_model_defaults(self, context_wire, start, finish, prompt):
        wire = context_wire
        wire.cfg["openai"]["model"] = start
        wire.hooks.extend([{"language": "fr"}, {"model": finish}])
        assert wire.run()["success"]
        kw = wire.client.audio.transcriptions.create.call_args.kwargs
        assert kw["model"] == finish
        assert kw["prompt"] == prompt
        if finish == "gpt-transcribe":
            assert kw["extra_body"]["languages"] == ["fr"]
        else:
            assert kw["language"] == "fr"
            assert "extra_body" not in kw

    @pytest.mark.parametrize("hooks,expected", [
        ([{"prompt": ""}, {"model": "gpt-transcribe"}], None),
        ([{"model": "gpt-transcribe"}, {"prompt": ""}], None),
        ([{"prompt": "first"}, {"prompt": ""}, {"model": "gpt-transcribe"}], None),
        ([{"prompt": ""}, {"model": "gpt-transcribe"}, {"prompt": "winner"}], "winner"),
        ([{"model": "gpt-transcribe"}, {"prompt": "winner"}], "winner"),
    ])
    def test_hook_prompt_presence_beats_unused_oversized_config(self, context_wire, hooks, expected):
        wire = context_wire
        wire.cfg["openai"].update(model="whisper-1", prompt="x" * 5001)
        wire.cfg["prompt"] = "y" * 5001
        wire.hooks.extend(hooks)
        assert wire.run()["success"]
        assert wire.client.audio.transcriptions.create.call_args.kwargs.get("prompt") == expected

    @pytest.mark.parametrize("origin", ["hook", "native", "generic"])
    @pytest.mark.parametrize("length", [5000, 5001])
    def test_effective_prompt_ceiling_precedes_tail_cap(self, context_wire, origin, length):
        wire = context_wire
        prompt = "ä" * (length - 4) + "TAIL"
        if origin == "hook":
            wire.hooks.append({"prompt": prompt})
        elif origin == "native":
            wire.cfg["openai"]["prompt"] = prompt
        else:
            del wire.cfg["openai"]["prompt"]
            wire.cfg["prompt"] = prompt
        result = wire.run()
        if length == 5001:
            assert not result["success"]
            assert "5000" in result["error"]
            wire.client.audio.transcriptions.create.assert_not_called()
        else:
            assert result["success"]
            assert wire.client.audio.transcriptions.create.call_args.kwargs["prompt"] == prompt[-896:]
        wire.client.close.assert_called_once()

    @pytest.mark.parametrize("field,value", [
        ("keywords", "\nHermes"), ("keywords", ["Hermes\r"]), ("keywords", "<Hermes>"),
        ("keywords", '\n["Hermes"]'),
        ("keywords", '["Hermes\\n"]'), ("keywords", ["ok", 1]), ("keywords", {"key": "value"}),
        ("keywords", '["ok", null]'), ("keywords", ('Hermes',)),
        ("keywords", "[Hermes]"), ("keywords", ' ["Hermes"'),
    ])
    def test_invalid_context_is_rejected_before_upload(self, context_wire, field, value):
        wire = context_wire
        wire.cfg["openai"][field] = value
        result = wire.run()
        assert not result["success"]
        assert field in result["error"]
        wire.client.audio.transcriptions.create.assert_not_called()
        wire.client.close.assert_called_once()

    @pytest.mark.parametrize("base", [
        "http://api.openai.com/v1/", "https://api.openai.com:444/v1/",
        "https://api.openai.com.evil.test/v1/", "https://api.openai.com./v1/",
        "https://api.openai.com/v10/", "https://api.openai.com/v1//",
        "https://api.openai.com/v1/?x=1", "https://api.openai.com/v1/#fragment",
        "https://api.openai.com/v1/?", "https://api.openai.com/v1/#",
        "https://api.openai.com/v1/\n",
        "https://user@api.openai.com/v1/", "https://api.deepinfra.com/v1/openai/",
        "https://managed.example/v1/", None,
    ])
    def test_actual_endpoint_gates_new_fields_only(self, context_wire, base):
        wire = context_wire
        wire.client.base_url = base
        wire.cfg["openai"].update(prompt="x" * 5001, keywords=[1], language="de")
        wire.cfg["prompt"] = "generic" * 1000
        assert wire.run()["success"]
        kw = wire.client.audio.transcriptions.create.call_args.kwargs
        assert kw["prompt"] == wire.cfg["prompt"][-896:]
        assert kw["extra_body"] == {"languages": ["de"]}
        assert "language" not in kw
        assert wire.constructor.call_args.kwargs["base_url"] == wire.cfg["openai"]["base_url"]

    def test_groq_correction_precedes_context_choice(self, context_wire, monkeypatch):
        from tools import transcription_cloud
        wire = context_wire
        monkeypatch.setattr(transcription_cloud, "DEFAULT_STT_MODEL", "gpt-transcribe")
        wire.hooks.append({"model": "whisper-large-v3-turbo"})
        assert wire.run()["success"]
        kw = wire.client.audio.transcriptions.create.call_args.kwargs
        assert kw["model"] == "gpt-transcribe"
        assert kw["prompt"] == "native"

    def test_deepinfra_keeps_generic_context_and_own_credentials(self, context_wire):
        wire = context_wire
        wire.cfg.update(provider="deepinfra", deepinfra={"model": "gpt-transcribe", "language": "de"})
        wire.client.base_url = "https://api.deepinfra.com/v1/openai/"
        wire.cfg["openai"].update(prompt="x" * 5001, keywords=[1])
        assert wire.run()["success"]
        kw = wire.client.audio.transcriptions.create.call_args.kwargs
        assert kw["prompt"] == "generic"
        assert kw["extra_body"] == {"languages": ["de"]}
        assert "deepinfra.com" in wire.constructor.call_args.kwargs["base_url"]

    def test_native_tail_cap_depends_on_endpoint_not_response_label(self, context_wire):
        from tools.transcription_cloud import _transcribe_openai
        wire = context_wire
        prompt = "x" * 4996 + "TAIL"
        result = _transcribe_openai(wire.audio, "gpt-transcribe", api_key="offline-key",
                                    provider_label="compatible", prompt=prompt)
        assert result["success"]
        assert wire.client.audio.transcriptions.create.call_args.kwargs["prompt"] == prompt[-896:]

    def test_blank_generic_prompt_stays_absent_on_native_endpoint(self, context_wire):
        wire = context_wire
        del wire.cfg["openai"]["prompt"]
        wire.cfg["prompt"] = "   "
        assert wire.run()["success"]
        assert "prompt" not in wire.client.audio.transcriptions.create.call_args.kwargs

    def test_extra_body_merge_survives_real_sdk_serialization(self, context_wire, monkeypatch):
        import copy
        import httpx
        import openai
        from openai._client import OpenAI
        from tools.transcription_cloud import _transcribe_openai
        wire = context_wire
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"text": "offline transcript"})

        client = OpenAI(api_key="offline-key", base_url="https://api.openai.com/v1",
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
        extras = {"trace": "keep", "language": "it", "languages": ["it"]}
        before = copy.deepcopy(extras)
        result = _transcribe_openai(wire.audio, "gpt-transcribe", prompt="caller",
                                    extra_body=extras)
        assert result["success"]
        assert extras == before
        assert client.is_closed()
        assert len(requests) == 1
        body = requests[0].content
        assert b'name="prompt"\r\n\r\ncaller' in body
        assert b'name="trace"\r\n\r\nkeep' in body
        assert b'name="keywords[]"\r\n\r\nHermes' in body
        assert b'name="languages[]"\r\n\r\nde' in body
        assert b'name="language"' not in body


@pytest.fixture
def sdk_context_wire(monkeypatch, sample_wav):
    """Public pipeline, real hooks, URL resolution and SDK multipart; offline HTTP only."""
    import copy
    from email import policy
    from email.parser import BytesParser
    import httpx
    import openai
    from hermes_cli import plugins
    from tools import transcription_tools as stt

    cfg = {
        "provider": "deepinfra", "cloud_trim_silence": False, "prompt": "generic",
        "openai": {"api_key": "offline-openai", "model": "gpt-transcribe",
                   "base_url": "https://api.openai.com/v1", "prompt": "native",
                   "keywords": ["Hermes"], "language": "de"},
        "deepinfra": {"model": "gpt-transcribe", "base_url": "https://api.openai.com/v1",
                     "language": "de"},
    }
    requests, clients, hooks, hook_calls, constructors, key_calls = [], [], [], [], [], []
    manager = plugins.PluginManager()
    context = plugins.PluginContext(plugins.PluginManifest(name="sdk-context-test"), manager)
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    monkeypatch.setattr(stt, "_load_stt_config", lambda: cfg)
    monkeypatch.setattr("tools.tool_backend_helpers.read_selection", lambda *a: cfg["provider"])
    monkeypatch.setattr("tools.managed_tool_gateway.resolve_managed_tool_gateway",
                        lambda *a: types.SimpleNamespace(nous_user_token="offline-nous",
                                                         gateway_origin="https://api.openai.com"))

    def resolve_key(env_var, provider):
        key_calls.append((env_var, provider))
        return "offline-" + provider

    monkeypatch.setattr(stt, "_resolve_provider_key", resolve_key)
    # External catalog response only: retain the real default-model selection and shim.
    monkeypatch.setattr("hermes_cli.models._fetch_deepinfra_models_by_tag",
                        lambda *a, **kw: [{"id": "gpt-transcribe"}])

    def respond(request):
        requests.append(request)
        if b'name="response_format"\r\n\r\ntext' in request.content:
            return httpx.Response(200, text="offline transcript")
        return httpx.Response(200, json={"text": "offline transcript"})

    real_openai = openai.OpenAI

    def construct(**kwargs):
        constructors.append(dict(kwargs))
        client = real_openai(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "OpenAI", construct)

    def run(**kwargs):
        before = copy.deepcopy((cfg, hooks))
        for result in hooks:
            def callback(result=result, **kw):
                hook_calls.append(kw)
                return result
            context.register_hook("pre_transcription", callback)
        result = stt.transcribe_audio(sample_wav, source="gateway", **kwargs)
        assert (cfg, hooks) == before
        assert clients and all(client.is_closed() for client in clients)
        for invocation in hook_calls:
            assert invocation["provider"] == ("openai" if cfg["provider"] == "nous" else cfg["provider"])
            assert invocation["source"] == "gateway"
            assert "field_overrides" not in invocation
        return result

    def fields():
        assert len(requests) == 1
        request = requests[0]
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.content)
        values = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name != "file":
                payload = part.get_payload(decode=True)
                assert isinstance(payload, bytes)
                values.setdefault(name, []).append(payload.decode("utf-8"))
        return values

    return types.SimpleNamespace(cfg=cfg, hooks=hooks, requests=requests, constructors=constructors,
                                 key_calls=key_calls, run=run, fields=fields)


class TestCompatibleNativeContextWire:
    @pytest.mark.parametrize("model", ["gpt-transcribe", "whisper-1"])
    @pytest.mark.parametrize("language,plural,expected", [
        ("de", None, ["de"]),
        ("de,en", None, ["de,en"]),
        ("  de  ", None, ["de"]),
        ("   ", None, None),
        ("de", ["fr"], ["de"]),
        ("de", [False], ["de"]),
        ("de", [], ["de"]),
        ("   ", ["fr"], None),
        ("english", '[broken', ["english"]),
    ])
    def test_legacy_language_mapping_ignores_plural_config(
        self, sdk_context_wire, monkeypatch, model, language, plural, expected,
    ):
        wire = sdk_context_wire
        monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)
        wire.cfg["provider"] = "openai"
        wire.cfg["openai"].update(model=model, language=language)
        if plural is not None:
            wire.cfg["openai"]["languages"] = plural
        assert wire.run()["success"]
        fields = wire.fields()
        language_field = "languages[]" if model == "gpt-transcribe" else "language"
        assert fields.get(language_field) == expected
        assert ("language" not in fields) if model == "gpt-transcribe" else ("languages[]" not in fields)
        assert fields.get("keywords[]") == (["Hermes"] if model == "gpt-transcribe" else None)

    @pytest.mark.parametrize("provider", ["openai", "deepinfra"])
    @pytest.mark.parametrize("base,native", [
        ("https://api.openai.com/V1", False),
        ("https://api.openai.com/V1/", False),
        ("HTTPS://API.OPENAI.COM/v1/", True),
    ])
    def test_only_scheme_and_host_are_case_insensitive(self, sdk_context_wire, provider, base, native):
        wire = sdk_context_wire
        wire.cfg["provider"] = provider
        wire.cfg[provider]["base_url"] = base
        assert wire.run()["success"]
        fields = wire.fields()
        path = "/v1/" if native else "/V1/"
        assert str(wire.requests[0].url) == "https://api.openai.com" + path + "audio/transcriptions"
        assert fields["prompt"] == ["native" if native else "generic"]
        assert fields.get("keywords[]") == (["Hermes"] if native else None)
        assert fields["languages[]"] == ["de"]
        assert "language" not in fields

    @pytest.mark.parametrize("provider", ["openai", "deepinfra", "nous"])
    @pytest.mark.parametrize("origin", ["hook", "generic", "native"])
    @pytest.mark.parametrize("length", [5000, 5001])
    def test_raw_effective_prompt_is_validated_before_upload(
        self, sdk_context_wire, provider, origin, length,
    ):
        wire = sdk_context_wire
        wire.cfg["provider"] = provider
        prompt = "ä" * (length - 4) + "TAIL"
        if origin == "hook":
            wire.hooks.append({"prompt": prompt})
        elif origin == "native":
            wire.cfg["openai"]["prompt"] = prompt
        else:
            del wire.cfg["openai"]["prompt"]
            wire.cfg["prompt"] = prompt
        result = wire.run()
        assert wire.constructors[0]["api_key"] == "offline-" + provider
        if length == 5001:
            assert not result["success"]
            assert "5000" in result["error"]
            assert wire.requests == []
        else:
            assert result["success"]
            assert wire.fields()["prompt"] == [prompt[-896:]]
            assert str(wire.requests[0].url) == "https://api.openai.com/v1/audio/transcriptions"

    @pytest.mark.parametrize("provider,model_source", [
        ("openai", "config"), ("openai", "hook"), ("nous", "config"),
        ("deepinfra", "config"), ("deepinfra", "hook"), ("deepinfra", "catalog"),
    ])
    @pytest.mark.parametrize("clear_first", [True, False])
    def test_hook_clear_survives_model_resolution(
        self, sdk_context_wire, provider, model_source, clear_first,
    ):
        wire = sdk_context_wire
        wire.cfg["provider"] = provider
        section = "openai" if provider == "nous" else provider
        wire.cfg["prompt"] = "g" * 5001
        wire.cfg["openai"]["prompt"] = "n" * 5001
        if model_source == "hook":
            wire.cfg[section]["model"] = "whisper-1"
            model_hook = {"model": "gpt-transcribe"}
        else:
            model_hook = {"language": "it"}
        if model_source == "catalog":
            del wire.cfg[section]["model"]
        wire.hooks.extend([{"prompt": ""}, model_hook] if clear_first else [model_hook, {"prompt": ""}])
        result = wire.run()
        assert result["success"]
        assert result["provider"] == ("openai" if provider == "nous" else provider)
        fields = wire.fields()
        assert fields["model"] == ["gpt-transcribe"]
        assert "prompt" not in fields
        assert fields["keywords[]"] == ["Hermes"]
        assert fields["languages[]"] == (["de"] if model_source == "hook" else ["it"])
        assert "language" not in fields

    @pytest.mark.parametrize("origin", ["clear", "generic", "hook"])
    @pytest.mark.parametrize("model", ["gpt-transcribe", "vendor/whisper"])
    def test_custom_deepinfra_endpoint_retains_legacy_context(
        self, sdk_context_wire, monkeypatch, origin, model,
    ):
        wire = sdk_context_wire
        wire.cfg["deepinfra"].update(base_url="https://custom.example/v1", model=model)
        monkeypatch.setenv("DEEPINFRA_BASE_URL", "https://api.openai.com/v1")
        wire.cfg["openai"].update(prompt="n" * 5001, keywords=[1])
        prompt = "g" * 4997 + "TAIL"
        wire.cfg["prompt"] = prompt
        if origin == "clear":
            wire.hooks.append({"prompt": ""})
        elif origin == "hook":
            prompt = "h" * 4997 + "TAIL"
            wire.hooks.append({"prompt": prompt})
        assert wire.run()["success"]
        fields = wire.fields()
        assert fields.get("prompt") == (None if origin == "clear" else [prompt[-896:]])
        assert "keywords[]" not in fields
        language_field = "languages[]" if model == "gpt-transcribe" else "language"
        assert fields[language_field] == ["de"]
        assert ("language" not in fields) if model == "gpt-transcribe" else ("languages[]" not in fields)
        assert str(wire.requests[0].url) == "https://custom.example/v1/audio/transcriptions"
        assert wire.constructors[0]["api_key"] == "offline-deepinfra"
        assert set(wire.key_calls) == {("DEEPINFRA_API_KEY", "deepinfra")}

    @pytest.mark.parametrize("intent", ["clear", "hook", "generic"])
    def test_deepinfra_env_url_and_catalog_model_preserve_raw_intent(
        self, sdk_context_wire, monkeypatch, intent,
    ):
        wire = sdk_context_wire
        wire.cfg["deepinfra"].pop("base_url")
        wire.cfg["deepinfra"].pop("model")
        monkeypatch.setenv("DEEPINFRA_BASE_URL", "https://api.openai.com/v1")
        wire.cfg["prompt"] = "g" * 5001
        wire.cfg["openai"].pop("prompt")
        if intent != "generic":
            wire.hooks.append({"prompt": "" if intent == "clear" else "h" * 5001})
        result = wire.run()
        if intent == "clear":
            assert result["success"]
            assert "prompt" not in wire.fields()
            assert str(wire.requests[0].url) == "https://api.openai.com/v1/audio/transcriptions"
        else:
            assert not result["success"]
            assert "5000" in result["error"]
            assert wire.requests == []


class TestTranscribeLocalCommand:
    def test_command_provider_uses_sanitized_child_env(self, monkeypatch):
        """Salvage of #56332: command STT must not inherit Hermes secrets."""
        monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "sk-vision")
        monkeypatch.setenv("GATEWAY_RELAY_SECRET", "relay-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("MY_SAFE_STT_VAR", "keep")

        captured = {}

        class _Stream:
            def read(self, size):
                return ""

        class Proc:
            returncode = 0
            stdout = _Stream()
            stderr = _Stream()

            def wait(self, timeout=None):
                return 0

        def fake_popen(command, **kwargs):
            captured["env"] = kwargs["env"]
            return Proc()

        monkeypatch.setattr("tools.tts_command_provider.subprocess.Popen", fake_popen)

        from tools.transcription_command import _run_command_stt

        result = _run_command_stt("echo hi", timeout=1)

        assert result.returncode == 0
        env = captured["env"]
        assert "AUXILIARY_VISION_API_KEY" not in env
        assert "GATEWAY_RELAY_SECRET" not in env
        assert "OPENAI_API_KEY" not in env
        assert env["MY_SAFE_STT_VAR"] == "keep"

    def test_local_whisper_subprocess_uses_sanitized_env(
        self, monkeypatch, sample_wav, tmp_path
    ):
        """Sibling path: local whisper subprocess.run also scrubbed (#56332 gap)."""
        monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "sk-vision")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("MY_SAFE_LOCAL_STT", "keep")
        monkeypatch.setenv(
            "HERMES_LOCAL_STT_COMMAND",
            "whisper {input_path} --model {model} --output_dir {output_dir} --language {language}",
        )

        captured = {}
        out_dir = tmp_path / "local-out"
        out_dir.mkdir()
        (out_dir / "transcript.txt").write_text("hello", encoding="utf-8")

        def fake_tempdir(prefix=None):
            class _TempDir:
                def __enter__(self_inner):
                    return str(out_dir)

                def __exit__(self_inner, *exc):
                    return False

            return _TempDir()

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs.get("env")
            class R:
                returncode = 0
            return R()

        monkeypatch.setattr("tools.transcription_local.tempfile.TemporaryDirectory", fake_tempdir)
        monkeypatch.setattr("tools.transcription_audio.subprocess.run", fake_run)
        monkeypatch.setattr(
            "tools.transcription_local._prepare_local_audio",
            lambda *a, **k: (str(sample_wav), None),
        )

        from tools.transcription_tools import _transcribe_local_command

        result = _transcribe_local_command(str(sample_wav), "base")
        assert result["success"] is True
        env = captured["env"]
        assert env is not None
        assert "AUXILIARY_VISION_API_KEY" not in env
        assert "OPENAI_API_KEY" not in env
        assert env["MY_SAFE_LOCAL_STT"] == "keep"

    def test_command_fallback_with_template(self, monkeypatch, sample_ogg, tmp_path):
        out_dir = tmp_path / "local-out"
        out_dir.mkdir()

        monkeypatch.setenv(
            "HERMES_LOCAL_STT_COMMAND",
            "whisper {input_path} --model {model} --output_dir {output_dir} --language {language}",
        )
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "en")

        def fake_tempdir(prefix=None):
            class _TempDir:
                def __enter__(self_inner):
                    return str(out_dir)

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _TempDir()

        def fake_run(cmd, *args, **kwargs):
            assert isinstance(cmd, list)
            (out_dir / "test.txt").write_text("hello from local command\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("tools.transcription_local.tempfile.TemporaryDirectory", fake_tempdir)
        monkeypatch.setattr("tools.transcription_audio._find_ffmpeg_binary", lambda: "/opt/homebrew/bin/ffmpeg")
        monkeypatch.setattr("tools.transcription_audio.subprocess.run", fake_run)

        from tools.transcription_tools import _transcribe_local_command

        result = _transcribe_local_command(sample_ogg, "base")

        assert result["success"] is True
        assert result["transcript"] == "hello from local command"
        assert result["provider"] == "local_command"


# ============================================================================
# _transcribe_local — additional tests
# ============================================================================

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("faster_whisper"),
    reason="faster_whisper not installed",
)
class TestLocalModelLoading:
    def test_cached_model_load_never_uses_online_resolution(self):
        cached_model = object()

        with patch("faster_whisper.WhisperModel", return_value=cached_model) as model_cls:
            from tools.transcription_local import _create_whisper_model

            assert _create_whisper_model("base", device="cpu", compute_type="int8") is cached_model

        model_cls.assert_called_once_with(
            "base", local_files_only=True, device="cpu", compute_type="int8"
        )

    @pytest.mark.parametrize("download_error", [None, "Got: ConnectTimeout: [Errno 110] Connection timed out"])
    def test_cache_miss_falls_back_with_actionable_download_failure(self, download_error):
        from tools.transcription_local import _create_whisper_model, _hub_cache_miss_error

        LocalEntryNotFoundError = _hub_cache_miss_error()
        if download_error:
            download_error = LocalEntryNotFoundError(download_error)

        downloaded_model = object()
        online_result = download_error or downloaded_model
        side_effect = [LocalEntryNotFoundError("not cached"), online_result]
        with patch("faster_whisper.WhisperModel", side_effect=side_effect) as model_cls:
            if download_error:
                with pytest.raises(RuntimeError) as exc_info:
                    _create_whisper_model("base", device="auto", compute_type="auto")
                assert "HF_ENDPOINT" in str(exc_info.value)
                assert "HF_HUB_DISABLE_XET=1" in str(exc_info.value)
            else:
                assert _create_whisper_model(
                    "base", device="auto", compute_type="auto"
                ) is downloaded_model

        assert model_cls.call_args_list == [
            call("base", local_files_only=True, device="auto", compute_type="auto"),
            call("base", local_files_only=False, device="auto", compute_type="auto"),
        ]

    def test_partial_cache_is_treated_as_a_cache_miss(self):
        # An interrupted first download leaves refs/main + a snapshot without model.bin;
        # snapshot_download(local_files_only=True) returns that folder and ctranslate2
        # raises RuntimeError, so the online path must still run.
        from tools.transcription_local import _create_whisper_model

        downloaded_model = object()
        side_effect = [RuntimeError("Unable to open file 'model.bin' in model '/cache/snap'"), downloaded_model]
        with patch("faster_whisper.WhisperModel", side_effect=side_effect) as model_cls:
            assert _create_whisper_model("base", device="cpu", compute_type="int8") is downloaded_model

        assert [c.kwargs["local_files_only"] for c in model_cls.call_args_list] == [True, False]


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("faster_whisper"),
    reason="faster_whisper not installed",
)
class TestTranscribeLocalExtended:
    def test_model_reuse_on_second_call(self, tmp_path):
        """Second call with same model should NOT reload the model."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls = MagicMock(return_value=mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            _transcribe_local(str(audio), "base")
            _transcribe_local(str(audio), "base")

        # WhisperModel should be created only once
        assert mock_whisper_cls.call_count == 1

    def test_config_device_and_compute_type_passed_to_whisper(self, tmp_path):
        """User-configured device and compute_type should be forwarded to WhisperModel.

        Regression test for #8319: these values were hardcoded to "auto".
        """
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls = MagicMock(return_value=mock_model)

        fake_config = {
            "local": {
                "device": "cpu",
                "compute_type": "float32",
            }
        }

        from tools.transcription_local import _load_local_whisper_model

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._load_local_whisper_model", wraps=_load_local_whisper_model) as mock_load, \
             patch("tools.transcription_local._should_force_faster_whisper_cpu", return_value=False), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None), \
             patch("tools.transcription_tools._load_stt_config", return_value=fake_config):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        mock_load.assert_called_once_with("base", device="cpu", compute_type="float32")
        mock_whisper_cls.assert_called_once_with(
            "base", local_files_only=True, device="cpu", compute_type="float32"
        )


    def test_cuda_out_of_memory_does_not_trigger_cpu_fallback(self, tmp_path):
        """'CUDA out of memory' is a real error, not a missing lib — surface it."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_whisper_cls = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        # Single call — no CPU retry, because OOM isn't a missing-lib symptom.
        assert mock_whisper_cls.call_count == 1
        assert result["success"] is False
        assert "CUDA out of memory" in result["error"]

    @staticmethod
    def _lazy_failure(message):
        """faster-whisper's transcribe() returns a lazy generator: the decode — and the CUDA
        dlopen-on-first-use — only fires while segments are iterated (#103793, #105295, #111929)."""
        def segments():
            raise RuntimeError(message)
            yield  # pragma: no cover
        return segments()

    def test_iteration_time_cuda_dlopen_retries_on_cpu(self, tmp_path):
        """A missing CUDA library raised while ITERATING segments must evict the cached model and
        retry on CPU/int8, not surface as a hard failure (Windows: cublas64_12.dll)."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")
        info = MagicMock(language="en", duration=1.0)

        cuda_model = MagicMock()
        cuda_model.transcribe.return_value = (
            self._lazy_failure("Library cublas64_12.dll is not found or cannot be loaded"), info)
        cpu_segment = MagicMock(text="hi", no_speech_prob=0.0, avg_logprob=0.0)
        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = ([cpu_segment], info)
        mock_whisper_cls = MagicMock(side_effect=[cuda_model, cpu_model])

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True, result.get("error")
        assert result["transcript"] == "hi"
        assert mock_whisper_cls.call_count == 2
        retry_kwargs = mock_whisper_cls.call_args_list[1].kwargs
        assert (retry_kwargs["device"], retry_kwargs["compute_type"]) == ("cpu", "int8")

    def test_iteration_time_non_lib_error_surfaces_without_cpu_retry(self, tmp_path):
        """A real runtime failure during iteration must NOT trigger the CPU retry."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        cuda_model = MagicMock()
        cuda_model.transcribe.return_value = (self._lazy_failure("CUDA out of memory"), MagicMock())
        mock_whisper_cls = MagicMock(return_value=cuda_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is False
        assert "CUDA out of memory" in result["error"]
        assert mock_whisper_cls.call_count == 1


# ============================================================================
# Model auto-correction
# ============================================================================

class TestModelAutoCorrection:
    def test_groq_corrects_openai_model(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq, DEFAULT_GROQ_STT_MODEL
            _transcribe_groq(sample_wav, "whisper-1")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_GROQ_STT_MODEL


    def test_unknown_model_passes_through_groq(self, monkeypatch, sample_wav):
        """A model not in either known set should not be overridden."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "my-custom-model")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "my-custom-model"


# ============================================================================
# _validate_audio_file — edge cases
# ============================================================================

class TestValidateAudioFileEdgeCases:
    def test_directory_is_not_a_file(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file
        # tmp_path itself is a directory with an .ogg-ish name? No.
        # Create a directory with a valid audio extension
        d = tmp_path / "audio.ogg"
        d.mkdir()
        result = _validate_audio_file(str(d))
        assert result is not None
        assert "not a file" in result["error"]

    def test_symlink_with_supported_extension_is_rejected(self, tmp_path):
        if not hasattr(os, "symlink"):
            pytest.skip("symlinks are not supported on this platform")

        target = tmp_path / "target.txt"
        target.write_bytes(b"not audio")
        link = tmp_path / "linked.wav"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

        from tools.transcription_tools import _validate_audio_file
        result = _validate_audio_file(str(link))
        assert result is not None
        assert "symbolic link" in result["error"]


    def test_all_supported_formats_accepted(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file
        from tools.transcription_common import SUPPORTED_FORMATS
        for fmt in SUPPORTED_FORMATS:
            f = tmp_path / f"test{fmt}"
            f.write_bytes(b"data")
            assert _validate_audio_file(str(f)) is None, f"Format {fmt} should be accepted"

# ============================================================================
# transcribe_audio — end-to-end dispatch
# ============================================================================

class TestTranscribeAudioDispatch:
    def test_oversized_local_file_reaches_dispatcher(self, oversized_wav):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "local"}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(oversized_wav)

        assert result["success"] is True
        mock_local.assert_called_once()


    def test_no_provider_returns_error(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="none"):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is False
        assert "No STT provider" in result["error"]
        assert "faster-whisper" in result["error"]
        assert "GROQ_API_KEY" in result["error"]


    def test_silk_symlink_is_rejected_before_preprocessing(self, tmp_path):
        """A Silk symlink must not reach the decoder before path safety validation."""
        if not hasattr(os, "symlink"):
            pytest.skip("symlinks are not supported on this platform")

        target = tmp_path / "voice.silk"
        target.write_bytes(b"\x02#!SILK_V3fake")
        link = tmp_path / "linked.silk"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

        with patch(
            "tools.transcription_tools._prepare_audio_for_transcription", create=True
        ) as mock_prepare:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(link))

        assert result["success"] is False
        assert "symbolic link" in result["error"]
        mock_prepare.assert_not_called()


    def test_config_local_model_used(self, sample_ogg):
        config = {"local": {"model": "small"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_local.call_args[0][1] == "small"

# ============================================================================
# _transcribe_mistral
# ============================================================================


@pytest.fixture
def mock_mistral_module():
    """Inject a fake mistralai module into sys.modules for testing."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_mistral_cls = MagicMock(return_value=mock_client)
    fake_module = MagicMock()
    fake_module.Mistral = mock_mistral_cls
    with patch.dict("sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}):
        yield mock_client


class TestTranscribeMistral:
    def test_successful_transcription(self, monkeypatch, sample_ogg, mock_mistral_module):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        mock_result = MagicMock()
        mock_result.text = "hello from mistral"
        mock_mistral_module.audio.transcriptions.complete.return_value = mock_result

        from tools.transcription_tools import _transcribe_mistral
        result = _transcribe_mistral(sample_ogg, "voxtral-mini-latest")

        assert result["success"] is True
        assert result["transcript"] == "hello from mistral"
        assert result["provider"] == "mistral"
        mock_mistral_module.audio.transcriptions.complete.assert_called_once()
        mock_mistral_module.__exit__.assert_called_once()

    def test_api_error_returns_failure(self, monkeypatch, sample_ogg, mock_mistral_module):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.transcriptions.complete.side_effect = RuntimeError("secret-key-leaked")

        from tools.transcription_tools import _transcribe_mistral
        result = _transcribe_mistral(sample_ogg, "voxtral-mini-latest")

        assert result["success"] is False
        assert "RuntimeError" in result["error"]
        assert "secret-key-leaked" not in result["error"]

# ============================================================================
# _get_provider — Mistral
# ============================================================================

class TestGetProviderMistral:
    """Mistral-specific provider selection tests."""

    def test_mistral_explicit_no_sdk_returns_none(self, monkeypatch):
        """Explicit mistral with key but no SDK returns none."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "mistral"}) == "none"

    def test_auto_detect_mistral_after_openai(self, monkeypatch):
        """Auto-detect: mistral is tried after openai when both are unavailable."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "mistral"

# ============================================================================
# transcribe_audio — Mistral dispatch
# ============================================================================

class TestTranscribeAudioMistralDispatch:
    def test_config_mistral_model_used(self, sample_ogg):
        config = {"provider": "mistral", "mistral": {"model": "voxtral-mini-2602"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="mistral"), \
             patch("tools.transcription_tools._transcribe_mistral",
                   return_value={"success": True, "transcript": "hi"}) as mock_mistral:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_mistral.call_args[0][1] == "voxtral-mini-2602"

# ============================================================================
# _transcribe_xai
# ============================================================================


@pytest.fixture
def mock_xai_http_module():
    """Inject a fake tools.xai_http module for testing."""
    fake_module = MagicMock()
    fake_module.hermes_xai_user_agent = MagicMock(return_value="hermes-xai/test")
    with patch.dict("sys.modules", {"tools.xai_http": fake_module}):
        yield fake_module


class TestTranscribeXAI:
    def test_successful_transcription(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "bonjour le monde",
            "language": "fr",
            "duration": 3.2,
        }

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is True
        assert result["transcript"] == "bonjour le monde"
        assert result["provider"] == "xai"


    @pytest.mark.parametrize("rejected_status", [401])
    def test_retries_auth_rejection_with_refreshed_oauth_credentials(
        self, sample_ogg, mock_xai_http_module, rejected_status
    ):
        mock_xai_http_module.resolve_xai_http_credentials.side_effect = [
            {
                "api_key": "stale-oauth-token",
                "base_url": "https://api.x.ai/v1",
                "provider": "xai-oauth",
            },
            {
                "api_key": "fresh-oauth-token",
                "base_url": "https://api.x.ai/v1",
                "provider": "xai-oauth",
            },
        ]

        rejected = MagicMock()
        rejected.status_code = rejected_status
        rejected.json.return_value = {
            "error": {"message": "OAuth2 access token could not be validated"}
        }
        accepted = MagicMock()
        accepted.status_code = 200
        accepted.json.return_value = {
            "text": "fleet speech transcription proof",
            "language": "en",
            "duration": 2.1,
        }

        stt_config = {"provider": "xai"}
        with patch("tools.transcription_tools._load_stt_config", return_value=stt_config), \
             patch("tools.transcription_tools._get_provider", return_value="xai"), \
             patch("requests.post", side_effect=[rejected, accepted]) as mock_post:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result == {
            "success": True,
            "transcript": "fleet speech transcription proof",
            "provider": "xai",
        }
        assert mock_post.call_count == 2
        assert mock_post.call_args_list[0].kwargs["headers"]["Authorization"] == (
            "Bearer stale-oauth-token"
        )
        assert mock_post.call_args_list[1].kwargs["headers"]["Authorization"] == (
            "Bearer fresh-oauth-token"
        )
        assert mock_xai_http_module.resolve_xai_http_credentials.call_args_list == [
            call(),
            call(force_refresh=True, api_key_hint="stale-oauth-token"),
        ]


    def test_sends_language_and_format(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        # Explicitly set language via env to exercise the override chain
        # (config > env > DEFAULT_LOCAL_STT_LANGUAGE)
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "fr")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "test", "language": "fr", "duration": 1.0}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_xai
            _transcribe_xai(sample_ogg, "grok-stt")

        call_kwargs = mock_post.call_args
        data = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))
        assert data.get("language") == "fr"
        assert data.get("format") == "true"

    def test_oauth_credentials_ignore_stt_base_url_override(
        self,
        monkeypatch,
        sample_ogg,
        mock_xai_http_module,
    ):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("XAI_STT_BASE_URL", "https://attacker.example/v1")
        mock_xai_http_module.resolve_xai_http_credentials.return_value = {
            "provider": "xai-oauth",
            "api_key": "oauth-bearer-token",
            "base_url": "https://api.x.ai/v1",
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "test", "language": "en", "duration": 1.0}

        with patch(
            "tools.transcription_tools._load_stt_config",
            return_value={"xai": {"base_url": "https://attacker.example/config"}},
        ), patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_xai

            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is True
        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert url == "https://api.x.ai/v1/stt"
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer oauth-bearer-token"

# ============================================================================
# _get_provider — xAI
# ============================================================================

class TestGetProviderXAI:
    """xAI-specific provider selection tests."""

    def test_auto_detect_xai_after_mistral(self, monkeypatch):
        """Auto-detect: xai is tried after mistral when all above are unavailable."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "xai"

# ============================================================================
# transcribe_audio — xAI dispatch
# ============================================================================

class TestTranscribeAudioXAIDispatch:
    def test_model_default_is_grok_stt(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "xai"}), \
             patch("tools.transcription_tools._get_provider", return_value="xai"), \
             patch("tools.transcription_tools._transcribe_xai",
                   return_value={"success": True, "transcript": "hi"}) as mock_xai:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_xai.call_args[0][1] == "grok-stt"

# ============================================================================
# _transcribe_elevenlabs
# ============================================================================

class TestTranscribeElevenLabs:
    def test_successful_transcription(self, monkeypatch, sample_ogg):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "hello from elevenlabs"}

        config = {
            "elevenlabs": {
                "language_code": "eng",
                "tag_audio_events": True,
                "diarize": True,
            }
        }
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_elevenlabs
            result = _transcribe_elevenlabs(sample_ogg, "scribe_v2")

        assert result["success"] is True
        assert result["transcript"] == "hello from elevenlabs"
        assert result["provider"] == "elevenlabs"
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["headers"]["xi-api-key"] == "eleven-test-key"
        assert call_kwargs["data"]["model_id"] == "scribe_v2"
        assert call_kwargs["data"]["language_code"] == "eng"
        assert call_kwargs["data"]["tag_audio_events"] == "true"
        assert call_kwargs["data"]["diarize"] == "true"

# ============================================================================
# _get_provider — ElevenLabs
# ============================================================================

class TestGetProviderElevenLabs:
    """ElevenLabs-specific provider selection tests."""

    def test_auto_detect_elevenlabs_after_xai(self, monkeypatch):
        """Auto-detect: elevenlabs is tried after xai when all above are unavailable."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "elevenlabs"

# ============================================================================
# transcribe_audio — ElevenLabs dispatch
# ============================================================================

class TestTranscribeAudioElevenLabsDispatch:
    def test_config_elevenlabs_model_used(self, sample_ogg):
        config = {"provider": "elevenlabs", "elevenlabs": {"model_id": "scribe_v1"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="elevenlabs"), \
             patch("tools.transcription_tools._transcribe_elevenlabs",
                   return_value={"success": True, "transcript": "hi"}) as mock_elevenlabs:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_elevenlabs.call_args[0][1] == "scribe_v1"

# ============================================================================
# _extract_transcript_text
# ============================================================================

class TestExtractTranscriptText:
    def test_strips_qwen3_asr_language_envelope(self):
        from tools.transcription_cloud import _extract_transcript_text

        result = _extract_transcript_text(
            "language zh\n<audio_language>zh</audio_language>\n<asr_text>你好，世界",
        )

        assert result == "你好，世界"

    def test_keeps_non_envelope_marker_literal(self):
        from tools.transcription_cloud import _extract_transcript_text

        result = _extract_transcript_text(
            "The user literally said <asr_text> while reading markup.",
        )

        assert result == "The user literally said <asr_text> while reading markup."


# Shell safety — shlex.split on auto-detected templates
# ============================================================================
class TestShellSafety:
    def test_auto_detected_template_is_shlex_safe(self, monkeypatch):
        """Auto-detected whisper command should be safely splittable."""
        import shlex
        monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)
        monkeypatch.setattr(
            "tools.transcription_local._find_whisper_binary",
            lambda: "/usr/bin/whisper",
        )
        from tools.transcription_local import _get_local_command_template
        template = _get_local_command_template()
        assert template is not None
        cmd = template.format(
            input_path=shlex.quote("/tmp/test.wav"),
            output_dir=shlex.quote("/tmp/out"),
            language=shlex.quote("en"),
            model=shlex.quote("base"),
        )
        parts = shlex.split(cmd)
        assert parts[0] == "/usr/bin/whisper"
        assert "/tmp/test.wav" in parts

    def test_env_var_template_metacharacters_are_literal_argv(
        self, monkeypatch, sample_wav, tmp_path
    ):
        from hermes_cli._subprocess_compat import windows_hide_flags
        from tools.transcription_tools import (
            LOCAL_STT_COMMAND_ENV,
            _transcribe_local_command,
        )

        output_dir = tmp_path / "transcript-output"
        output_dir.mkdir()
        monkeypatch.setenv(
            LOCAL_STT_COMMAND_ENV,
            (
                "whisper {input_path} ; printf injected | tee log.txt "
                "&& echo $(id) `whoami` --output_dir {output_dir}"
            ),
        )

        def fake_tempdir(prefix=None):
            class _TempDir:
                def __enter__(self):
                    return str(output_dir)

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _TempDir()

        invocation = {}

        def fake_run(command, **kwargs):
            invocation["command"] = command
            invocation["kwargs"] = kwargs
            (output_dir / "transcript.txt").write_text("safe", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(
            "tools.transcription_local.tempfile.TemporaryDirectory", fake_tempdir
        )
        monkeypatch.setattr("tools.transcription_audio.subprocess.run", fake_run)

        result = _transcribe_local_command(sample_wav, "base")

        assert result["transcript"] == "safe"
        assert invocation["command"] == [
            "whisper",
            sample_wav,
            ";",
            "printf",
            "injected",
            "|",
            "tee",
            "log.txt",
            "&&",
            "echo",
            "$(id)",
            "`whoami`",
            "--output_dir",
            str(output_dir),
        ]
        assert invocation["kwargs"].pop("env") is not None
        assert invocation["kwargs"] == {
            "check": True,
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": 300,
            "stdin": subprocess.DEVNULL,
            "creationflags": windows_hide_flags(),
        }


class TestLocalModelLock:
    """#24767 — concurrent first-use must not double-load the whisper model."""

    def test_concurrent_transcribe_loads_model_once(self, tmp_path):
        import threading
        from tools.transcription_tools import _transcribe_local

        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        seg = MagicMock()
        seg.text = "hello"
        info = MagicMock()
        info.language = "en"
        info.duration = 1.0

        load_count = 0
        load_started = threading.Event()

        def slow_load(model_name, device="auto", compute_type="auto"):
            nonlocal load_count
            load_count += 1
            load_started.set()
            import time
            time.sleep(0.05)
            model = MagicMock()
            model.transcribe.return_value = ([seg], info)
            return model

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._load_local_whisper_model", side_effect=slow_load), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            threads = [
                threading.Thread(target=_transcribe_local, args=(str(audio), "base"))
                for _ in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert load_count == 1


class TestLocalBaseUrlNoApiKey:
    """#25193 — empty api_key with a local base_url should not raise."""

    def test_local_base_url_returns_placeholder_key(self):
        from tools.transcription_tools import _resolve_openai_audio_client_config
        with patch(
            "tools.transcription_tools._load_stt_config",
            return_value={"openai": {"base_url": "http://localhost:8504/v1"}},
        ):
            api_key, base_url = _resolve_openai_audio_client_config()
        assert api_key == "not-needed"
        assert base_url == "http://localhost:8504/v1"


    def test_is_local_or_private_url(self):
        from tools.transcription_cloud import _is_local_or_private_url
        assert _is_local_or_private_url("http://localhost:8504/v1")
        assert _is_local_or_private_url("http://127.0.0.1:9000")
        assert _is_local_or_private_url("http://10.0.0.5/v1")
        assert _is_local_or_private_url("http://stt.internal/v1")
        assert not _is_local_or_private_url("https://api.openai.com/v1")
        assert not _is_local_or_private_url("")


# =====================================================================


# CAF (iMessage voice note) conversion tests
# ============================================================================

class TestCafConversion:
    """Tests for _convert_caf_to_wav and CAF dispatch in transcribe_audio."""

    def test_convert_caf_with_ffmpeg(self, tmp_path, monkeypatch):
        """_convert_caf_to_wav uses ffmpeg when available."""
        caf_path = tmp_path / "voice.caf"
        caf_path.write_bytes(b"caff\x00" * 20)
        wav_path = str(tmp_path / "voice.wav")

        def fake_run(cmd, **kwargs):
            Path(wav_path).write_bytes(b"RIFF\x00\x00\x00\x00")
            return MagicMock(returncode=0)

        monkeypatch.setattr(
            "tools.transcription_audio._find_ffmpeg_binary",
            lambda: "/usr/bin/ffmpeg",
        )
        monkeypatch.setattr(subprocess, "run", fake_run)

        from tools.transcription_tools import _convert_caf_to_wav
        result = _convert_caf_to_wav(str(caf_path))
        assert result == wav_path
        assert Path(result).exists()


    def test_transcribe_caf_not_converted_for_local(self, tmp_path, monkeypatch):
        """CAF conversion is skipped for local provider (native handling)."""
        caf_path = tmp_path / "voice.caf"
        caf_path.write_bytes(b"caff\x00" * 20)

        with patch("tools.transcription_tools._load_stt_config",
                   return_value={"provider": "local"}), \
             patch("tools.transcription_tools._get_provider",
                   return_value="local"), \
             patch("tools.transcription_tools._convert_caf_to_wav") as mock_convert, \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(caf_path))

        assert result["success"] is True
        mock_convert.assert_not_called()


class TestTranscribeCredentialReadGuard:
    """transcribe_audio must refuse credential/secret stores before dispatch."""

    def test_transcribe_audio_blocks_credential_read(self, tmp_path):
        """A ``.env`` (secret-bearing) file is refused up front, so its
        plaintext is never shipped to an external STT provider — mirroring the
        read guard added to image-gen (587be5b5b) and xAI video-gen
        (104232979)."""
        from tools.transcription_tools import transcribe_audio
        from agent.file_safety import get_read_block_error

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-secret\n")

        expected = get_read_block_error(str(env_file))
        assert expected, "test setup: a .env file should be read-blocked"

        result = transcribe_audio(str(env_file))

        assert result["success"] is False
        # The error is the shared read-guard message, not an audio-validation
        # or provider error — proving the guard fired before dispatch.
        assert result["error"] == expected


class TestRunCommandSttIdleTimeout:
    """_run_command_stt uses a progress-based idle timeout (mirrors TTS runner)."""

    @staticmethod
    def _shell_command(*args):
        import shlex
        if os.name == "nt":
            return subprocess.list2cmdline(list(args))
        return " ".join(shlex.quote(str(arg)) for arg in args)

    def test_stderr_progress_extends_beyond_timeout(self, tmp_path):
        """A slow-but-alive command that keeps emitting output survives an
        idle timeout shorter than its total runtime."""
        from tools.transcription_command import _run_command_stt

        script = tmp_path / "progress_then_exit.py"
        # Emit immediately, then keep the longer heartbeat sequence. Its 4.8s
        # runtime exceeds the 2s idle window, so a pass proves progress extension.
        script.write_text(
            "\n".join([
                "import sys, time",
                "print('tick 0', file=sys.stderr, flush=True)",
                "for idx in range(1, 9):",
                "    time.sleep(0.6)",
                "    print(f'tick {idx}', file=sys.stderr, flush=True)",
                "print('done', flush=True)",
            ]),
            encoding="utf-8",
        )

        # Leave room for interpreter startup on a cold or sandboxed host; total runtime
        # still exceeds the idle window, so stderr must extend the deadline.
        result = _run_command_stt(
            self._shell_command(sys.executable, "-u", str(script)),
            timeout=2.0,
        )

        assert result.returncode == 0
        assert "tick 3" in result.stderr
        assert "tick 8" in result.stderr
        assert "done" in result.stdout

    def test_silent_stall_still_times_out(self, tmp_path):
        """A silently stalled command is killed once the idle window elapses,
        and pre-stall output is preserved on the TimeoutExpired."""
        from tools.transcription_command import _run_command_stt

        script = tmp_path / "progress_then_hang.py"
        script.write_text(
            "\n".join([
                "import sys, time",
                "print('starting pass 1', file=sys.stderr, flush=True)",
                "time.sleep(30)",
            ]),
            encoding="utf-8",
        )

        # Same budget rule as the progress test above: the idle window must
        # comfortably exceed process spawn latency on a loaded runner, or the
        # child is killed before its first stderr line is ever read and the
        # pre-stall-output assertion fails spuriously. 30s of silence still
        # trips a 0.25s window by a wide margin.
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            _run_command_stt(
                self._shell_command(sys.executable, "-u", str(script)),
                timeout=0.25,
            )

        assert "starting pass 1" in (excinfo.value.stderr or "")


# ============================================================================
# Explicit openai selection keeps its selection-specific error (#93045)
# ============================================================================

class TestExplicitOpenaiSelectionError:
    """A managed-route outage must not be reported as generic setup guidance.

    When ``_resolve_openai_audio_client_config()`` raises its
    selection-specific ValueError (managed openai-audio gateway unavailable,
    with the ``hermes tools`` remediation for managed-Nous users), the old
    boolean probe flattened it into False — the log said "no API key" and
    the transcription result returned the all-provider install hint,
    pointing operators at unrelated setup instead of their managed route.
    """

    def _no_openai_credentials(self, monkeypatch):
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            "tools.tool_backend_helpers.resolve_openai_audio_api_key",
            lambda: None,
        )
        monkeypatch.setattr(
            "tools.managed_tool_gateway.resolve_managed_tool_gateway",
            lambda vendor: None,
        )

    def test_get_provider_openai_none_not_generic_when_managed_route_down(
        self, monkeypatch, caplog
    ):
        self._no_openai_credentials(monkeypatch)
        monkeypatch.setattr(
            "tools.tool_backend_helpers.managed_nous_tools_enabled", lambda: True
        )
        monkeypatch.setattr(
            "tools.transcription_tools._load_stt_config", lambda: {}
        )
        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False):
            from tools.transcription_tools import _get_provider

            with caplog.at_level("WARNING"):
                result = _get_provider({"provider": "openai"})

        assert result == "none"
        warning = caplog.records[-1].getMessage()
        assert "unavailable" in warning
        # The selection-specific blocker is named, not a bare API-key hint.
        assert "managed" in warning or "gateway" in warning
        assert "no API key available" not in warning

    def test_dispatch_returns_selection_specific_error(self, monkeypatch):
        """The final transcription result carries the managed-route error and
        its hermes tools remediation instead of the all-provider install
        hint."""
        self._no_openai_credentials(monkeypatch)
        monkeypatch.setattr(
            "tools.tool_backend_helpers.managed_nous_tools_enabled", lambda: True
        )
        monkeypatch.setattr(
            "tools.transcription_tools._load_stt_config", lambda: {}
        )
        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch(
                 "tools.tool_backend_helpers.nous_tool_gateway_unavailable_message",
                 lambda what: f"managed route down for {what}; run `hermes tools`",
             ):
            from tools.transcription_tools import _dispatch_stt_provider

            result = _dispatch_stt_provider(
                "/tmp/nonexistent.wav", "none", {"provider": "openai"}
            )

        assert result["success"] is False
        assert "managed route down" in result["error"]
        assert "hermes tools" in result["error"]
        assert "No STT provider available" not in result["error"]

    def test_auto_detect_none_keeps_generic_hint(self, monkeypatch):
        """Auto-detect with no credentials at all still returns the generic
        all-provider hint — the selection-specific branch must not fire
        without an explicit provider choice."""
        self._no_openai_credentials(monkeypatch)
        monkeypatch.setattr(
            "tools.transcription_tools._load_stt_config", lambda: {}
        )
        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch(
                 "tools.transcription_tools._has_local_command", return_value=False
             ), \
             patch(
                 "tools.transcription_tools._try_lazy_install_stt",
                 return_value=False,
             ):
            from tools.transcription_tools import _dispatch_stt_provider

            result = _dispatch_stt_provider("/tmp/x.wav", "none", {})

        assert result["success"] is False
        assert "No STT provider available" in result["error"]


class TestProviderOverride:
    def test_public_transcribe_audio_signature_does_not_expose_provider_override(self):
        import inspect
        from tools.transcription_tools import transcribe_audio
        from tools.registry import registry

        parameters = inspect.signature(transcribe_audio).parameters
        assert list(parameters) == ["file_path", "model", "source"]
        assert parameters["model"].default is None
        assert parameters["source"].default is None
        # Transcription is an internal service, not a registered model tool.
        assert registry.get_entry("transcribe_audio") is None
        assert registry.get_entry("_transcribe_audio_with_provider") is None
        with pytest.raises(TypeError, match="provider"):
            transcribe_audio("unused.wav", **{"provider": "openai"})

    @pytest.mark.parametrize("failure", [False, True])
    def test_override_is_exact_with_configured_backups(self, sample_wav, failure):
        from copy import deepcopy
        from tools.transcription_tools import _transcribe_audio_with_provider

        config = {"provider": "local", "fallback_providers": ["groq", "local"],
                  "cloud_trim_silence": False, "openai": {"model": "whisper-1", "api_key": "test-key"}}
        original = deepcopy(config)
        client = MagicMock()
        if failure:
            client.audio.transcriptions.create.side_effect = RuntimeError("selected route failed")
        else:
            client.audio.transcriptions.create.return_value = "selected route"
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=client), \
             patch("tools.transcription_tools._transcribe_groq") as groq, \
             patch("tools.transcription_tools._transcribe_local") as local, \
             patch("tools.transcription_tools._get_provider") as configured:
            result = _transcribe_audio_with_provider(sample_wav, provider="openai")
        assert result["success"] is not failure
        client.audio.transcriptions.create.assert_called_once()
        groq.assert_not_called()
        local.assert_not_called()
        configured.assert_not_called()
        assert config == original

    @pytest.mark.parametrize("stage", ["trim", "dispatch"])
    @pytest.mark.parametrize("explicit", [True, False])
    def test_explicit_route_contains_local_errors(self, sample_wav, stage, explicit):
        from copy import deepcopy
        from tools import transcription_tools as stt

        config = {"provider": "openai", "fallback_providers": ["groq", "local"],
                  "cloud_trim_silence": True}
        original = deepcopy(config)
        failure = RuntimeError("selected route preparation failed")
        with patch.object(stt, "_load_stt_config", return_value=config), \
             patch.object(stt, "_get_provider", return_value="openai") as configured, \
             patch.object(stt, "_trim_silence_for_cloud_stt", return_value=None,
                          side_effect=failure if stage == "trim" else None), \
             patch.object(stt, "_transcribe_openai", side_effect=failure) as selected, \
             patch.object(stt, "_transcribe_groq") as groq, \
             patch.object(stt, "_transcribe_local") as local:
            if explicit:
                result = stt._transcribe_audio_with_provider(sample_wav, provider="openai")
                assert result["success"] is False
                assert result["provider"] == "openai"
                assert "selected route preparation failed" in result["error"]
                configured.assert_not_called()
            else:
                with pytest.raises(RuntimeError, match="selected route preparation failed"):
                    stt.transcribe_audio(sample_wav)
            assert selected.call_count == int(stage == "dispatch")
            groq.assert_not_called()
            local.assert_not_called()
        assert config == original

    def test_override_and_default_calls_remain_isolated_while_interleaved(self, sample_wav):
        from concurrent.futures import ThreadPoolExecutor
        from copy import deepcopy
        from threading import Event
        from tools import transcription_tools as stt

        config = {"provider": "local", "local": {"model": "small"},
                  "fallback_providers": ["groq"], "cloud_trim_silence": False,
                  "openai": {"api_key": "test-key"}}
        original = deepcopy(config)
        entered, release = Event(), Event()
        client = MagicMock()

        def transcribe(**kwargs):
            entered.set()
            assert release.wait(5), "override was not released"
            return "override"

        client.audio.transcriptions.create.side_effect = transcribe
        with patch.object(stt, "_load_stt_config", return_value=config), \
             patch("tools.tool_backend_helpers.read_selection", return_value="local"), \
             patch.object(stt, "_HAS_FASTER_WHISPER", True), \
             patch.object(stt, "_HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=client), \
             patch.object(stt, "_transcribe_local", return_value={"success": True, "transcript": "default"}) as local, \
             ThreadPoolExecutor(max_workers=1) as executor:
            assert stt.transcribe_audio(sample_wav)["transcript"] == "default"
            future = executor.submit(stt._transcribe_audio_with_provider, sample_wav, provider="openai")
            try:
                assert entered.wait(5), "override never reached the selected API"
                assert config == original
                assert stt.transcribe_audio(sample_wav)["transcript"] == "default"
            finally:
                release.set()
            assert future.result(timeout=5)["transcript"] == "override"
            assert stt.transcribe_audio(sample_wav)["transcript"] == "default"
        assert local.call_args_list == [call(sample_wav, "small", language=None, prompt=None)] * 3
        client.audio.transcriptions.create.assert_called_once()
        assert config == original

    @pytest.mark.parametrize("guard, message", [
        ("disabled", "STT is disabled"), ("secret", ""),
        ("missing", "not found"), ("directory", "not a file"),
        ("symlink", "symbolic link"), ("unsupported", "Unsupported format"),
        ("oversized", "File too large"), ("silk_oversized", "File too large"),
    ])
    def test_override_cannot_bypass_native_guards(
        self, guard, message, sample_wav, oversized_wav, tmp_path,
    ):
        from tools.transcription_tools import _transcribe_audio_with_provider
        from agent.file_safety import get_read_block_error

        secret = tmp_path / ".env"
        secret.write_text("synthetic test data")
        unsupported = tmp_path / "audio.txt"
        unsupported.write_text("not audio")
        directory = tmp_path / "directory.wav"
        directory.mkdir()
        link = tmp_path / "linked.silk"
        link.symlink_to(sample_wav)
        silk = tmp_path / "oversized.silk"
        with silk.open("wb") as file:
            file.truncate(Path(oversized_wav).stat().st_size)
        paths = {"secret": secret, "missing": tmp_path / "absent.wav",
                 "directory": directory, "symlink": link, "unsupported": unsupported,
                 "oversized": oversized_wav, "silk_oversized": silk}
        path = str(paths.get(guard, sample_wav))
        config = {"provider": "local", "enabled": guard != "disabled",
                  "fallback_providers": ["groq"]}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._dispatch_stt_provider") as dispatch, \
             patch("tools.transcription_tools._get_provider") as configured, \
             patch("tools.transcription_audio._lazy_ensure_quietly") as install:
            result = _transcribe_audio_with_provider(path, provider="openai")
        assert result["success"] is False
        if guard == "secret":
            assert result["error"] == get_read_block_error(path)
        else:
            assert message in result["error"]
        dispatch.assert_not_called()
        configured.assert_not_called()
        install.assert_not_called()

    @pytest.mark.parametrize("prepared_name", ["invalid.txt", ".env"])
    def test_override_revalidates_prepared_input_and_cleans_it(self, sample_wav, tmp_path, prepared_name):
        from tools.transcription_tools import _transcribe_audio_with_provider

        work = tmp_path / "prepared"
        work.mkdir()
        prepared = work / prepared_name
        prepared.write_text("synthetic test data")
        with patch("tools.transcription_tools._prepare_audio_for_transcription",
                   return_value=(str(prepared), str(work), None)), \
             patch("tools.transcription_tools._dispatch_stt_provider") as dispatch:
            result = _transcribe_audio_with_provider(sample_wav, provider="openai")
        assert result["success"] is False
        dispatch.assert_not_called()
        assert not work.exists()
        assert Path(sample_wav).exists()

    @pytest.mark.parametrize("provider", [None, "openai"])
    def test_override_composes_with_native_caf_trim_cleanup(
        self, tmp_path, monkeypatch, provider,
    ):
        from copy import deepcopy
        from tools import transcription_tools as stt

        caf = tmp_path / "voice.caf"
        caf.write_bytes(b"caff" * 20)
        config = {"provider": "openai", "fallback_providers": ["local", "local"],
                  "local": {"model": "small"}, "openai": {"api_key": "test-key"}}
        original = deepcopy(config)
        outputs = []

        def run_audio_command(command, **kwargs):
            if command[0] == "test-ffprobe":
                duration = "20" if command[-1] == str(caf.with_suffix(".wav")) else "5"
                return subprocess.CompletedProcess(command, 0, stdout=duration)
            output = Path(command[-1])
            output.write_bytes(b"RIFF" * 20)
            outputs.append(output)
            return subprocess.CompletedProcess(command, 0, stdout="")

        monkeypatch.setattr("tools.transcription_audio._find_ffmpeg_binary", lambda: "test-ffmpeg")
        monkeypatch.setattr("tools.transcription_audio._find_ffprobe_binary", lambda: "test-ffprobe")
        monkeypatch.setattr("tools.transcription_audio._run_quiet", run_audio_command)
        client = MagicMock()
        client.audio.transcriptions.create.side_effect = RuntimeError("selected route failed")
        with patch.object(stt, "_load_stt_config", return_value=config), \
             patch.object(stt, "_HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=client), \
             patch.object(stt, "_transcribe_local", return_value={"success": True, "transcript": "fallback"}) as local:
            result = stt._transcribe_audio_with_provider(str(caf), model="whisper-1", source="gateway", provider=provider)
        assert result["success"] is False
        assert len(outputs) == 2
        assert client.audio.transcriptions.create.call_args.kwargs["file"].name == str(outputs[1])
        assert not outputs[1].parent.exists()
        assert caf.exists()
        local.assert_not_called()
        assert config == original

    def test_native_default_does_not_activate_configured_fallbacks(self, oversized_wav):
        from tools.transcription_tools import transcribe_audio

        config = {"provider": "local", "fallback_providers": ["openai"]}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.tool_backend_helpers.read_selection", return_value="local"), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._transcribe_local", return_value={"success": False}) as local, \
             patch("tools.transcription_tools._transcribe_openai") as cloud:
            result = transcribe_audio(oversized_wav)
        assert result["success"] is False
        assert result == {"success": False}
        local.assert_called_once()
        cloud.assert_not_called()
