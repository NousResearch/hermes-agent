"""Native keyword and prompt context coverage for transcription dispatch."""

import struct
import types
import wave
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.usefixtures("disable_lazy_stt_install")


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


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Keep host credentials and binary discovery outside these offline tests."""
    for name in (
        "VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
        "MISTRAL_API_KEY", "ELEVENLABS_API_KEY", "HERMES_LOCAL_STT_COMMAND",
        "HERMES_LOCAL_STT_LANGUAGE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("tools.transcription_audio._find_ffmpeg_binary", lambda: None)


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
