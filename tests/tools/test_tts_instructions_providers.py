"""Behaviour contracts for style instructions across TTS providers."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.tts_command_provider import _generate_command_tts
from tools.tts_tool_instructions import (
    TTS_INSTRUCTIONS_MAX_CHARS,
    _resolve_tts_instructions,
    _sanitize_tts_instructions,
    _tts_instructions_applied,
)
from tools.tts_tool_providers import (
    _compose_gemini_tts_prompt,
    _generate_elevenlabs,
    _generate_minimax_tts,
    _generate_xai_tts,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "MINIMAX_API_KEY",
        "MINIMAX_GROUP_ID",
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


class _FakeHttpResponse:
    """Minimal requests.Response stand-in for the non-streaming read path."""

    def __init__(self, content: bytes = b"mp3"):
        self.content = content

    def raise_for_status(self):
        pass


class TestSanitizeTtsInstructions:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("whisper", "whisper"),
            ("  calm and slow  ", "calm and slow"),
            ("[whisper]", "[whisper]"),
            ("<excited>", "<excited>"),
            ("{gravelly}", "{gravelly}"),
            ("calm\nand   slow", "calm and slow"),
            ("", ""),
            ("   ", ""),
            (None, ""),
            ("[]<>{}", "[]<>{}"),
        ],
    )
    def test_sanitize(self, raw, expected):
        assert _sanitize_tts_instructions(raw) == expected

    def test_long_value_is_capped(self):
        clean = _sanitize_tts_instructions("a" * 500)
        assert clean == "a" * TTS_INSTRUCTIONS_MAX_CHARS


class TestResolveTtsInstructions:
    def test_provider_default_overrides_global(self):
        cfg = {"instructions": "excited", "openai": {"instructions": "calm"}}
        assert _resolve_tts_instructions("openai", cfg, None) == "calm"

    def test_explicit_empty_override_suppresses_config(self):
        cfg = {"instructions": "excited"}
        assert _resolve_tts_instructions("edge", cfg, "") == ""

    def test_command_provider_default(self):
        cfg = {
            "providers": {
                "mycli": {"type": "command", "command": "x", "instructions": "soft"},
            },
        }
        assert _resolve_tts_instructions("mycli", cfg, None) == "soft"


class TestXaiRendering:
    def _run(self, tmp_path, monkeypatch, tts_config, text="hello there friend"):
        captured = {}

        def fake_post(url, headers, json, timeout, stream=False):
            captured["json"] = json
            return _FakeHttpResponse()

        monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
        monkeypatch.setattr("requests.post", fake_post)
        _generate_xai_tts(text, str(tmp_path / "out.mp3"), tts_config)
        return captured["json"]

    def test_tag_like_instructions_wrap_text(self, tmp_path, monkeypatch):
        payload = self._run(tmp_path, monkeypatch, {"instructions": "whisper"})
        assert payload["text"] == "<whisper>hello there friend</whisper>"

    def test_direction_reaches_auxiliary_system_prompt(self):
        from tools.tts_tool_providers import _apply_xai_auto_speech_tags

        rewriter_output = (
            "<soft>Welcome to the demo of our new product line.</soft> "
            "[pause] It has many features."
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=rewriter_output))]
        )
        with patch("agent.auxiliary_client.call_llm", return_value=response) as mock_call:
            result = _apply_xai_auto_speech_tags(
                "Welcome to the demo of our new product line. It has many features.",
                direction="gravelly noir narrator",
            )

        assert result == rewriter_output
        system_prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        assert "gravelly noir narrator" in system_prompt


class TestGeminiRendering:
    def test_compose_persona_placeholder_keeps_instructions(self):
        prompt = _compose_gemini_tts_prompt(
            "hello",
            {},
            persona_prompt="DIRECTOR'S NOTES\n\n{transcript}",
            instructions="excited",
        )
        assert "#### STYLE DIRECTION\nexcited" in prompt
        assert "{transcript}" not in prompt
        assert "hello" in prompt

    def test_generator_folds_channel_into_prompt(self, tmp_path, monkeypatch):
        import base64

        captured = {}
        body = json.dumps({
            "candidates": [{
                "content": {
                    "parts": [{
                        "inlineData": {
                            "data": base64.b64encode(b"\x00\x01").decode("ascii"),
                        },
                    }],
                },
            }],
        }).encode("utf-8")

        class FakeResponse:
            status_code = 200
            content = body

            def raise_for_status(self):
                pass

        def fake_post(url, params=None, headers=None, json=None, timeout=60, stream=False):
            captured["json"] = json
            return FakeResponse()

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("requests.post", fake_post)

        from tools.tts_tool_providers import _generate_gemini_tts
        _generate_gemini_tts(
            "hello", str(tmp_path / "out.wav"), {"instructions": "excited"}
        )

        prompt = captured["json"]["contents"][0]["parts"][0]["text"]
        assert "#### STYLE DIRECTION\nexcited" in prompt
        assert "#### TRANSCRIPT\nhello" in prompt


class TestElevenlabsRendering:
    def _run(self, tmp_path, monkeypatch, tts_config):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = [b"\x00"]
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_elevenlabs", return_value=mock_cls):
            _generate_elevenlabs("hello", str(tmp_path / "out.mp3"), tts_config)
        return mock_client.text_to_speech.convert

    def test_v3_model_gets_prefixed_chunk(self, tmp_path, monkeypatch):
        convert = self._run(
            tmp_path,
            monkeypatch,
            {"elevenlabs": {"model_id": "eleven_v3"}, "instructions": "[excited]"},
        )
        assert convert.call_args[1]["text"] == "[excited] hello"

    def test_default_v2_model_ignores_instructions(self, tmp_path, monkeypatch):
        convert = self._run(tmp_path, monkeypatch, {"instructions": "excited"})
        assert convert.call_args[1]["text"] == "hello"


class TestMinimaxRendering:
    def _run(self, tmp_path, monkeypatch, tts_config):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "data": {"audio": b"\x00".hex(), "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        with patch("requests.post", return_value=mock_response) as mock_post:
            _generate_minimax_tts("hello", str(tmp_path / "out.mp3"), tts_config)
        return mock_post.call_args[1]["json"]

    def test_enum_instructions_map_to_emotion(self, tmp_path, monkeypatch):
        payload = self._run(tmp_path, monkeypatch, {"instructions": "Happy"})
        assert payload["voice_setting"]["emotion"] == "happy"

    def test_non_enum_instructions_keep_configured_emotion(self, tmp_path, monkeypatch):
        payload = self._run(
            tmp_path,
            monkeypatch,
            {"minimax": {"emotion": "sad"}, "instructions": "gravelly"},
        )
        assert payload["voice_setting"]["emotion"] == "sad"


class TestCommandPlaceholder:
    def _run(self, tmp_path, config, tts_config):
        out = tmp_path / "out.mp3"
        captured = {}

        def fake_run(command, timeout, env_passthrough=None):
            captured["command"] = command
            out.write_bytes(b"\x00")

        with patch("tools.tts_command_provider.run_command_provider", side_effect=fake_run):
            _generate_command_tts("hello", str(out), "mycli", config, tts_config)
        return captured["command"]

    def test_placeholder_substituted_and_quoted(self, tmp_path):
        command = self._run(
            tmp_path,
            {"type": "command", "command": "say --style {instructions} -o {output_path}"},
            {"instructions": "calm and slow"},
        )
        assert "'calm and slow'" in command


class TestPluginDispatchInstructions:
    def _fake_provider(self):
        from agent.tts_provider import TTSProvider

        class _FakeProvider(TTSProvider):
            def __init__(self):
                self.last_call = None

            @property
            def name(self):
                return "myplugin"

            def synthesize(self, text, output_path, **kw):
                self.last_call = {"text": text, "kwargs": dict(kw)}
                return output_path

        return _FakeProvider()

    def _dispatch(self, tmp_path, tts_config):
        from agent import tts_registry
        from tools.tts_tool_plugins import _dispatch_to_plugin_provider

        tts_registry._reset_for_tests()
        provider = self._fake_provider()
        tts_registry.register_provider(provider)
        try:
            with patch("hermes_cli.plugins._ensure_plugins_discovered"):
                _dispatch_to_plugin_provider(
                    "hello", str(tmp_path / "out.mp3"), "myplugin", tts_config
                )
        finally:
            tts_registry._reset_for_tests()
        return provider.last_call

    def test_instructions_forwarded_when_set(self, tmp_path):
        call = self._dispatch(tmp_path, {"instructions": "excited"})
        assert call["kwargs"]["instructions"] == "excited"


class TestInstructionsOverhead:
    def test_every_chunk_carries_the_wrap(self, tmp_path, monkeypatch):
        """Long xAI input is split with wrap headroom; each chunk is wrapped."""
        payloads = []

        def fake_post(url, headers, json, timeout, stream=False):
            payloads.append(json)
            return _FakeHttpResponse()

        monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
        monkeypatch.setattr("requests.post", fake_post)
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config",
            lambda: {
                "provider": "xai",
                "instructions": "whisper",
                "xai": {"max_text_length": 60},
            },
        )

        from tools.tts_tool import text_to_speech_tool

        text = "A clear sentence here. " * 8
        result = json.loads(
            text_to_speech_tool(text.strip(), str(tmp_path / "out.mp3"))
        )
        assert result["success"] is True
        assert result["chunk_count"] > 1
        assert len(payloads) == result["chunk_count"]
        for payload in payloads:
            assert payload["text"].startswith("<whisper>")
            assert payload["text"].endswith("</whisper>")
            # Wrapped chunk still fits the provider cap.
            assert len(payload["text"]) <= 60

    def test_gemini_chunks_fit_the_composed_prompt_limit(
        self, tmp_path, monkeypatch
    ):
        prompts = []
        max_prompt_length = 280

        def fake_single(text, output_path, **kwargs):
            prompt = _compose_gemini_tts_prompt(
                text,
                kwargs["tts_config"]["gemini"],
                persona_prompt="",
                instructions=kwargs["instructions"],
            )
            prompts.append(prompt)
            if len(prompt) > max_prompt_length:
                return json.dumps({"success": False, "error": "prompt too long"})
            with open(output_path, "wb") as handle:
                handle.write(b"audio")
            return json.dumps({
                "success": True,
                "file_path": output_path,
                "provider": "gemini",
                "voice_compatible": False,
                "instructions_applied": True,
            })

        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config",
            lambda: {
                "provider": "gemini",
                "instructions": "excited",
                "gemini": {"max_text_length": max_prompt_length},
            },
        )
        monkeypatch.setattr("tools.tts_tool._text_to_speech_single", fake_single)
        monkeypatch.setattr(
            "tools.tts_tool._build_audio_delivery_files",
            lambda paths, *_args, **_kwargs: (paths, False),
        )

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(
            text_to_speech_tool(
                "A sentence with enough words to require several chunks. " * 4,
                str(tmp_path / "out.mp3"),
            )
        )

        assert result["success"] is True
        assert len(prompts) > 1
        assert all(len(prompt) <= max_prompt_length for prompt in prompts)


class TestInstructionsApplied:
    def test_xai_auxiliary_direction_is_not_reported_as_applied(self):
        assert not _tts_instructions_applied(
            "xai",
            "gravelly narrator",
            {"xai": {"auto_speech_tags": True}},
        )

    def test_plugin_forwarding_is_not_reported_as_applied(self, monkeypatch):
        monkeypatch.setattr("agent.tts_registry.get_provider", lambda _name: object())
        assert not _tts_instructions_applied("myplugin", "excited", {})


class TestToolLevelChannel:
    def _invoke(self, tmp_path, monkeypatch, tts_config, **kwargs):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()

        def fake_stream(path):
            with open(path, "wb") as f:
                f.write(b"ID3\x03\x00\x00\x00\x00\x00\x00")

        response = MagicMock()
        response.stream_to_file.side_effect = fake_stream
        mock_client.audio.speech.create.return_value = response
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool_openai._resolve_openai_audio_client_config",
                   return_value=("test-key", None, False)), \
             patch("tools.tts_tool._load_tts_config", return_value=tts_config):
            from tools.tts_tool import text_to_speech_tool
            result = text_to_speech_tool(
                "Hello world", str(tmp_path / "out.mp3"), **kwargs
            )
        return mock_client.audio.speech.create, json.loads(result)

    def test_provider_config_default_beats_global(self, tmp_path, monkeypatch):
        create, _ = self._invoke(
            tmp_path,
            monkeypatch,
            {
                "provider": "openai",
                "instructions": "excited",
                "openai": {"instructions": "calm"},
            },
        )
        assert create.call_args[1]["instructions"] == "calm"

    def test_empty_param_suppresses_config_default(self, tmp_path, monkeypatch):
        create, result = self._invoke(
            tmp_path,
            monkeypatch,
            {"provider": "openai", "instructions": "excited"},
            instructions="",
        )
        assert result["success"] is True
        assert "instructions" not in create.call_args[1]

    def test_applied_stamp_present_for_openai(self, tmp_path, monkeypatch):
        create, result = self._invoke(
            tmp_path, monkeypatch, {"provider": "openai"}, instructions="[whisper] <soft>"
        )
        assert {
            "instructions": create.call_args[1]["instructions"],
            "instructions_applied": result.get("instructions_applied"),
        } == {"instructions": "[whisper] <soft>", "instructions_applied": True}

    def test_no_stamp_for_edge(self, tmp_path, monkeypatch):
        def fake_edge(text, out, cfg):
            with open(out, "wb") as f:
                f.write(b"ID3\x03\x00\x00\x00\x00\x00\x00")
            return out

        async def fake_edge_async(text, out, cfg):
            return fake_edge(text, out, cfg)

        # The dispatcher import-checks edge-tts before calling the generator;
        # stub the import so the test also passes where the package is not
        # installed (matches the TestEdgeTtsSpeed pattern).
        monkeypatch.setattr(
            "tools.tts_tool._import_edge_tts", lambda: MagicMock()
        )
        monkeypatch.setattr("tools.tts_tool._generate_edge_tts", fake_edge_async)
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config", lambda: {"provider": "edge"}
        )

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(
            text_to_speech_tool(
                "Hello world", str(tmp_path / "out.mp3"), instructions="excited"
            )
        )
        assert result["success"] is True
        assert "instructions_applied" not in result
