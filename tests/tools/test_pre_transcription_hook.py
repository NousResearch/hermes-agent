"""Tests for the ``pre_transcription`` plugin hook and STT prompt threading
(issue #64168) wired into ``tools.transcription_tools.transcribe_audio``.

Covers:

1. Fixture plugin returning a prompt → the backend call receives
   ``initial_prompt`` (faster-whisper) / ``prompt`` (OpenAI, Groq, Mistral).
   The API boundary is stubbed — no live model is loaded or called.
2. Two hooks → last-writer-wins per field, in registration order.
3. Hook returning the read-only ``file_path`` field → dropped with a log.
4. No hook registered → invoke_hook is never called and the backend
   dispatch kwargs are identical to a control run (no prompt/language keys
   on the wire).
5. ``stt.prompt`` config alone → threaded without any hook.
6. Config + hook → hook wins (config is the base, hooks mutate on top).
7. Unsupported backend (xAI, ElevenLabs) → DEBUG note and the call
   proceeds without the prompt.
8. Plugin-registered providers receive the prompt via the ABC's ``**extra``
   kwargs — no signature change.

Mirrors the ``transform_tool_result`` hook test conventions from
``tests/test_transform_tool_result_hook.py``.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.plugins as plugins_mod
from tools import transcription_tools


PROMPT = "Hermes, Teknium, Nous Research, kanban"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(tmp_path):
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"fake audio data")
    return str(audio)


def _fake_hooks(monkeypatch, results):
    """Install fake has_hook/invoke_hook returning *results* and capture kwargs."""
    captured = {}

    def _invoke(hook_name, **kw):
        captured["hook_name"] = hook_name
        captured["kwargs"] = kw
        return list(results)

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke)
    return captured


def _no_hooks(monkeypatch):
    """No hook registered: has_hook is False and invoke_hook must not fire."""
    def _boom(hook_name, **kw):  # pragma: no cover - the assert is the point
        raise AssertionError(
            "invoke_hook must not be called when has_hook() is False"
        )

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _boom)


def _dispatch_ctx(stt_config, provider):
    """Patch config load + provider resolution around transcribe_audio."""
    return (
        patch("tools.transcription_tools._load_stt_config", return_value=stt_config),
        patch("tools.transcription_tools._get_provider", return_value=provider),
    )


# ---------------------------------------------------------------------------
# Hook registration surface
# ---------------------------------------------------------------------------


def test_pre_transcription_in_valid_hooks():
    assert "pre_transcription" in plugins_mod.VALID_HOOKS


# ---------------------------------------------------------------------------
# Prompt threading into backends (API boundary stubbed)
# ---------------------------------------------------------------------------


class TestPromptThreading:
    def test_hook_prompt_reaches_faster_whisper_initial_prompt(
        self, monkeypatch, tmp_path,
    ):
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [{"prompt": PROMPT}])

        mock_segment = MagicMock()
        mock_segment.text = "hello"
        mock_info = MagicMock(language="en", duration=1.0)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        cfg_patch, prov_patch = _dispatch_ctx({"provider": "local"}, "local")
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._load_local_whisper_model",
                   return_value=mock_model), \
             patch("tools.transcription_tools._local_model", None):
            result = transcription_tools.transcribe_audio(audio)

        assert result["success"] is True
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["initial_prompt"] == PROMPT

    def test_hook_prompt_and_language_reach_openai(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        _fake_hooks(monkeypatch, [{"prompt": PROMPT, "language": "en"}])

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello"

        cfg_patch, prov_patch = _dispatch_ctx({"provider": "openai"}, "openai")
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            result = transcription_tools.transcribe_audio(audio)

        assert result["success"] is True
        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["prompt"] == PROMPT
        assert kwargs["language"] == "en"

    def test_hook_prompt_reaches_groq(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        _fake_hooks(monkeypatch, [{"prompt": PROMPT}])

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello"

        cfg_patch, prov_patch = _dispatch_ctx({"provider": "groq"}, "groq")
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            result = transcription_tools.transcribe_audio(audio)

        assert result["success"] is True
        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["prompt"] == PROMPT

    def test_prompt_reaches_mistral(self, monkeypatch, tmp_path):
        """Unit-level: _transcribe_mistral forwards prompt to the SDK call."""
        audio = _make_audio(tmp_path)
        monkeypatch.setenv("MISTRAL_API_KEY", "mk-test")
        # Never attempt a lazy install in tests.
        monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **kw: None)

        mistral_cls = MagicMock()
        mock_client = mistral_cls.return_value.__enter__.return_value
        mock_client.audio.transcriptions.complete.return_value = SimpleNamespace(
            text="hello",
        )
        fake_mistralai = SimpleNamespace(client=SimpleNamespace(Mistral=mistral_cls))
        monkeypatch.setitem(sys.modules, "mistralai", fake_mistralai)
        monkeypatch.setitem(sys.modules, "mistralai.client", fake_mistralai.client)

        result = transcription_tools._transcribe_mistral(
            audio, "voxtral-mini-latest", prompt=PROMPT,
        )

        assert result["success"] is True
        _, kwargs = mock_client.audio.transcriptions.complete.call_args
        assert kwargs["prompt"] == PROMPT


# ---------------------------------------------------------------------------
# Hook merge mechanics
# ---------------------------------------------------------------------------


class TestHookMergeMechanics:
    def test_two_hooks_last_writer_wins_per_field(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        # Two hooks in registration order: the second overwrites ``prompt``
        # but leaves ``language`` untouched — last-writer-wins PER FIELD.
        _fake_hooks(
            monkeypatch,
            [{"prompt": "first", "language": "ja"}, {"prompt": "second"}],
        )

        backend = MagicMock(return_value={"success": True, "transcript": "hi"})
        cfg_patch, prov_patch = _dispatch_ctx({"provider": "openai"}, "openai")
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._transcribe_openai", backend):
            transcription_tools.transcribe_audio(audio)

        _, kwargs = backend.call_args
        assert kwargs["prompt"] == "second"
        assert kwargs["language"] == "ja"

    def test_hook_model_override_flows_to_backend(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [{"model": "gpt-4o-transcribe"}])

        backend = MagicMock(return_value={"success": True, "transcript": "hi"})
        cfg_patch, prov_patch = _dispatch_ctx({"provider": "openai"}, "openai")
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._transcribe_openai", backend):
            transcription_tools.transcribe_audio(audio)

        args, _ = backend.call_args
        assert args[1] == "gpt-4o-transcribe"

    def test_file_path_mutation_dropped_with_log(
        self, monkeypatch, tmp_path, caplog,
    ):
        audio = _make_audio(tmp_path)
        _fake_hooks(
            monkeypatch,
            [{"file_path": "/evil/other.ogg", "prompt": PROMPT}],
        )

        backend = MagicMock(return_value={"success": True, "transcript": "hi"})
        cfg_patch, prov_patch = _dispatch_ctx({"provider": "openai"}, "openai")
        with caplog.at_level(logging.WARNING, logger="tools.transcription_tools"), \
             cfg_patch, prov_patch, \
             patch("tools.transcription_tools._transcribe_openai", backend):
            transcription_tools.transcribe_audio(audio)

        args, kwargs = backend.call_args
        # Original file_path untouched, valid fields still applied.
        assert args[0] == audio
        assert kwargs["prompt"] == PROMPT
        assert "read-only" in caplog.text

    def test_non_string_field_values_ignored(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [{"prompt": 123, "language": ["en"]}])

        backend = MagicMock(return_value={"success": True, "transcript": "hi"})
        cfg_patch, prov_patch = _dispatch_ctx({"provider": "openai"}, "openai")
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._transcribe_openai", backend):
            transcription_tools.transcribe_audio(audio)

        _, kwargs = backend.call_args
        assert kwargs["prompt"] is None
        assert kwargs["language"] is None

    def test_hook_receives_expected_kwargs(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        captured = _fake_hooks(monkeypatch, [])

        backend = MagicMock(return_value={"success": True, "transcript": "hi"})
        cfg_patch, prov_patch = _dispatch_ctx(
            {"provider": "openai", "prompt": "config base"}, "openai",
        )
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._transcribe_openai", backend):
            transcription_tools.transcribe_audio(
                audio, model="whisper-1", source="gateway",
            )

        assert captured["hook_name"] == "pre_transcription"
        kw = captured["kwargs"]
        assert kw["file_path"] == audio
        assert kw["provider"] == "openai"
        assert kw["model"] == "whisper-1"
        # Config is the base — the hook sees the static stt.prompt value.
        assert kw["prompt"] == "config base"
        assert kw["source"] == "gateway"


# ---------------------------------------------------------------------------
# No-hook path stays identical
# ---------------------------------------------------------------------------


class TestNoHookPath:
    def test_no_hook_dispatch_kwargs_identical_to_control(
        self, monkeypatch, tmp_path,
    ):
        audio = _make_audio(tmp_path)
        _no_hooks(monkeypatch)  # invoke_hook raises if ever called

        backend = MagicMock(return_value={"success": True, "transcript": "hi"})
        cfg_patch, prov_patch = _dispatch_ctx({"provider": "openai"}, "openai")
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._transcribe_openai", backend):
            transcription_tools.transcribe_audio(audio)

        args, kwargs = backend.call_args
        assert args == (audio, "whisper-1")
        # No prompt/language reach the backend — same effective dispatch as
        # a control run without the hook plumbing.
        assert kwargs == {"language": None, "prompt": None}

    def test_no_hook_openai_wire_call_has_no_prompt_or_language(
        self, monkeypatch, tmp_path,
    ):
        """Wire-level control: with prompt/language unset, the OpenAI SDK
        call carries exactly the same kwargs as before this feature."""
        audio = _make_audio(tmp_path)
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            transcription_tools._transcribe_openai(audio, "whisper-1")

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert set(kwargs) == {"model", "file", "response_format"}


# ---------------------------------------------------------------------------
# stt.prompt config key
# ---------------------------------------------------------------------------


class TestSttPromptConfig:
    def test_config_prompt_alone_is_threaded(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        _no_hooks(monkeypatch)

        backend = MagicMock(return_value={"success": True, "transcript": "hi"})
        cfg_patch, prov_patch = _dispatch_ctx(
            {"provider": "openai", "prompt": PROMPT}, "openai",
        )
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._transcribe_openai", backend):
            transcription_tools.transcribe_audio(audio)

        _, kwargs = backend.call_args
        assert kwargs["prompt"] == PROMPT

    def test_hook_prompt_wins_over_config_prompt(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [{"prompt": "hook wins"}])

        backend = MagicMock(return_value={"success": True, "transcript": "hi"})
        cfg_patch, prov_patch = _dispatch_ctx(
            {"provider": "openai", "prompt": "config base"}, "openai",
        )
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._transcribe_openai", backend):
            transcription_tools.transcribe_audio(audio)

        _, kwargs = backend.call_args
        assert kwargs["prompt"] == "hook wins"

    def test_final_prompt_is_not_silently_truncated(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        long_prompt = "domain-vocabulary " * 300
        _fake_hooks(monkeypatch, [{"prompt": long_prompt}])

        backend = MagicMock(return_value={"success": True, "transcript": "hi"})
        cfg_patch, prov_patch = _dispatch_ctx(
            {"provider": "openai", "prompt": "config base"}, "openai",
        )
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._transcribe_openai", backend):
            transcription_tools.transcribe_audio(audio)

        _, kwargs = backend.call_args
        assert kwargs["prompt"] == long_prompt

    def test_blank_config_prompt_ignored(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        _no_hooks(monkeypatch)

        backend = MagicMock(return_value={"success": True, "transcript": "hi"})
        cfg_patch, prov_patch = _dispatch_ctx(
            {"provider": "openai", "prompt": "   "}, "openai",
        )
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._transcribe_openai", backend):
            transcription_tools.transcribe_audio(audio)

        _, kwargs = backend.call_args
        assert kwargs["prompt"] is None


# ---------------------------------------------------------------------------
# Backends without prompt support
# ---------------------------------------------------------------------------


class TestUnsupportedBackends:
    def test_xai_logs_debug_and_proceeds_without_prompt(
        self, monkeypatch, tmp_path, caplog,
    ):
        audio = _make_audio(tmp_path)
        monkeypatch.setattr(
            "tools.xai_http.resolve_xai_http_credentials",
            lambda: {"api_key": "xk-test", "base_url": None},
        )
        monkeypatch.setattr(
            "tools.xai_http.hermes_xai_user_agent", lambda: "test-ua",
        )

        response = MagicMock(status_code=200)
        response.json.return_value = {"text": "hello", "language": "en", "duration": 1.0}
        fake_requests = SimpleNamespace(post=MagicMock(return_value=response))
        monkeypatch.setitem(sys.modules, "requests", fake_requests)

        with caplog.at_level(logging.DEBUG, logger="tools.transcription_tools"), \
             patch("tools.transcription_tools._load_stt_config", return_value={}):
            result = transcription_tools._transcribe_xai(
                audio, "grok-stt", prompt=PROMPT,
            )

        assert result["success"] is True
        assert "does not support transcription prompts" in caplog.text
        _, kwargs = fake_requests.post.call_args
        assert "prompt" not in kwargs["data"]

    def test_elevenlabs_logs_debug_and_proceeds_without_prompt(
        self, monkeypatch, tmp_path, caplog,
    ):
        audio = _make_audio(tmp_path)
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")

        response = MagicMock(status_code=200)
        response.json.return_value = {"text": "hello"}
        fake_requests = SimpleNamespace(post=MagicMock(return_value=response))
        monkeypatch.setitem(sys.modules, "requests", fake_requests)

        with caplog.at_level(logging.DEBUG, logger="tools.transcription_tools"), \
             patch("tools.transcription_tools._load_stt_config", return_value={}):
            result = transcription_tools._transcribe_elevenlabs(
                audio, "scribe_v2", prompt=PROMPT,
            )

        assert result["success"] is True
        assert "does not support transcription prompts" in caplog.text
        _, kwargs = fake_requests.post.call_args
        assert "prompt" not in kwargs["data"]


# ---------------------------------------------------------------------------
# Plugin-registered providers (TranscriptionProvider ABC)
# ---------------------------------------------------------------------------


class TestPluginProviderThreading:
    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        from agent import transcription_registry
        transcription_registry._reset_for_tests()
        yield
        transcription_registry._reset_for_tests()

    def _register_fake_provider(self):
        from agent import transcription_registry
        from agent.transcription_provider import TranscriptionProvider

        class _FakeProvider(TranscriptionProvider):
            def __init__(self):
                self.last_call = None

            @property
            def name(self):
                return "openrouter"

            def transcribe(self, file_path, **kw):
                self.last_call = {"file_path": file_path, "kwargs": dict(kw)}
                return {"success": True, "transcript": "hi", "provider": "openrouter"}

        provider = _FakeProvider()
        transcription_registry.register_provider(provider)
        return provider

    def test_plugin_provider_receives_prompt_via_extra_kwargs(
        self, monkeypatch, tmp_path,
    ):
        audio = _make_audio(tmp_path)
        provider = self._register_fake_provider()
        _fake_hooks(monkeypatch, [{"prompt": PROMPT, "language": "en"}])

        cfg_patch, prov_patch = _dispatch_ctx(
            {"provider": "openrouter"}, "openrouter",
        )
        with cfg_patch, prov_patch:
            result = transcription_tools.transcribe_audio(audio)

        assert result["success"] is True
        assert provider.last_call["kwargs"]["prompt"] == PROMPT
        assert provider.last_call["kwargs"]["language"] == "en"

    def test_plugin_provider_sees_no_prompt_key_when_unset(
        self, monkeypatch, tmp_path,
    ):
        audio = _make_audio(tmp_path)
        provider = self._register_fake_provider()
        _no_hooks(monkeypatch)

        cfg_patch, prov_patch = _dispatch_ctx(
            {"provider": "openrouter"}, "openrouter",
        )
        with cfg_patch, prov_patch:
            transcription_tools.transcribe_audio(audio)

        # Byte-identical no-prompt path: the key is not even sent.
        assert "prompt" not in provider.last_call["kwargs"]


# ---------------------------------------------------------------------------
# End-to-end with a real fixture plugin (real PluginManager mechanics)
# ---------------------------------------------------------------------------


def test_real_fixture_plugins_thread_prompt_in_registration_order(
    monkeypatch, tmp_path,
):
    """Two callbacks registered by a real plugin, applied in registration
    order with last-writer-wins — verified against the faster-whisper
    backend stub receiving ``initial_prompt``."""
    import os
    from pathlib import Path

    import yaml

    hermes_home = Path(os.environ["HERMES_HOME"])
    plugin_dir = hermes_home / "plugins" / "stt_vocab"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: stt_vocab\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        '    ctx.register_hook("pre_transcription", '
        'lambda **kw: {"prompt": "loser", "language": "en"})\n'
        '    ctx.register_hook("pre_transcription", '
        f'lambda **kw: {{"prompt": "{PROMPT}"}})\n',
        encoding="utf-8",
    )
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"plugins": {"enabled": ["stt_vocab"]}}),
        encoding="utf-8",
    )

    old_manager = plugins_mod._plugin_manager
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    try:
        plugins_mod.discover_plugins()

        audio = _make_audio(tmp_path)
        mock_segment = MagicMock()
        mock_segment.text = "hello"
        mock_info = MagicMock(language="en", duration=1.0)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("tools.transcription_tools._load_stt_config",
                   return_value={"provider": "local"}), \
             patch("tools.transcription_tools._get_provider",
                   return_value="local"), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._load_local_whisper_model",
                   return_value=mock_model), \
             patch("tools.transcription_tools._local_model", None):
            result = transcription_tools.transcribe_audio(audio)
    finally:
        plugins_mod._plugin_manager = old_manager

    assert result["success"] is True
    _, kwargs = mock_model.transcribe.call_args
    assert kwargs["initial_prompt"] == PROMPT  # last writer won
    assert kwargs["language"] == "en"  # earlier hook's field preserved
