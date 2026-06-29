"""Tests for TTS plugin dispatch in tools/tts_tool.py (issue #30398).

Covers the three core invariants of the plugin dispatcher:

1. Built-in provider names short-circuit — plugins NEVER win over a
   built-in. Even if a plugin somehow ended up in the registry with a
   built-in name (which the registry already blocks), the dispatcher
   re-checks defensively.
2. Command-type providers declared under ``tts.providers.<name>: type:
   command`` (PR #17843) win over a plugin with the same name. Config
   is more local than plugin install.
3. Plugin dispatch fires only when the configured provider is neither
   a built-in nor a command-type entry, AND a plugin is registered
   under that name. Unknown names fall through.

Also exercises:
- Plugin exceptions surface to the outer error envelope (don't crash)
- Plugin returning a different path is honored
- voice_compatible: True triggers ffmpeg opus conversion path
- voice_compatible: False keeps the file as-is

The dispatcher is exercised in isolation — we don't actually call
``text_to_speech_tool`` because that would require real audio file
writes. Each test directly calls
``tools.tts_tool._dispatch_to_plugin_provider`` / the predicate
helpers.
"""

from __future__ import annotations

from typing import Optional

import pytest

from agent import tts_registry
from agent.tts_provider import TTSProvider
from tools import tts_tool


class _FakeTTSProvider(TTSProvider):
    def __init__(
        self,
        name: str,
        voice_compat: bool = False,
        raise_exc: Optional[BaseException] = None,
        return_path: Optional[str] = None,
    ):
        self._name = name
        self._voice_compat = voice_compat
        self._raise_exc = raise_exc
        self._return_path = return_path
        # Recorded for assertions
        self.last_call: Optional[dict] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def voice_compatible(self) -> bool:
        return self._voice_compat

    def synthesize(self, text, output_path, **kw):
        self.last_call = {
            "text": text,
            "output_path": output_path,
            "kwargs": dict(kw),
        }
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._return_path if self._return_path is not None else output_path


@pytest.fixture(autouse=True)
def _reset_registry():
    tts_registry._reset_for_tests()
    yield
    tts_registry._reset_for_tests()


# ---------------------------------------------------------------------------
# Resolution invariants
# ---------------------------------------------------------------------------


class TestBuiltinAlwaysWins:
    """Built-in TTS provider names short-circuit the dispatcher.

    Even with a plugin registered (which the registry would reject —
    but the dispatcher is defensive), built-in names return None so
    the caller's elif chain handles them natively.
    """

    @pytest.mark.parametrize(
        "builtin",
        ["edge", "openai", "elevenlabs", "minimax", "gemini",
         "mistral", "xai", "piper", "kittentts", "neutts"],
    )
    def test_dispatcher_short_circuits_builtin(self, builtin):
        result = tts_tool._dispatch_to_plugin_provider(
            text="hello",
            output_path="/tmp/out.mp3",
            provider=builtin,
            tts_config={},
        )
        assert result is None, (
            f"Built-in {builtin!r} must short-circuit plugin dispatch. "
            "If this test fails, the dispatcher would silently let a "
            "plugin with a built-in name shadow the native handler — "
            "violating the precedence rule from PR #17843."
        )

    def test_dispatcher_short_circuits_builtin_case_insensitive(self):
        for variant in ("EDGE", "Edge", "  edge  ", "eDgE"):
            assert (
                tts_tool._dispatch_to_plugin_provider(
                    text="hello", output_path="/tmp/x.mp3",
                    provider=variant, tts_config={},
                ) is None
            )


class TestCommandProviderWins:
    """A same-name ``tts.providers.<name>: type: command`` config beats a plugin.

    Locality: a user's command-provider config is more specific than
    whichever plugin happens to be installed.
    """

    def test_command_config_beats_plugin(self):
        tts_registry.register_provider(_FakeTTSProvider(name="my-tts"))

        result = tts_tool._dispatch_to_plugin_provider(
            text="hello",
            output_path="/tmp/out.mp3",
            provider="my-tts",
            tts_config={
                "providers": {
                    "my-tts": {
                        "type": "command",
                        "command": "echo 'hi' > {output_path}",
                    },
                },
            },
        )
        # Plugin path returns None → caller falls back to command
        # provider dispatch (handled by the outer text_to_speech_tool
        # via _resolve_command_provider_config).
        assert result is None


class TestPluginDispatch:
    """Happy path: configured name matches a registered plugin, dispatcher fires."""

    def test_registered_plugin_called(self):
        provider = _FakeTTSProvider(name="cartesia")
        tts_registry.register_provider(provider)

        result = tts_tool._dispatch_to_plugin_provider(
            text="hello world",
            output_path="/tmp/out.mp3",
            provider="cartesia",
            tts_config={},
        )
        assert result == "/tmp/out.mp3"
        assert provider.last_call is not None
        assert provider.last_call["text"] == "hello world"
        assert provider.last_call["output_path"] == "/tmp/out.mp3"

    def test_unregistered_name_returns_none(self):
        result = tts_tool._dispatch_to_plugin_provider(
            text="hello",
            output_path="/tmp/out.mp3",
            provider="unknown-tts",
            tts_config={},
        )
        assert result is None

    def test_voice_model_speed_format_forwarded(self):
        provider = _FakeTTSProvider(name="cartesia")
        tts_registry.register_provider(provider)

        result = tts_tool._dispatch_to_plugin_provider(
            text="hello",
            output_path="/tmp/out.opus",
            provider="cartesia",
            tts_config={
                "voice": "voice-aria",
                "model": "sonic-2",
                "speed": 1.2,
                "output_format": "opus",
            },
        )
        assert result == "/tmp/out.opus"
        kwargs = provider.last_call["kwargs"]
        assert kwargs["voice"] == "voice-aria"
        assert kwargs["model"] == "sonic-2"
        assert kwargs["speed"] == 1.2
        assert kwargs["format"] == "opus"

    def test_empty_string_voice_passed_as_none(self):
        """Empty-string config values are normalized to None so providers can
        fall back to their own defaults (matches the ABC contract)."""
        provider = _FakeTTSProvider(name="cartesia")
        tts_registry.register_provider(provider)

        tts_tool._dispatch_to_plugin_provider(
            text="hello",
            output_path="/tmp/out.mp3",
            provider="cartesia",
            tts_config={"voice": "", "model": ""},
        )
        kwargs = provider.last_call["kwargs"]
        assert kwargs["voice"] is None
        assert kwargs["model"] is None

    def test_provider_returning_different_path_honored(self):
        """If a provider rewrites the output path (e.g. format-driven extension
        change), the dispatcher returns the new path."""
        provider = _FakeTTSProvider(name="cartesia", return_path="/tmp/rewritten.opus")
        tts_registry.register_provider(provider)

        result = tts_tool._dispatch_to_plugin_provider(
            text="hi",
            output_path="/tmp/out.mp3",
            provider="cartesia",
            tts_config={},
        )
        assert result == "/tmp/rewritten.opus"

    def test_provider_returning_none_falls_back_to_output_path(self):
        """Defensive: a provider returning None means the dispatcher should
        report the caller-supplied output_path (matches the ABC contract — the
        provider is supposed to write to output_path)."""
        provider = _FakeTTSProvider(name="cartesia", return_path=None)
        # Override the default-output-path behavior to return None explicitly
        provider._return_path = None

        class _ReturnsNone(_FakeTTSProvider):
            def synthesize(self, text, output_path, **kw):
                return None  # type: ignore[return-value]

        provider2 = _ReturnsNone(name="weird")
        tts_registry.register_provider(provider2)

        result = tts_tool._dispatch_to_plugin_provider(
            text="hi",
            output_path="/tmp/out.mp3",
            provider="weird",
            tts_config={},
        )
        assert result == "/tmp/out.mp3"

    def test_provider_exception_bubbles_up(self):
        """Plugin exceptions are NOT swallowed by the dispatcher — they bubble
        up so the outer ``text_to_speech_tool`` try/except converts them to
        the standard error envelope. Matches command-provider failure
        behavior."""
        provider = _FakeTTSProvider(
            name="cartesia",
            raise_exc=RuntimeError("network down"),
        )
        tts_registry.register_provider(provider)

        with pytest.raises(RuntimeError, match="network down"):
            tts_tool._dispatch_to_plugin_provider(
                text="hi",
                output_path="/tmp/out.mp3",
                provider="cartesia",
                tts_config={},
            )


# ---------------------------------------------------------------------------
# voice_compatible flag
# ---------------------------------------------------------------------------


class TestVoiceCompatibleHelper:
    def test_voice_compatible_true(self):
        tts_registry.register_provider(
            _FakeTTSProvider(name="cartesia", voice_compat=True)
        )
        assert tts_tool._plugin_provider_is_voice_compatible("cartesia") is True

    def test_voice_compatible_false_by_default(self):
        tts_registry.register_provider(_FakeTTSProvider(name="cartesia"))
        assert tts_tool._plugin_provider_is_voice_compatible("cartesia") is False

    def test_unregistered_provider_returns_false(self):
        assert tts_tool._plugin_provider_is_voice_compatible("unknown") is False

    def test_empty_provider_name_returns_false(self):
        assert tts_tool._plugin_provider_is_voice_compatible("") is False

    @pytest.mark.parametrize(
        "builtin",
        ["edge", "openai", "elevenlabs", "minimax", "gemini",
         "mistral", "xai", "piper", "kittentts", "neutts"],
    )
    def test_builtin_names_return_false(self, builtin):
        """voice_compatible helper short-circuits built-ins so they go
        through the legacy code path that handles their format quirks."""
        assert tts_tool._plugin_provider_is_voice_compatible(builtin) is False

    def test_voice_compatible_case_insensitive(self):
        tts_registry.register_provider(
            _FakeTTSProvider(name="cartesia", voice_compat=True)
        )
        assert tts_tool._plugin_provider_is_voice_compatible("CARTESIA") is True
        assert tts_tool._plugin_provider_is_voice_compatible("  cartesia  ") is True

    def test_provider_property_exception_returns_false(self):
        """A buggy ``voice_compatible`` property raising must not crash the
        TTS pipeline."""

        class _ExplodingProvider(_FakeTTSProvider):
            @property
            def voice_compatible(self) -> bool:
                raise RuntimeError("boom")

        tts_registry.register_provider(_ExplodingProvider(name="cartesia"))
        assert tts_tool._plugin_provider_is_voice_compatible("cartesia") is False


class TestPluginRequirements:
    def test_check_tts_requirements_accepts_available_plugin(self, monkeypatch):
        provider = _FakeTTSProvider(name="cartesia")
        provider.is_available = lambda: True  # type: ignore[method-assign]
        tts_registry.register_provider(provider)

        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "cartesia"})
        monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(tts_tool, "_has_any_command_tts_provider", lambda: False)

        assert tts_tool._has_available_plugin_tts_provider() is True
