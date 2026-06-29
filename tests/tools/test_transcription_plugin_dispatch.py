"""Tests for STT plugin dispatch in tools/transcription_tools.py.

Covers the resolution invariants of the new plugin dispatcher (follow-up
to #30398 — STT pluggability):

1. Built-in provider names short-circuit — plugins NEVER win over a
   built-in. Even if a plugin somehow ended up in the registry with a
   built-in name (which the registry blocks), the dispatcher re-checks
   defensively.
2. Unknown name with no plugin → returns None (caller surfaces the
   "provider_not_registered" error).
3. Unknown name with plugin registered → dispatches, returns result.
4. Plugin exceptions are caught and converted to the standard error
   envelope.
5. Plugin returning non-dict → caught with error envelope.
6. Plugin result has ``provider`` field stamped if missing.
"""

from __future__ import annotations

import pytest

from agent import transcription_registry
from agent.transcription_provider import TranscriptionProvider
from tools import transcription_tools


class _FakeProvider(TranscriptionProvider):
    def __init__(
        self,
        name: str,
        result: dict | None = None,
        raise_exc: BaseException | None = None,
        available: bool = True,
        available_raises: BaseException | None = None,
    ):
        self._name = name
        self._result = result
        self._raise_exc = raise_exc
        self._available = available
        self._available_raises = available_raises
        self.last_call: dict | None = None

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        if self._available_raises is not None:
            raise self._available_raises
        return self._available

    def transcribe(self, file_path: str, **kw):
        self.last_call = {"file_path": file_path, "kwargs": dict(kw)}
        if self._raise_exc is not None:
            raise self._raise_exc
        if self._result is not None:
            return self._result
        return {"success": True, "transcript": "fake transcript", "provider": self._name}


@pytest.fixture(autouse=True)
def _reset_registry():
    transcription_registry._reset_for_tests()
    yield
    transcription_registry._reset_for_tests()


# ---------------------------------------------------------------------------
# Built-in always wins
# ---------------------------------------------------------------------------


class TestBuiltinAlwaysWins:
    """Built-in STT provider names short-circuit the dispatcher.

    Even with a plugin registered (which the registry would reject —
    but the dispatcher is defensive), built-in names return None so
    the caller's elif chain handles them natively.
    """

    @pytest.mark.parametrize(
        "builtin",
        ["local", "local_command", "groq", "openai", "mistral", "xai"],
    )
    def test_dispatcher_short_circuits_builtin(self, builtin):
        result = transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", builtin,
        )
        assert result is None, (
            f"Built-in {builtin!r} must short-circuit plugin dispatch."
        )

    def test_dispatcher_short_circuits_none(self):
        """The ``none`` sentinel from _get_provider() means no provider
        available — must not reach plugin registry."""
        result = transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "none",
        )
        assert result is None

    def test_dispatcher_short_circuits_empty(self):
        assert transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "",
        ) is None

    def test_dispatcher_short_circuits_builtin_case_insensitive(self):
        for variant in ("OPENAI", "OpenAI", "  openai  ", "oPeNaI"):
            assert (
                transcription_tools._dispatch_to_plugin_provider(
                    "/tmp/audio.mp3", variant,
                ) is None
            )


# ---------------------------------------------------------------------------
# Unknown names
# ---------------------------------------------------------------------------


class TestPluginDispatch:
    def test_registered_plugin_called(self):
        provider = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(provider)

        result = transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "openrouter",
        )
        assert result is not None
        assert result["success"] is True
        assert result["transcript"] == "fake transcript"
        assert result["provider"] == "openrouter"
        assert provider.last_call is not None
        assert provider.last_call["file_path"] == "/tmp/audio.mp3"

    def test_unregistered_name_returns_none(self):
        """Unknown name + no plugin → return None so the caller surfaces
        the legacy 'No STT provider available' error."""
        result = transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "unknown-stt",
        )
        assert result is None

    def test_model_kwarg_forwarded(self):
        provider = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(provider)

        transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "openrouter", model="whisper-large-v3",
        )
        assert provider.last_call["kwargs"]["model"] == "whisper-large-v3"

    def test_language_kwarg_forwarded(self):
        provider = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(provider)

        transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "openrouter", language="en",
        )
        assert provider.last_call["kwargs"]["language"] == "en"

    def test_provider_exception_converted_to_error_envelope(self):
        provider = _FakeProvider(name="openrouter", raise_exc=RuntimeError("network down"))
        transcription_registry.register_provider(provider)

        result = transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "openrouter",
        )
        assert result is not None
        assert result["success"] is False
        assert "network down" in result["error"]
        assert result["transcript"] == ""
        assert result["provider"] == "openrouter"

    def test_provider_non_dict_result_converted_to_error(self):
        provider = _FakeProvider(name="openrouter", result="weird string")  # type: ignore[arg-type]
        transcription_registry.register_provider(provider)

        result = transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "openrouter",
        )
        assert result is not None
        assert result["success"] is False
        assert "non-dict" in result["error"]
        assert result["provider"] == "openrouter"

    def test_provider_field_stamped_if_missing(self):
        """If a plugin forgets to set ``provider`` in its result, the
        dispatcher stamps it from the registered name."""
        provider = _FakeProvider(
            name="openrouter",
            result={"success": True, "transcript": "hi"},  # no provider key
        )
        transcription_registry.register_provider(provider)

        result = transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "openrouter",
        )
        assert result is not None
        assert result["provider"] == "openrouter"


# ---------------------------------------------------------------------------
# End-to-end via transcribe_audio
# ---------------------------------------------------------------------------


class TestTranscribeAudioE2E:
    """transcribe_audio() routes plugin dispatch correctly when the
    configured name is unknown to the built-in branches.

    Note: we mock _validate_audio_file and _get_provider so the real
    file-validation and provider-resolution don't fire — we're testing
    the plugin-dispatch wiring, not those helpers.
    """

    def test_unknown_name_with_plugin_dispatches(self):
        from unittest.mock import patch
        provider = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(provider)

        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openrouter"}), \
             patch("tools.transcription_tools.is_stt_enabled", return_value=True), \
             patch("tools.transcription_tools._get_provider", return_value="openrouter"):
            result = transcription_tools.transcribe_audio("/tmp/audio.mp3")

        assert result["success"] is True
        assert result["transcript"] == "fake transcript"
        assert result["provider"] == "openrouter"

    def test_unknown_name_without_plugin_returns_provider_specific_error(self):
        """Explicit unknown providers should get a named registration error."""
        from unittest.mock import patch

        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openrouter"}), \
             patch("tools.transcription_tools.is_stt_enabled", return_value=True), \
             patch("tools.transcription_tools._get_provider", return_value="openrouter"):
            result = transcription_tools.transcribe_audio("/tmp/audio.mp3")

        assert result["success"] is False
        assert result["provider"] == "openrouter"
        assert result["error_type"] == "provider_not_registered"
        assert "stt.provider='openrouter'" in result["error"]
        assert "hermes plugins list" in result["error"]
        assert "No STT provider available" not in result["error"]

    def test_auto_detect_failure_keeps_legacy_no_provider_message(self):
        """No explicit stt.provider remains the generic setup guidance path."""
        from unittest.mock import patch

        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools.is_stt_enabled", return_value=True), \
             patch("tools.transcription_tools._get_provider", return_value="none"):
            result = transcription_tools.transcribe_audio("/tmp/audio.mp3")

        assert result["success"] is False
        assert result.get("error_type") is None
        assert "No STT provider available" in result["error"]

    def test_builtin_name_does_not_consult_plugin_registry(self):
        """Even if a plugin's name collides with a built-in (which the
        registry blocks, but defense in depth matters), transcribe_audio
        with provider='groq' goes through the legacy elif chain, never
        the plugin dispatcher."""
        from unittest.mock import patch
        # Register a plugin that WOULD respond to 'openrouter' — but
        # we're asking for 'groq', so it shouldn't be called.
        provider = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(provider)

        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value={"provider": "groq"}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "from groq", "provider": "groq"}) as mock_groq:
            result = transcription_tools.transcribe_audio("/tmp/audio.mp3")

        assert result["provider"] == "groq"
        assert result["transcript"] == "from groq"
        mock_groq.assert_called_once()
        # Plugin was never called
        assert provider.last_call is None


# ---------------------------------------------------------------------------
# Availability gating (codex review feedback on PR #30493)
# ---------------------------------------------------------------------------


class TestAvailabilityGate:
    """When the configured plugin reports ``is_available() == False``,
    the dispatcher MUST short-circuit with a clear unavailability
    envelope instead of routing the call into a plugin that'll crash.

    The user explicitly set ``stt.provider: <plugin>`` so falling
    through to the generic "No STT provider available" message would
    be misleading — surface the plugin's own unavailability instead.
    """

    def test_unavailable_plugin_returns_envelope_not_none(self):
        provider = _FakeProvider(name="openrouter", available=False)
        transcription_registry.register_provider(provider)

        result = transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "openrouter",
        )
        assert result is not None, (
            "Unavailable plugin must return an envelope, not None — "
            "otherwise we fall through to the generic auto-detect error "
            "even though the user explicitly opted into this plugin."
        )
        assert result["success"] is False
        assert result["provider"] == "openrouter"
        assert "not available" in result["error"]
        # Plugin's transcribe MUST NOT have been called
        assert provider.last_call is None

    def test_available_plugin_dispatches_normally(self):
        provider = _FakeProvider(name="openrouter", available=True)
        transcription_registry.register_provider(provider)

        result = transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "openrouter",
        )
        assert result["success"] is True
        assert provider.last_call is not None

    def test_is_available_raising_treated_as_unavailable(self):
        """Per the ABC contract ``is_available()`` MUST NOT raise; we
        defend anyway so a buggy plugin can't break dispatch."""
        provider = _FakeProvider(
            name="openrouter",
            available_raises=RuntimeError("creds check exploded"),
        )
        transcription_registry.register_provider(provider)

        result = transcription_tools._dispatch_to_plugin_provider(
            "/tmp/audio.mp3", "openrouter",
        )
        assert result is not None
        assert result["success"] is False
        assert result["provider"] == "openrouter"
        assert "not available" in result["error"]
        assert provider.last_call is None

    def test_unavailable_plugin_at_transcribe_audio_level(self):
        """End-to-end: ``stt.provider: openrouter`` + plugin reports
        unavailable → ``transcribe_audio`` returns the unavailability
        envelope, NOT the generic "No STT provider available" message.
        """
        from unittest.mock import patch
        provider = _FakeProvider(name="openrouter", available=False)
        transcription_registry.register_provider(provider)

        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openrouter"}), \
             patch("tools.transcription_tools.is_stt_enabled", return_value=True), \
             patch("tools.transcription_tools._get_provider", return_value="openrouter"):
            result = transcription_tools.transcribe_audio("/tmp/audio.mp3")

        assert result["success"] is False
        # Must surface the plugin's unavailability — NOT the generic
        # "No STT provider available" auto-detect-failure message.
        assert "not available" in result["error"]
        assert "No STT provider available" not in result["error"]
        assert result["provider"] == "openrouter"


# ---------------------------------------------------------------------------
# Language forwarding from config (codex review feedback on PR #30493)
# ---------------------------------------------------------------------------


class TestLanguageForwardingFromConfig:
    """``transcribe_audio`` must forward ``stt.<provider>.language``
    from config to the plugin (mirrors how built-ins read
    ``stt.local.language``).
    """

    def test_language_read_from_provider_namespaced_config(self):
        """``stt.openrouter.language: ja`` reaches the plugin's
        transcribe() call as language='ja'."""
        from unittest.mock import patch
        provider = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(provider)

        stt_config = {
            "provider": "openrouter",
            "openrouter": {"language": "ja"},
        }
        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value=stt_config), \
             patch("tools.transcription_tools.is_stt_enabled", return_value=True), \
             patch("tools.transcription_tools._get_provider", return_value="openrouter"):
            transcription_tools.transcribe_audio("/tmp/audio.mp3")

        assert provider.last_call is not None
        assert provider.last_call["kwargs"]["language"] == "ja"

    def test_model_from_provider_namespaced_config(self):
        """``stt.openrouter.model: whisper-large-v3`` reaches the
        plugin as model='whisper-large-v3' when caller doesn't
        override."""
        from unittest.mock import patch
        provider = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(provider)

        stt_config = {
            "provider": "openrouter",
            "openrouter": {"model": "whisper-large-v3"},
        }
        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value=stt_config), \
             patch("tools.transcription_tools.is_stt_enabled", return_value=True), \
             patch("tools.transcription_tools._get_provider", return_value="openrouter"):
            transcription_tools.transcribe_audio("/tmp/audio.mp3")

        assert provider.last_call["kwargs"]["model"] == "whisper-large-v3"

    def test_caller_model_overrides_config_model(self):
        """An explicit ``model`` arg to transcribe_audio wins over
        ``stt.<provider>.model`` in config."""
        from unittest.mock import patch
        provider = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(provider)

        stt_config = {
            "provider": "openrouter",
            "openrouter": {"model": "config-model"},
        }
        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value=stt_config), \
             patch("tools.transcription_tools.is_stt_enabled", return_value=True), \
             patch("tools.transcription_tools._get_provider", return_value="openrouter"):
            transcription_tools.transcribe_audio(
                "/tmp/audio.mp3", model="explicit-arg-model",
            )

        assert provider.last_call["kwargs"]["model"] == "explicit-arg-model"

    def test_missing_provider_namespace_passes_none(self):
        """No ``stt.<provider>`` subsection → language is None,
        model falls back to caller arg or None. No crash."""
        from unittest.mock import patch
        provider = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(provider)

        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openrouter"}), \
             patch("tools.transcription_tools.is_stt_enabled", return_value=True), \
             patch("tools.transcription_tools._get_provider", return_value="openrouter"):
            transcription_tools.transcribe_audio("/tmp/audio.mp3")

        assert provider.last_call["kwargs"]["language"] is None
        assert provider.last_call["kwargs"]["model"] is None

    def test_non_dict_provider_namespace_does_not_crash(self):
        """If someone accidentally writes ``stt.openrouter: "foo"`` (a
        string instead of a dict), we should not crash — treat as
        empty config."""
        from unittest.mock import patch
        provider = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(provider)

        stt_config = {"provider": "openrouter", "openrouter": "garbage"}
        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value=stt_config), \
             patch("tools.transcription_tools.is_stt_enabled", return_value=True), \
             patch("tools.transcription_tools._get_provider", return_value="openrouter"):
            result = transcription_tools.transcribe_audio("/tmp/audio.mp3")

        # Should still dispatch successfully (config is just ignored)
        assert result["success"] is True
        assert provider.last_call["kwargs"]["language"] is None
        assert provider.last_call["kwargs"]["model"] is None
