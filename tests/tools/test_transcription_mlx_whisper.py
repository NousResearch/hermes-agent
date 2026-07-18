"""Tests for the MLX Whisper transcription provider plugin.

Covers registration, availability gating, model alias resolution,
transcribe dispatch, and error handling.
"""

from __future__ import annotations

import pytest

from agent import transcription_registry
from agent.transcription_provider import TranscriptionProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_provider_module(monkeypatch, *, apple_silicon=True, mlx_importable=True):
    """Reload the mlx_whisper plugin module with controlled platform state."""
    import importlib
    import sys
    import types

    # Force re-import of the plugin module
    for mod in list(sys.modules):
        if mod.startswith("plugins.transcription.mlx_whisper"):
            del sys.modules[mod]

    if apple_silicon:
        monkeypatch.setattr("platform.machine", lambda: "arm64")
        monkeypatch.setattr("platform.system", lambda: "Darwin")
    else:
        monkeypatch.setattr("platform.machine", lambda: "x86_64")
        monkeypatch.setattr("platform.system", lambda: "Linux")

    if mlx_importable:
        fake_mlx = types.ModuleType("mlx_whisper")

        def _fake_transcribe(file_path, path_or_hf_repo=None):
            return {"text": f"transcribed from {path_or_hf_repo or 'default'}"}

        fake_mlx.transcribe = _fake_transcribe
        monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)
    else:
        # Ensure mlx_whisper is NOT importable
        monkeypatch.delitem(sys.modules, "mlx_whisper", raising=False)

    transcription_registry._reset_for_tests()

    # Import the plugin module fresh and call register()
    spec = importlib.util.find_spec("plugins.transcription.mlx_whisper")
    module = importlib.util.module_from_spec(spec)
    sys.modules["plugins.transcription.mlx_whisper"] = module
    spec.loader.exec_module(module)

    # Simulate plugin discovery: call register() with a mock ctx
    class _MockCtx:
        pass

    ctx = _MockCtx()
    ctx.register_transcription_provider = transcription_registry.register_provider
    module.register(ctx)

    return module


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMLXWhisperRegistration:
    """Provider registration and availability."""

    def test_registers_on_apple_silicon(self, monkeypatch):
        """Provider is registered on macOS arm64 with mlx_whisper."""
        _reload_provider_module(monkeypatch, apple_silicon=True, mlx_importable=True)
        provider = transcription_registry.get_provider("mlx_whisper")
        assert provider is not None
        assert provider.name == "mlx_whisper"
        assert isinstance(provider, TranscriptionProvider)

    def test_skips_on_intel(self, monkeypatch):
        """Registration skipped when not on Apple Silicon."""
        _reload_provider_module(monkeypatch, apple_silicon=False, mlx_importable=True)
        provider = transcription_registry.get_provider("mlx_whisper")
        assert provider is None

    def test_registers_but_unavailable_without_package(self, monkeypatch):
        """Registered but is_available() returns False without mlx_whisper."""
        _reload_provider_module(monkeypatch, apple_silicon=True, mlx_importable=False)
        provider = transcription_registry.get_provider("mlx_whisper")
        assert provider is not None
        assert provider.is_available() is False


class TestMLXWhisperModels:
    """Model alias resolution."""

    def _get_provider(self, monkeypatch):
        _reload_provider_module(monkeypatch, apple_silicon=True, mlx_importable=True)
        return transcription_registry.get_provider("mlx_whisper")

    def test_default_model(self, monkeypatch):
        provider = self._get_provider(monkeypatch)
        assert provider.default_model() == "mlx-community/whisper-base-mlx"

    def test_list_models_returns_entries(self, monkeypatch):
        provider = self._get_provider(monkeypatch)
        models = provider.list_models()
        assert len(models) > 0
        ids = {m["id"] for m in models}
        assert "mlx-community/whisper-tiny-mlx" in ids
        assert "mlx-community/whisper-base-mlx" in ids
        assert "mlx-community/whisper-large-v3-mlx" in ids
        # large and large-v3 map to same repo — deduped
        assert "mlx-community/whisper-large-v3-turbo" in ids

    def test_alias_resolution(self, monkeypatch):
        """Short aliases resolve to full HF repo ids."""
        from plugins.transcription.mlx_whisper import _resolve_model

        assert _resolve_model("tiny") == "mlx-community/whisper-tiny-mlx"
        assert _resolve_model("base") == "mlx-community/whisper-base-mlx"
        assert _resolve_model("small") == "mlx-community/whisper-small-mlx"
        assert _resolve_model("large") == "mlx-community/whisper-large-v3-mlx"
        assert _resolve_model("large-v3") == "mlx-community/whisper-large-v3-mlx"
        assert _resolve_model("turbo") == "mlx-community/whisper-large-v3-turbo"

    def test_passthrough_unknown_model(self, monkeypatch):
        """Unknown model id passes through unchanged."""
        from plugins.transcription.mlx_whisper import _resolve_model

        assert _resolve_model("custom/whisper-foo") == "custom/whisper-foo"

    def test_none_model_uses_default(self, monkeypatch):
        from plugins.transcription.mlx_whisper import _resolve_model

        assert _resolve_model(None) == "mlx-community/whisper-base-mlx"


class TestMLXWhisperTranscribe:
    """Transcription dispatch."""

    def _get_provider(self, monkeypatch):
        _reload_provider_module(monkeypatch, apple_silicon=True, mlx_importable=True)
        return transcription_registry.get_provider("mlx_whisper")

    def test_transcribe_success(self, monkeypatch, tmp_path):
        """Successful transcription returns the standard envelope."""
        provider = self._get_provider(monkeypatch)
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00" * 1024)

        result = provider.transcribe(str(audio))
        assert result["success"] is True
        assert "transcribed from" in result["transcript"]
        assert result["provider"] == "mlx_whisper"

    def test_transcribe_unavailable(self, monkeypatch, tmp_path):
        """Returns error envelope when mlx_whisper package is missing."""
        _reload_provider_module(monkeypatch, apple_silicon=True, mlx_importable=False)
        provider = transcription_registry.get_provider("mlx_whisper")
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00" * 1024)

        result = provider.transcribe(str(audio))
        assert result["success"] is False
        assert result["transcript"] == ""
        assert "requires" in result["error"].lower()
        assert result["provider"] == "mlx_whisper"

    def test_transcribe_with_model_alias(self, monkeypatch, tmp_path):
        """Alias resolves to full HF repo id."""
        provider = self._get_provider(monkeypatch)
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00" * 1024)

        result = provider.transcribe(str(audio), model="tiny")
        assert result["success"] is True
        assert "mlx-community/whisper-tiny-mlx" in result["transcript"]

    def test_transcribe_exception_becomes_error_envelope(self, monkeypatch, tmp_path):
        """Provider exceptions are caught and returned as error envelope."""
        import sys

        provider = self._get_provider(monkeypatch)
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00" * 1024)

        # Make the fake mlx_whisper.transcribe raise
        fake_mlx = sys.modules["mlx_whisper"]

        def _raise(*args, **kwargs):
            raise RuntimeError("fake GPU error")

        fake_mlx.transcribe = _raise

        result = provider.transcribe(str(audio))
        assert result["success"] is False
        assert "fake GPU error" in result["error"]
        assert result["provider"] == "mlx_whisper"

    def test_transcribe_not_apple_silicon(self, monkeypatch, tmp_path):
        """Returns error on non-Apple-Silicon."""
        _reload_provider_module(monkeypatch, apple_silicon=False, mlx_importable=True)
        # On intel, the plugin doesn't register at all
        provider = transcription_registry.get_provider("mlx_whisper")
        assert provider is None


class TestMLXWhisperSetupSchema:
    """Provider setup schema for tools picker."""

    def _get_provider(self, monkeypatch):
        _reload_provider_module(monkeypatch, apple_silicon=True, mlx_importable=True)
        return transcription_registry.get_provider("mlx_whisper")

    def test_schema_structure(self, monkeypatch):
        provider = self._get_provider(monkeypatch)
        schema = provider.get_setup_schema()
        assert schema["name"] == "MLX Whisper"
        assert schema["badge"] == "free"
        assert isinstance(schema["env_vars"], list)
        assert len(schema["env_vars"]) == 0
