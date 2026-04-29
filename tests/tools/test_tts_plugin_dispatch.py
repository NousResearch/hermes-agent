"""Tests for plugin-registry dispatch inside tools/tts_tool.py.

These lock in the prepend-dispatcher contract: when ``tts.provider`` names
a registered plugin provider (not in ``LEGACY_TTS_PROVIDERS``), the
``text_to_speech`` tool must route to that plugin before any legacy
hardcoded branch runs. Legacy names and unset config must pass through
untouched.
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Full isolation: registry reset + HERMES_HOME + plugin discovery bypass."""
    from agent import tts_registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tts_registry._reset_for_tests()

    # Monkeypatch _ensure_plugins_discovered to be a no-op so tests don't
    # accidentally trigger bundled-plugin registration.
    try:
        from hermes_cli import plugins as _plugins

        monkeypatch.setattr(_plugins, "_ensure_plugins_discovered", lambda *_a, **_k: None)
    except Exception:
        pass

    yield
    tts_registry._reset_for_tests()


def _write_config(tmp_path, provider: str | None):
    """Write a minimal config.yaml selecting ``provider`` (or omitting it)."""
    import yaml

    data = {}
    if provider is not None:
        data["tts"] = {"provider": provider}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data))


def _register_fake_tts(name: str, *, write_file: bool = True,
                       output_format: str = "mp3",
                       native_opus: bool = False,
                       voice_compatible: bool = False,
                       raises: Exception | None = None,
                       returns_error: dict | None = None):
    """Register a minimal plugin TTS provider for dispatch tests."""
    from agent.tts_provider import TtsProvider
    from agent.tts_registry import register_provider

    class FakeTts(TtsProvider):
        @property
        def name(self) -> str:
            return name

        def synthesize(self, text, output_path, config):
            if raises is not None:
                raise raises
            if returns_error is not None:
                return returns_error
            if write_file:
                # Ensure a real file exists for the finalizer check.
                with open(output_path, "wb") as f:
                    f.write(b"\x00" * 1024)
            return {
                "success": True,
                "file_path": output_path,
                "format": output_format,
                "native_opus": native_opus,
                "voice_compatible": voice_compatible,
            }

    register_provider(FakeTts())


class TestDispatcher:
    def test_legacy_name_passes_through(self, tmp_path, monkeypatch):
        """provider='edge' must NOT enter the plugin dispatch (returns None)."""
        from tools.tts_tool import _dispatch_to_plugin_tts_provider

        _write_config(tmp_path, "edge")
        result = _dispatch_to_plugin_tts_provider("hi", "/tmp/x.mp3", {}, "edge")
        assert result is None

    def test_unset_provider_passes_through(self, tmp_path):
        from tools.tts_tool import _dispatch_to_plugin_tts_provider

        _write_config(tmp_path, None)
        result = _dispatch_to_plugin_tts_provider("hi", "/tmp/x.mp3", {}, "")
        assert result is None

    def test_configured_plugin_routes_through(self, tmp_path):
        from tools.tts_tool import _dispatch_to_plugin_tts_provider

        _register_fake_tts("volcengine", write_file=True)
        result = _dispatch_to_plugin_tts_provider(
            "hi", str(tmp_path / "out.mp3"), {}, "volcengine"
        )
        assert result is not None
        payload = json.loads(result)
        assert payload["success"] is True
        assert payload["provider"] == "volcengine"

    def test_configured_plugin_not_registered_returns_error(self, tmp_path):
        """If tts.provider=unknown-plugin but nothing is registered, surface
        a helpful error rather than silently falling through."""
        from tools.tts_tool import _dispatch_to_plugin_tts_provider

        result = _dispatch_to_plugin_tts_provider(
            "hi", str(tmp_path / "out.mp3"), {}, "mystery-backend"
        )
        assert result is not None
        payload = json.loads(result)
        assert payload["success"] is False
        assert "not registered" in payload["error"].lower()
        assert payload["error_type"] == "provider_not_registered"

    def test_plugin_returning_error_dict_is_surfaced(self, tmp_path):
        from tools.tts_tool import _dispatch_to_plugin_tts_provider

        _register_fake_tts(
            "bad-creds",
            returns_error={
                "success": False,
                "error": "missing API key",
                "error_type": "config",
            },
        )
        result = _dispatch_to_plugin_tts_provider(
            "hi", str(tmp_path / "out.mp3"), {}, "bad-creds"
        )
        payload = json.loads(result)
        assert payload["success"] is False
        assert "missing API key" in payload["error"]
        assert payload["error_type"] == "config"

    def test_plugin_raising_valueerror_becomes_config_error(self, tmp_path):
        from tools.tts_tool import _dispatch_to_plugin_tts_provider

        _register_fake_tts("bad", raises=ValueError("cred missing"))
        result = _dispatch_to_plugin_tts_provider(
            "hi", str(tmp_path / "out.mp3"), {}, "bad"
        )
        payload = json.loads(result)
        assert payload["success"] is False
        assert payload["error_type"] == "config"
        assert "cred missing" in payload["error"]

    def test_plugin_raising_runtimeerror_becomes_runtime_error(self, tmp_path):
        from tools.tts_tool import _dispatch_to_plugin_tts_provider

        _register_fake_tts("bad", raises=RuntimeError("api down"))
        result = _dispatch_to_plugin_tts_provider(
            "hi", str(tmp_path / "out.mp3"), {}, "bad"
        )
        payload = json.loads(result)
        assert payload["success"] is False
        assert payload["error_type"] == "runtime"
        assert "api down" in payload["error"]

    def test_plugin_native_opus_path_marked_voice_compatible(self, tmp_path):
        from tools.tts_tool import _dispatch_to_plugin_tts_provider

        _register_fake_tts(
            "ogg-backend",
            write_file=True,
            output_format="ogg",
            native_opus=True,
            voice_compatible=True,
        )
        out = str(tmp_path / "out.ogg")
        result = _dispatch_to_plugin_tts_provider("hi", out, {}, "ogg-backend")
        payload = json.loads(result)
        assert payload["success"] is True
        assert payload["voice_compatible"] is True
        assert payload["media_tag"].startswith("[[audio_as_voice]]")


class TestRequirements:
    def test_check_tts_requirements_true_with_only_plugin(self, tmp_path, monkeypatch):
        """If no hardcoded provider is available but a plugin reports
        is_available() True, check_tts_requirements should be True."""
        from agent.tts_registry import register_provider
        from agent.tts_provider import TtsProvider

        # Disable edge-tts so hardcoded path can't succeed
        monkeypatch.setattr(
            "tools.tts_tool._import_edge_tts",
            lambda: (_ for _ in ()).throw(ImportError()),
        )
        monkeypatch.setattr(
            "tools.tts_tool._import_elevenlabs",
            lambda: (_ for _ in ()).throw(ImportError()),
        )
        monkeypatch.setattr(
            "tools.tts_tool._import_openai_client",
            lambda: (_ for _ in ()).throw(ImportError()),
        )
        monkeypatch.setattr(
            "tools.tts_tool._import_mistral_client",
            lambda: (_ for _ in ()).throw(ImportError()),
        )
        # clear all env checks
        for k in ("MINIMAX_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(k, raising=False)

        class GoodPlugin(TtsProvider):
            @property
            def name(self):
                return "goodplugin"

            def is_available(self):
                return True

            def synthesize(self, text, output_path, config):
                return {"success": True, "file_path": output_path, "format": "mp3",
                        "native_opus": False, "voice_compatible": False}

        register_provider(GoodPlugin())

        from tools.tts_tool import check_tts_requirements
        assert check_tts_requirements() is True


class TestResolveMaxTextLength:
    def test_resolve_max_text_length_uses_plugin_value(self):
        """_resolve_max_text_length should ask a registered plugin for its
        cap when the configured provider is that plugin."""
        from agent.tts_provider import TtsProvider
        from agent.tts_registry import register_provider

        class BigPlugin(TtsProvider):
            @property
            def name(self):
                return "bigplugin"

            def max_text_length(self):
                return 9999

            def synthesize(self, text, output_path, config):
                return {"success": True, "file_path": output_path, "format": "mp3",
                        "native_opus": False, "voice_compatible": False}

        register_provider(BigPlugin())

        from tools.tts_tool import _resolve_max_text_length
        cap = _resolve_max_text_length("bigplugin", {})
        assert cap == 9999
