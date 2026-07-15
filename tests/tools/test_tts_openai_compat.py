"""Tests for the OpenAI-compatible TTS surface.

Covers the building blocks the /v1/models and /v1/audio/speech endpoints rely
on inside ``tools.tts_tool``:

- ``list_available_tts_providers`` — enumerates every usable provider.
- ``_apply_tts_overrides`` — injects per-request voice/speed into config.
- ``text_to_speech_tool`` — honors the new voice/speed/response_format args.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import tts_tool
from tools.tts_tool import (
    _apply_tts_overrides,
    list_available_tts_providers,
    check_tts_requirements,
    text_to_speech_tool,
)


_API_KEYS = (
    "ELEVENLABS_API_KEY",
    "OPENAI_API_KEY",
    "VOICE_TOOLS_OPENAI_KEY",
    "MINIMAX_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "XAI_API_KEY",
    "HERMES_SESSION_PLATFORM",
)


@pytest.fixture
def all_unavailable(monkeypatch):
    """Baseline where no provider is available.

    Tests opt specific providers back in by re-patching the relevant hook or
    setting the relevant env var on top of this baseline.
    """
    for key in _API_KEYS:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(tts_tool, "_import_edge_tts", MagicMock(side_effect=ImportError))
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", MagicMock(side_effect=ImportError))
    monkeypatch.setattr(tts_tool, "_import_openai_client", MagicMock(side_effect=ImportError))
    monkeypatch.setattr(tts_tool, "_import_mistral_client", MagicMock(side_effect=ImportError))
    monkeypatch.setattr(tts_tool, "_has_openai_audio_backend", MagicMock(return_value=False))
    monkeypatch.setattr(tts_tool, "_check_neutts_available", MagicMock(return_value=False))
    monkeypatch.setattr(tts_tool, "_check_kittentts_available", MagicMock(return_value=False))
    monkeypatch.setattr(tts_tool, "_check_piper_available", MagicMock(return_value=False))
    monkeypatch.setattr(
        "tools.xai_http.resolve_xai_http_credentials",
        MagicMock(return_value={}),
    )
    # Neutralize the plugin-registry source too: keep the real
    # _iter_plugin_providers in the path but starve it of inputs, so the
    # baseline is truly "nothing available" and no real plugin discovery runs.
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", MagicMock())
    monkeypatch.setattr("agent.tts_registry.list_providers", MagicMock(return_value=[]))
    yield monkeypatch


# ---------------------------------------------------------------------------
# list_available_tts_providers
# ---------------------------------------------------------------------------


class TestListAvailableProviders:
    def test_empty_when_nothing_available(self, all_unavailable):
        assert list_available_tts_providers({}) == []
        assert check_tts_requirements() is False

    def test_edge_only(self, all_unavailable):
        all_unavailable.setattr(tts_tool, "_import_edge_tts", MagicMock(return_value=MagicMock()))
        assert list_available_tts_providers({}) == ["edge"]
        assert check_tts_requirements() is True

    def test_env_keyed_providers(self, all_unavailable):
        all_unavailable.setenv("MINIMAX_API_KEY", "k")
        all_unavailable.setenv("GEMINI_API_KEY", "k")
        assert list_available_tts_providers({}) == ["gemini", "minimax"]

    def test_openai_requires_backend(self, all_unavailable):
        all_unavailable.setattr(tts_tool, "_import_openai_client", MagicMock(return_value=MagicMock()))
        # No backend yet -> not listed.
        assert "openai" not in list_available_tts_providers({})
        all_unavailable.setattr(tts_tool, "_has_openai_audio_backend", MagicMock(return_value=True))
        assert "openai" in list_available_tts_providers({})

    def test_deepinfra_requires_client_and_key(self, all_unavailable):
        # Enumeration must agree with check_tts_requirements on deepinfra.
        all_unavailable.setattr(tts_tool, "_import_openai_client", MagicMock(return_value=MagicMock()))
        assert "deepinfra" not in list_available_tts_providers({})
        all_unavailable.setenv("DEEPINFRA_API_KEY", "k")
        assert "deepinfra" in list_available_tts_providers({})

    def test_includes_command_providers(self, all_unavailable):
        config = {
            "providers": {
                "mycli": {"type": "command", "command": "say {text_path} {output_path}"},
            }
        }
        assert list_available_tts_providers(config) == ["mycli"]

    def test_result_is_sorted_and_deduped(self, all_unavailable):
        all_unavailable.setattr(tts_tool, "_import_edge_tts", MagicMock(return_value=MagicMock()))
        all_unavailable.setenv("MINIMAX_API_KEY", "k")
        config = {"providers": {"aaa": {"type": "command", "command": "x {output_path}"}}}
        assert list_available_tts_providers(config) == ["aaa", "edge", "minimax"]

    def test_loads_config_when_arg_omitted(self, all_unavailable):
        all_unavailable.setattr(tts_tool, "_import_edge_tts", MagicMock(return_value=MagicMock()))
        all_unavailable.setattr(tts_tool, "_load_tts_config", MagicMock(return_value={}))
        assert list_available_tts_providers() == ["edge"]


# ---------------------------------------------------------------------------
# Plugin-registered providers feed into availability (must be advertised and
# not rejected before dispatch — see _dispatch_to_plugin_provider).
# ---------------------------------------------------------------------------


class TestPluginProviderEnumeration:
    @staticmethod
    def _fake_provider(name):
        return SimpleNamespace(name=name)

    def test_registered_plugin_is_available(self, all_unavailable):
        all_unavailable.setattr(
            "agent.tts_registry.list_providers",
            MagicMock(return_value=[self._fake_provider("myplugin")]),
        )
        # Enumeration surfaces the plugin so /v1/models advertises it and the
        # audio endpoint doesn't reject it before dispatch. (check_tts_requirements
        # is configured-provider-scoped and covered separately.)
        assert list_available_tts_providers({}) == ["myplugin"]

    def test_plugin_listed_alongside_builtin(self, all_unavailable):
        all_unavailable.setattr(tts_tool, "_import_edge_tts", MagicMock(return_value=MagicMock()))
        all_unavailable.setattr(
            "agent.tts_registry.list_providers",
            MagicMock(return_value=[self._fake_provider("myplugin")]),
        )
        assert list_available_tts_providers({}) == ["edge", "myplugin"]

    def test_plugin_shadowing_builtin_is_skipped(self, all_unavailable):
        # The registry blocks this at registration; enumeration re-checks
        # defensively so a shadowed built-in name is never advertised.
        all_unavailable.setattr(
            "agent.tts_registry.list_providers",
            MagicMock(return_value=[self._fake_provider("edge")]),
        )
        assert list_available_tts_providers({}) == []

    def test_discovery_failure_is_non_fatal(self, all_unavailable):
        all_unavailable.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        # Swallowed — plugins simply contribute nothing.
        assert list_available_tts_providers({}) == []


# ---------------------------------------------------------------------------
# _apply_tts_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def test_noop_returns_same_object(self):
        cfg = {"provider": "edge"}
        assert _apply_tts_overrides(cfg, "edge", None, None) is cfg

    def test_voice_written_to_both_keys(self):
        out = _apply_tts_overrides({}, "openai", "nova", None)
        assert out["openai"]["voice"] == "nova"
        assert out["openai"]["voice_id"] == "nova"

    def test_speed_written_to_section_and_top_level(self):
        out = _apply_tts_overrides({}, "edge", None, 1.5)
        assert out["edge"]["speed"] == 1.5
        assert out["speed"] == 1.5

    def test_does_not_mutate_original(self):
        cfg = {"edge": {"voice": "orig"}}
        out = _apply_tts_overrides(cfg, "edge", "new", None)
        assert cfg["edge"]["voice"] == "orig"
        assert out["edge"]["voice"] == "new"

    def test_command_provider_injects_providers_section(self):
        out = _apply_tts_overrides({}, "mycli", "Bob", 2.0)
        # Non-built-in providers read from tts.providers.<name>.
        assert out["providers"]["mycli"]["voice"] == "Bob"
        assert out["providers"]["mycli"]["voice_id"] == "Bob"
        assert out["providers"]["mycli"]["speed"] == 2.0

    def test_replaces_non_dict_section(self):
        out = _apply_tts_overrides({"edge": "weird"}, "edge", "v", None)
        assert out["edge"]["voice"] == "v"


# ---------------------------------------------------------------------------
# text_to_speech_tool param threading
# ---------------------------------------------------------------------------


class TestToolParamThreading:
    def _edge_mock(self, monkeypatch):
        import pathlib

        comm = MagicMock()
        comm.save = AsyncMock(side_effect=lambda p: pathlib.Path(p).write_bytes(b"ID3audio"))
        edge = MagicMock()
        edge.Communicate = MagicMock(return_value=comm)
        monkeypatch.setattr(tts_tool, "_import_edge_tts", MagicMock(return_value=edge))
        return edge

    def test_voice_and_speed_reach_edge(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            MagicMock(return_value={"provider": "edge"}))
        edge = self._edge_mock(monkeypatch)
        res = json.loads(text_to_speech_tool(
            "hello", output_path=str(tmp_path / "o.mp3"),
            provider="edge", voice="zh-CN-XiaoxiaoNeural", speed=1.5,
        ))
        assert res["success"] is True
        kwargs = edge.Communicate.call_args.kwargs
        assert kwargs["voice"] == "zh-CN-XiaoxiaoNeural"
        assert kwargs["rate"] == "+50%"

    def test_response_format_opus_picks_ogg_extension(self, monkeypatch):
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            MagicMock(return_value={"provider": "openai"}))
        monkeypatch.setattr(tts_tool, "_import_openai_client", MagicMock(return_value=MagicMock()))

        captured = {}

        def fake_gen(text, output_path, cfg):
            import pathlib
            captured["path"] = output_path
            captured["voice"] = cfg.get("openai", {}).get("voice")
            pathlib.Path(output_path).write_bytes(b"OggSaudio")
            return output_path

        monkeypatch.setattr(tts_tool, "_generate_openai_tts", fake_gen)
        res = json.loads(text_to_speech_tool(
            "hi", provider="openai", voice="nova", response_format="opus",
        ))
        assert res["success"] is True
        assert captured["path"].endswith(".ogg")
        assert captured["voice"] == "nova"

    def test_response_format_mp3_keeps_mp3(self, monkeypatch):
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            MagicMock(return_value={"provider": "openai"}))
        monkeypatch.setattr(tts_tool, "_import_openai_client", MagicMock(return_value=MagicMock()))

        captured = {}

        def fake_gen(text, output_path, cfg):
            import pathlib
            captured["path"] = output_path
            pathlib.Path(output_path).write_bytes(b"audio")
            return output_path

        monkeypatch.setattr(tts_tool, "_generate_openai_tts", fake_gen)
        json.loads(text_to_speech_tool("hi", provider="openai", response_format="mp3"))
        assert captured["path"].endswith(".mp3")
