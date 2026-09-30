"""Tests for the TTS provider fallback chain (tts.fallback_chain).

Covers the config resolver (dedupe, invalid entries, opt-out), the entry-point
replay semantics (primary failure -> next provider, caller-explicit provider=
disables the chain, output-path rejections never launder through a fallback),
and the previous-behavior contract when no chain is configured.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _resolve_tts_fallback_chain
# ---------------------------------------------------------------------------

class TestResolveFallbackChain:
    def test_empty_config_returns_empty(self):
        from tools.tts_tool import _resolve_tts_fallback_chain
        assert _resolve_tts_fallback_chain({}, "edge") == []

    def test_basic_chain(self):
        from tools.tts_tool import _resolve_tts_fallback_chain
        assert _resolve_tts_fallback_chain(
            {"fallback_chain": ["deepinfra", "openai"]}, "edge") == ["deepinfra", "openai"]

    def test_drops_primary_and_duplicates_case_insensitive(self):
        from tools.tts_tool import _resolve_tts_fallback_chain
        chain = _resolve_tts_fallback_chain(
            {"fallback_chain": ["EDGE", "deepinfra", "DeepInfra", "edge"]}, "edge")
        assert chain == ["deepinfra"]

    def test_non_string_entries_skipped(self):
        from tools.tts_tool import _resolve_tts_fallback_chain
        assert _resolve_tts_fallback_chain(
            {"fallback_chain": ["deepinfra", 42, None, "  " ]}, "edge") == ["deepinfra"]

    def test_non_list_value_disables_chain(self):
        from tools.tts_tool import _resolve_tts_fallback_chain
        assert _resolve_tts_fallback_chain({"fallback_chain": "off"}, "edge") == []
        assert _resolve_tts_fallback_chain({"fallback_chain": "deepinfra"}, "edge") == []

    def test_nous_alias_resolves_like_primary_path(self):
        from tools.tts_tool import _resolve_tts_fallback_chain
        assert _resolve_tts_fallback_chain({"fallback_chain": ["nous"]}, "edge") == ["openai"]

    def test_nous_alias_collapses_against_openai_primary(self):
        from tools.tts_tool import _resolve_tts_fallback_chain
        assert _resolve_tts_fallback_chain({"fallback_chain": ["nous"]}, "openai") == []


# ---------------------------------------------------------------------------
# text_to_speech_tool replay semantics
# ---------------------------------------------------------------------------

def _wire_tool(monkeypatch, tts_config, generator_behaviors):
    """Patch the tool entry point's seams. generator_behaviors maps provider name to
    either bytes (success: written to the target file) or an Exception to raise."""
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: dict(tts_config))

    def fake_generate(provider):
        def _gen(text, file_str, config, **kw):
            behavior = generator_behaviors[provider]
            if isinstance(behavior, Exception):
                raise behavior
            with open(file_str, "wb") as f:
                f.write(behavior)
        return _gen

    patches = [
        patch("tools.tts_tool._generate_edge_tts", side_effect=fake_generate("edge")),
        patch("tools.tts_tool._generate_deepinfra_tts", side_effect=fake_generate("deepinfra")),
        patch("tools.tts_tool._get_provider", return_value=tts_config.get("provider", "edge")),
        patch("tools.tts_tool._resolve_command_provider_config", return_value=None),
        # Bypass the SDK-availability probe: the test mocks the generator itself.
        patch("tools.tts_tool._select_builtin_engine", side_effect=lambda prov: (prov, None)),
        patch("gateway.session_context.get_session_env", return_value=""),
    ]
    return patches


class TestToolFallbackReplay:
    def test_primary_failure_recovers_with_next_provider(self, tmp_path, monkeypatch):
        out = str(tmp_path / "out.mp3")
        patches = _wire_tool(
            monkeypatch,
            {"provider": "edge", "fallback_chain": ["deepinfra"], "edge": {}, "deepinfra": {}},
            {"edge": RuntimeError("Microsoft endpoint down"),
             "deepinfra": b"\x00\x01\x02\x03"},
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            from tools.tts_tool import text_to_speech_tool
            result = json.loads(text_to_speech_tool("Hola", out))
        assert result["success"] is True
        assert result["provider"] == "deepinfra"

    def test_success_on_primary_never_reaches_fallback(self, tmp_path, monkeypatch):
        out = str(tmp_path / "out.mp3")
        deepinfra = MagicMock(side_effect=AssertionError("must not be called"))
        patches = _wire_tool(
            monkeypatch,
            {"provider": "edge", "fallback_chain": ["deepinfra"], "edge": {}, "deepinfra": {}},
            {"edge": b"\x00\x01\x02\x03", "deepinfra": RuntimeError("unused")},
        )
        with patch("tools.tts_tool._generate_edge_tts", side_effect=lambda *a, **k: open(a[1], "wb").write(b"\x00\x01\x02\x03")), \
             patch("tools.tts_tool._generate_deepinfra_tts", deepinfra), \
             patch("tools.tts_tool._select_builtin_engine", side_effect=lambda prov: (prov, None)), \
             patches[2], patches[3], patches[4]:
            from tools.tts_tool import text_to_speech_tool
            result = json.loads(text_to_speech_tool("Hola", out))
        assert result["success"] is True
        assert result["provider"] == "edge"
        deepinfra.assert_not_called()

    def test_caller_explicit_provider_disables_chain(self, tmp_path, monkeypatch):
        out = str(tmp_path / "out.mp3")
        deepinfra = MagicMock(side_effect=AssertionError("must not be called"))
        # ConnectionError (not RuntimeError): _run_edge_tts treats RuntimeErrors as an
        # event-loop conflict and re-invokes once — that retry is out of scope here.
        edge_boom = MagicMock(side_effect=ConnectionError("edge down"))
        with patch("tools.tts_tool._load_tts_config",
                   lambda: {"provider": "edge", "fallback_chain": ["deepinfra"], "edge": {}, "deepinfra": {}}), \
             patch("tools.tts_tool._generate_edge_tts", edge_boom), \
             patch("tools.tts_tool._generate_deepinfra_tts", deepinfra), \
             patch("tools.tts_tool._select_builtin_engine", side_effect=lambda prov: (prov, None)), \
             patch("tools.tts_tool._resolve_command_provider_config", return_value=None), \
             patch("tools.tts_tool._select_builtin_engine", side_effect=lambda prov: (prov, None)), \
             patch("gateway.session_context.get_session_env", return_value=""):
            from tools.tts_tool import text_to_speech_tool
            result = json.loads(text_to_speech_tool("Hola", out, provider="edge"))
        assert result["success"] is False
        edge_boom.assert_called_once()
        deepinfra.assert_not_called()

    def test_all_providers_fail_reports_last_error(self, tmp_path, monkeypatch):
        out = str(tmp_path / "out.mp3")
        patches = _wire_tool(
            monkeypatch,
            {"provider": "edge", "fallback_chain": ["deepinfra"], "edge": {}, "deepinfra": {}},
            {"edge": RuntimeError("edge down"), "deepinfra": RuntimeError("deepinfra down")},
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            from tools.tts_tool import text_to_speech_tool
            result = json.loads(text_to_speech_tool("Hola", out))
        assert result["success"] is False
        assert "deepinfra down" in result["error"]

    def test_no_chain_keeps_previous_behavior(self, tmp_path, monkeypatch):
        out = str(tmp_path / "out.mp3")
        patches = _wire_tool(
            monkeypatch,
            {"provider": "edge", "edge": {}},
            {"edge": RuntimeError("edge down"), "deepinfra": RuntimeError("unused")},
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            from tools.tts_tool import text_to_speech_tool
            result = json.loads(text_to_speech_tool("Hola", out))
        assert result["success"] is False
        assert "edge down" in result["error"]

    def test_unsafe_output_path_not_laundered_through_fallback(self, tmp_path, monkeypatch):
        deepinfra = MagicMock(side_effect=AssertionError("must not be called"))
        with patch("tools.tts_tool._load_tts_config",
                   lambda: {"provider": "edge", "fallback_chain": ["deepinfra"], "edge": {}, "deepinfra": {}}), \
             patch("tools.tts_tool._generate_deepinfra_tts", deepinfra), \
             patch("tools.tts_tool._select_builtin_engine", side_effect=lambda prov: (prov, None)), \
             patch("tools.tts_tool._resolve_command_provider_config", return_value=None), \
             patch("gateway.session_context.get_session_env", return_value=""):
            from tools.tts_tool import text_to_speech_tool
            result = json.loads(text_to_speech_tool("Hola", "../escape/out.mp3"))
        assert result["success"] is False
        assert "traversal" in result["error"]
        deepinfra.assert_not_called()
