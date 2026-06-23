"""Tests for the TTS text-wrapping tag.

Many TTS voices interpret a wrapping style tag, e.g. ``<fast>...</fast>``, to
control delivery. Hermes lets the user set a default tag in config
(``tts.tag`` globally, or ``tts.<provider>.tag`` per provider) and lets the
model override it per call via the ``tag`` parameter on ``text_to_speech``.
"""

import json

import pytest

from tools.tts_tool import (
    TTS_SCHEMA,
    _build_dynamic_tts_schema,
    _normalize_tts_tag,
    _resolve_tts_tag,
)


class TestNormalizeTtsTag:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("fast", "fast"),
            ("  fast  ", "fast"),
            ("<fast>", "fast"),
            ("[fast]", "fast"),
            ("</fast>", "fast"),
            # Stray bracket/slash characters anywhere are removed, not just at
            # the ends, so wrapping can never produce unbalanced tags.
            ("fast>x", "fastx"),
            ("fa/st", "fast"),
            ("<fast><loud>", "fastloud"),
            ("", ""),
            ("   ", ""),
            (None, ""),
        ],
    )
    def test_normalize(self, raw, expected):
        assert _normalize_tts_tag(raw) == expected


class TestResolveTtsTag:
    def test_no_config_no_override(self):
        assert _resolve_tts_tag("edge", {}, None) == ""

    def test_global_tag(self):
        assert _resolve_tts_tag("edge", {"tag": "fast"}, None) == "fast"

    def test_provider_tag_overrides_global(self):
        cfg = {"tag": "fast", "edge": {"tag": "slow"}}
        assert _resolve_tts_tag("edge", cfg, None) == "slow"

    def test_explicit_override_beats_config(self):
        cfg = {"tag": "fast", "edge": {"tag": "slow"}}
        assert _resolve_tts_tag("edge", cfg, "whisper") == "whisper"

    def test_explicit_empty_override_disables_config(self):
        cfg = {"tag": "fast"}
        assert _resolve_tts_tag("edge", cfg, "") == ""

    def test_override_is_normalized(self):
        assert _resolve_tts_tag("edge", {}, "<whisper>") == "whisper"

    def test_command_provider_tag(self):
        cfg = {"providers": {"mycli": {"type": "command", "command": "x", "tag": "loud"}}}
        assert _resolve_tts_tag("mycli", cfg, None) == "loud"

    def test_provider_section_not_a_dict(self):
        cfg = {"tag": "fast", "edge": "nonsense"}
        assert _resolve_tts_tag("edge", cfg, None) == "fast"


class TestDynamicTagSchema:
    """The ``tag`` parameter description reflects the user's configured default."""

    def _tag_description(self, monkeypatch, tts_config):
        monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: tts_config)
        overrides = _build_dynamic_tts_schema()
        return overrides["parameters"]["properties"]["tag"]["description"]

    def test_no_default_leaves_static_schema(self, monkeypatch):
        # No configured tag => empty override, so the static schema is used.
        monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {"provider": "edge"})
        assert _build_dynamic_tts_schema() == {}

    def test_global_default_shown(self, monkeypatch):
        desc = self._tag_description(monkeypatch, {"provider": "edge", "tag": "fast"})
        assert 'configured a default of "fast"' in desc
        assert "<fast>...</fast>" in desc

    def test_provider_default_shown(self, monkeypatch):
        cfg = {"provider": "edge", "tag": "fast", "edge": {"tag": "slow"}}
        desc = self._tag_description(monkeypatch, cfg)
        # The per-provider tag ("slow") wins over the global default ("fast").
        assert 'configured a default of "slow"' in desc
        assert "<slow>...</slow>" in desc

    def test_other_properties_preserved(self, monkeypatch):
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config", lambda: {"provider": "edge", "tag": "fast"}
        )
        params = _build_dynamic_tts_schema()["parameters"]
        assert set(params["properties"]) == set(TTS_SCHEMA["parameters"]["properties"])
        assert params["required"] == TTS_SCHEMA["parameters"]["required"]


class TestTextToSpeechToolWrapsTag:
    """End-to-end: the resolved tag wraps the text handed to the provider."""

    def _run(self, tmp_path, monkeypatch, tts_config, **kwargs):
        captured = {}

        def fake_edge(text, out, cfg):
            captured["text"] = text
            with open(out, "wb") as f:
                f.write(b"\x00")
            return out

        async def fake_edge_async(text, out, cfg):
            return fake_edge(text, out, cfg)

        monkeypatch.setattr("tools.tts_tool._generate_edge_tts", fake_edge_async)
        monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: tts_config)

        from tools.tts_tool import text_to_speech_tool

        out = str(tmp_path / "out.mp3")
        result = json.loads(text_to_speech_tool(text="hello", output_path=out, **kwargs))
        assert result["success"] is True
        return captured["text"]

    def test_no_tag_leaves_text_unchanged(self, tmp_path, monkeypatch):
        assert self._run(tmp_path, monkeypatch, {"provider": "edge"}) == "hello"

    def test_param_wraps_text(self, tmp_path, monkeypatch):
        text = self._run(tmp_path, monkeypatch, {"provider": "edge"}, tag="fast")
        assert text == "<fast>hello</fast>"

    def test_config_default_wraps_text(self, tmp_path, monkeypatch):
        text = self._run(tmp_path, monkeypatch, {"provider": "edge", "tag": "fast"})
        assert text == "<fast>hello</fast>"

    def test_param_overrides_config_default(self, tmp_path, monkeypatch):
        text = self._run(
            tmp_path, monkeypatch, {"provider": "edge", "tag": "fast"}, tag="whisper"
        )
        assert text == "<whisper>hello</whisper>"

    def test_empty_param_disables_config_default(self, tmp_path, monkeypatch):
        text = self._run(
            tmp_path, monkeypatch, {"provider": "edge", "tag": "fast"}, tag=""
        )
        assert text == "hello"

    def test_provider_config_overrides_global_config(self, tmp_path, monkeypatch):
        text = self._run(
            tmp_path, monkeypatch, {"provider": "edge", "tag": "fast", "edge": {"tag": "slow"}}
        )
        assert text == "<slow>hello</slow>"

    def test_tag_wraps_after_truncation_keeps_closing_tag(self, tmp_path, monkeypatch):
        """Wrapping happens after truncation, so the spoken text is capped at the
        provider limit while the closing tag survives intact."""
        captured = {}

        def fake_openai(text, out, cfg):
            captured["text"] = text
            with open(out, "wb") as f:
                f.write(b"\x00")
            return out

        monkeypatch.setattr("tools.tts_tool._generate_openai_tts", fake_openai)
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config", lambda: {"provider": "openai"}
        )

        from tools.tts_tool import text_to_speech_tool

        out = str(tmp_path / "out.mp3")
        # OpenAI caps input at 4096 chars; feed well over it.
        result = json.loads(
            text_to_speech_tool(text="A" * 5000, output_path=out, tag="fast")
        )
        assert result["success"] is True

        wrapped = captured["text"]
        assert wrapped.startswith("<fast>")
        assert wrapped.endswith("</fast>")
        # Inner spoken text is truncated to the provider cap, not the wrapped whole.
        assert wrapped[len("<fast>"):-len("</fast>")] == "A" * 4096
