"""Tests for the TTS text-wrapping tag.

Many TTS voices interpret a wrapping style tag, e.g. ``<fast>...</fast>`` or
``[fast]...[/fast]``, to control delivery. Hermes lets the user set a default
tag in config (``tts.tag`` globally, or ``tts.<provider>.tag`` per provider)
and lets the model override it per call via the ``tag`` parameter on
``text_to_speech``. The bracket syntax is resolved per provider.
"""

import json

import pytest

from tools.tts_tool import (
    TTS_SCHEMA,
    TTS_TAG_SYNTAX_ANGLE,
    TTS_TAG_SYNTAX_SQUARE,
    _apply_tts_tag,
    _build_dynamic_tts_schema,
    _normalize_tts_tag,
    _resolve_tts_tag,
    _resolve_tts_tag_syntax,
)


class TestNormalizeTtsTag:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("fast", "fast"),
            ("  fast  ", "fast"),
            ("FAST", "fast"),
            ("<fast>", "fast"),
            ("[fast]", "fast"),
            ("</fast>", "fast"),
            ("[/fast]", "fast"),
            ("< fast >", "fast"),
            ("sing-song", "sing-song"),
            ("long_pause", "long_pause"),
            ("fast2", "fast2"),
            ("", ""),
            ("   ", ""),
            (None, ""),
            # Anything that is not a bare tag name after the framing is
            # stripped is rejected outright, so wrapping can never emit
            # unbalanced tags or attribute-carrying markup.
            ("fast>x", ""),
            ("fa/st", ""),
            ("<fast><loud>", ""),
            ('<fast mode="x">', ""),
            ("fast slow", ""),
            ("'fast'", ""),
            ("-fast", ""),
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

    def test_invalid_tag_name_rejected(self):
        assert _resolve_tts_tag("edge", {"tag": 'fast mode="x"'}, None) == ""

    def test_xai_accepts_documented_wrapping_tag(self):
        assert _resolve_tts_tag("xai", {"tag": "whisper"}, None) == "whisper"

    def test_xai_rejects_unknown_tag(self):
        assert _resolve_tts_tag("xai", {"tag": "banana"}, None) == ""

    def test_xai_rejects_unknown_override(self):
        assert _resolve_tts_tag("xai", {}, "banana") == ""

    def test_xai_rejects_inline_only_tag(self):
        # [sigh] is an inline xAI speech tag, not a wrapping one.
        assert _resolve_tts_tag("xai", {"tag": "sigh"}, None) == ""

    def test_other_providers_accept_freeform_names(self):
        assert _resolve_tts_tag("edge", {"tag": "banana"}, None) == "banana"


class TestResolveTtsTagSyntax:
    @pytest.mark.parametrize(
        "provider,expected",
        [
            ("edge", TTS_TAG_SYNTAX_ANGLE),
            ("openai", TTS_TAG_SYNTAX_ANGLE),
            ("xai", TTS_TAG_SYNTAX_SQUARE),
            ("gemini", TTS_TAG_SYNTAX_SQUARE),
            ("mycli", TTS_TAG_SYNTAX_ANGLE),
        ],
    )
    def test_defaults(self, provider, expected):
        assert _resolve_tts_tag_syntax(provider, {}) == expected

    def test_global_config_applies_to_unmapped_provider(self):
        cfg = {"tag_syntax": "square"}
        assert _resolve_tts_tag_syntax("edge", cfg) == TTS_TAG_SYNTAX_SQUARE

    def test_builtin_mapping_wins_over_config(self):
        # xAI rejects angle-bracket tags, so config cannot force them.
        cfg = {"tag_syntax": "angle", "xai": {"tag_syntax": "angle"}}
        assert _resolve_tts_tag_syntax("xai", cfg) == TTS_TAG_SYNTAX_SQUARE

    def test_command_provider_config_wins_over_global(self):
        cfg = {
            "tag_syntax": "angle",
            "providers": {"mycli": {"type": "command", "command": "x", "tag_syntax": "square"}},
        }
        assert _resolve_tts_tag_syntax("mycli", cfg) == TTS_TAG_SYNTAX_SQUARE

    def test_invalid_value_ignored(self):
        assert _resolve_tts_tag_syntax("edge", {"tag_syntax": "curly"}) == TTS_TAG_SYNTAX_ANGLE


class TestApplyTtsTag:
    def test_angle(self):
        assert _apply_tts_tag("hello", "fast", TTS_TAG_SYNTAX_ANGLE) == "<fast>hello</fast>"

    def test_square(self):
        assert _apply_tts_tag("hello", "fast", TTS_TAG_SYNTAX_SQUARE) == "[fast]hello[/fast]"

    def test_default_syntax_is_angle(self):
        assert _apply_tts_tag("hello", "fast") == "<fast>hello</fast>"

    def test_empty_tag_returns_text_unchanged(self):
        assert _apply_tts_tag("hello", "") == "hello"

    def test_invalid_tag_returns_text_unchanged(self):
        assert _apply_tts_tag("hello", 'fast mode="x"') == "hello"


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

    def test_square_syntax_provider_shows_square_example(self, monkeypatch):
        desc = self._tag_description(monkeypatch, {"provider": "xai", "tag": "whisper"})
        assert 'configured a default of "whisper"' in desc
        assert "[whisper]...[/whisper]" in desc

    def test_unsupported_provider_tag_leaves_static_schema(self, monkeypatch):
        # xAI does not support "banana", so there is no default to surface.
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config", lambda: {"provider": "xai", "tag": "banana"}
        )
        assert _build_dynamic_tts_schema() == {}

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

    def test_invalid_tag_param_leaves_text_unchanged(self, tmp_path, monkeypatch):
        text = self._run(
            tmp_path, monkeypatch, {"provider": "edge"}, tag='fast mode="x"'
        )
        assert text == "hello"

    def _run_xai(self, tmp_path, monkeypatch, tts_config, **kwargs):
        captured = {}

        def fake_xai(text, out, cfg):
            captured["text"] = text
            with open(out, "wb") as f:
                f.write(b"\x00")
            return out

        monkeypatch.setattr("tools.tts_tool._generate_xai_tts", fake_xai)
        monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: tts_config)

        from tools.tts_tool import text_to_speech_tool

        out = str(tmp_path / "out.mp3")
        result = json.loads(text_to_speech_tool(text="hello", output_path=out, **kwargs))
        assert result["success"] is True
        return captured["text"]

    def test_xai_wraps_with_square_bracket_syntax(self, tmp_path, monkeypatch):
        text = self._run_xai(tmp_path, monkeypatch, {"provider": "xai"}, tag="whisper")
        assert text == "[whisper]hello[/whisper]"

    def test_xai_unknown_tag_leaves_text_unchanged(self, tmp_path, monkeypatch):
        text = self._run_xai(tmp_path, monkeypatch, {"provider": "xai"}, tag="banana")
        assert text == "hello"

    def test_gemini_config_tag_uses_square_bracket_syntax(self, tmp_path, monkeypatch):
        captured = {}

        def fake_gemini(text, out, cfg):
            captured["text"] = text
            with open(out, "wb") as f:
                f.write(b"\x00")
            return out

        monkeypatch.setattr("tools.tts_tool._generate_gemini_tts", fake_gemini)
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config",
            lambda: {"provider": "gemini", "tag": "whispers"},
        )

        from tools.tts_tool import text_to_speech_tool

        out = str(tmp_path / "out.mp3")
        result = json.loads(text_to_speech_tool(text="hello", output_path=out))
        assert result["success"] is True
        assert captured["text"] == "[whispers]hello[/whispers]"

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
