"""Tests that text_to_speech_tool strips markdown before dispatching to provider.

Regression for the bug where the agent-callable TTS entry point passed raw
markdown (e.g. **bold**, # headers, `code`) to providers like Edge TTS, which
then verbalized the markdown artifacts ("double-asterisk Bold double-asterisk").

Two other TTS call sites (gateway/run.py:_send_voice_reply and
gateway/platforms/base.py Auto-TTS path) already strip markdown; this aligns
the third call site (text_to_speech_tool) with them.
"""

import json
from unittest.mock import patch

import pytest

from tools.tts_tool import text_to_speech_tool


class TestTextToSpeechToolMarkdownStrip:
    """text_to_speech_tool() must strip markdown before provider dispatch."""

    def test_strips_bold_before_provider_call(self, tmp_path):
        """**bold** must reach the provider as plain 'bold'."""
        captured = {}

        def fake_edge(text, output_path, tts_config):
            captured["text"] = text
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-mp3")
            return output_path

        with patch("tools.tts_tool._generate_edge_tts", side_effect=fake_edge), \
             patch("tools.tts_tool._import_edge_tts", return_value=None), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "edge"}):
            text_to_speech_tool(
                text="This is **bold** text",
                output_path=str(tmp_path / "out.mp3"),
            )

        assert "**" not in captured["text"]
        assert "bold" in captured["text"]

    def test_strips_headers_before_provider_call(self, tmp_path):
        """# Header must reach the provider without leading hashes."""
        captured = {}

        def fake_edge(text, output_path, tts_config):
            captured["text"] = text
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-mp3")
            return output_path

        with patch("tools.tts_tool._generate_edge_tts", side_effect=fake_edge), \
             patch("tools.tts_tool._import_edge_tts", return_value=None), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "edge"}):
            text_to_speech_tool(
                text="## Summary\nSome text",
                output_path=str(tmp_path / "out.mp3"),
            )

        assert "##" not in captured["text"]
        assert "Summary" in captured["text"]

    def test_strips_inline_code_before_provider_call(self, tmp_path):
        """Backtick-wrapped inline code must lose its backticks."""
        captured = {}

        def fake_edge(text, output_path, tts_config):
            captured["text"] = text
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-mp3")
            return output_path

        with patch("tools.tts_tool._generate_edge_tts", side_effect=fake_edge), \
             patch("tools.tts_tool._import_edge_tts", return_value=None), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "edge"}):
            text_to_speech_tool(
                text="Run `pip install foo` to install",
                output_path=str(tmp_path / "out.mp3"),
            )

        assert "`" not in captured["text"]
        assert "pip install foo" in captured["text"]

    def test_strips_list_markers_before_provider_call(self, tmp_path):
        """- and * list markers should not be spoken."""
        captured = {}

        def fake_edge(text, output_path, tts_config):
            captured["text"] = text
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-mp3")
            return output_path

        with patch("tools.tts_tool._generate_edge_tts", side_effect=fake_edge), \
             patch("tools.tts_tool._import_edge_tts", return_value=None), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "edge"}):
            text_to_speech_tool(
                text="- item one\n- item two",
                output_path=str(tmp_path / "out.mp3"),
            )

        assert "- " not in captured["text"]
        assert "item one" in captured["text"]
        assert "item two" in captured["text"]

    def test_truncation_uses_stripped_length(self, tmp_path):
        """Provider max_len budget must apply to spoken length, not raw markdown."""
        # 50 chars of '*' bracketing 50 chars of plain text = 100 raw, ~50 spoken.
        # If max_len is 80 and we don't strip first, truncation cuts spoken text.
        # After strip-first, all 50 spoken chars survive.
        raw = "*" * 25 + "Hello clean world. This is the actual content." + "*" * 25
        captured = {}

        def fake_edge(text, output_path, tts_config):
            captured["text"] = text
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-mp3")
            return output_path

        with patch("tools.tts_tool._generate_edge_tts", side_effect=fake_edge), \
             patch("tools.tts_tool._import_edge_tts", return_value=None), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "edge"}), \
             patch("tools.tts_tool._resolve_max_text_length", return_value=80):
            text_to_speech_tool(
                text=raw,
                output_path=str(tmp_path / "out.mp3"),
            )

        # The full plain content should survive (50 chars < 80 budget) once stripped.
        assert "Hello clean world" in captured["text"]
        assert "actual content" in captured["text"]

    def test_command_provider_skip_markdown_strip_opt_out(self, tmp_path):
        """A command provider with skip_markdown_strip:true gets raw text."""
        captured = {}

        def fake_command(text, output_path, provider_name, config, tts_config):
            captured["text"] = text
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-wav")
            return output_path

        tts_config = {
            "provider": "ssml-piper",
            "providers": {
                "ssml-piper": {
                    "type": "command",
                    "command": "echo {input_path} > {output_path}",
                    "output_format": "wav",
                    "skip_markdown_strip": True,
                },
            },
        }

        with patch("tools.tts_tool._generate_command_tts", side_effect=fake_command), \
             patch("tools.tts_tool._load_tts_config", return_value=tts_config):
            text_to_speech_tool(
                text="<break time=\"500ms\"/>**Important**",
                output_path=str(tmp_path / "out.wav"),
            )

        # SSML and markdown both pass through untouched.
        assert "<break" in captured["text"]
        assert "**Important**" in captured["text"]

    def test_command_provider_default_strips_markdown(self, tmp_path):
        """Without skip_markdown_strip, command provider also gets stripped text."""
        captured = {}

        def fake_command(text, output_path, provider_name, config, tts_config):
            captured["text"] = text
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-wav")
            return output_path

        tts_config = {
            "provider": "voxcpm",
            "providers": {
                "voxcpm": {
                    "type": "command",
                    "command": "voxcpm --in {input_path} --out {output_path}",
                    "output_format": "wav",
                },
            },
        }

        with patch("tools.tts_tool._generate_command_tts", side_effect=fake_command), \
             patch("tools.tts_tool._load_tts_config", return_value=tts_config):
            text_to_speech_tool(
                text="This is **bold** text",
                output_path=str(tmp_path / "out.wav"),
            )

        assert "**" not in captured["text"]
        assert "bold" in captured["text"]


def _command_tts_config(**provider_overrides):
    provider = {
        "type": "command",
        "command": "echo {input_path} > {output_path}",
        "output_format": "wav",
        **provider_overrides,
    }
    return {"provider": "ssml-cmd", "providers": {"ssml-cmd": provider}}


class TestSkipMarkdownStripBooleanSemantics:
    """Review blocker: the opt-out must follow the repository's shared boolean
    config semantics (like the adjacent ``voice_compatible``) — a quoted
    false-like value must keep stripping ENABLED."""

    @pytest.mark.parametrize("false_like", ["false", "0", "no", "off", "False", ""])
    def test_false_like_strings_keep_stripping_enabled(self, tmp_path, false_like):
        captured = {}

        def fake_command(text, output_path, provider_name, config, tts_config):
            captured["text"] = text
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-wav")
            return output_path

        cfg = _command_tts_config(skip_markdown_strip=false_like)
        with patch("tools.tts_tool._generate_command_tts", side_effect=fake_command), \
             patch("tools.tts_tool._load_tts_config", return_value=cfg):
            text_to_speech_tool(
                text="This is **bold** text",
                output_path=str(tmp_path / "out.wav"),
            )

        # A false-like string is DISABLED: markdown is stripped as by default.
        assert "**" not in captured["text"]
        assert "bold" in captured["text"]

    @pytest.mark.parametrize("true_like", [True, "true", "1", "yes"])
    def test_true_like_values_enable_the_opt_out(self, tmp_path, true_like):
        captured = {}

        def fake_command(text, output_path, provider_name, config, tts_config):
            captured["text"] = text
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-wav")
            return output_path

        cfg = _command_tts_config(skip_markdown_strip=true_like)
        with patch("tools.tts_tool._generate_command_tts", side_effect=fake_command), \
             patch("tools.tts_tool._load_tts_config", return_value=cfg):
            text_to_speech_tool(
                text="This is **bold** text",
                output_path=str(tmp_path / "out.wav"),
            )

        # Raw markdown passes through untouched.
        assert "**bold**" in captured["text"]


class TestRawMarkupIsNeverSplit:
    """Review blocker: raw markup (SSML) must not be fed through the plain-text
    splitter — each dispatched payload must stay valid markup."""

    SSML = (
        '<speak><prosody rate="slow">'
        "This sentence is deliberately long enough to cross a tiny "
        "configured provider limit several times over, so a naive "
        "whitespace splitter would cut it into non-XML fragments."
        "</prosody></speak>"
    )

    def test_over_limit_raw_markup_is_refused_not_fragmented(self, tmp_path):
        calls = []

        def fake_command(text, output_path, provider_name, config, tts_config):
            calls.append(text)
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-wav")
            return output_path

        cfg = _command_tts_config(skip_markdown_strip=True, max_text_length=48)
        with patch("tools.tts_tool._generate_command_tts", side_effect=fake_command), \
             patch("tools.tts_tool._load_tts_config", return_value=cfg):
            result = json.loads(text_to_speech_tool(
                text=self.SSML,
                output_path=str(tmp_path / "out.wav"),
            ))

        # Nothing malformed was dispatched, and the caller gets a clear error.
        assert calls == []
        assert result.get("success") is False
        assert "max_text_length" in (result.get("error") or "")

    def test_under_limit_raw_markup_is_sent_as_one_valid_chunk(self, tmp_path):
        calls = []

        def fake_command(text, output_path, provider_name, config, tts_config):
            calls.append(text)
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-wav")
            return output_path

        cfg = _command_tts_config(skip_markdown_strip=True, max_text_length=5000)
        with patch("tools.tts_tool._generate_command_tts", side_effect=fake_command), \
             patch("tools.tts_tool._load_tts_config", return_value=cfg):
            text_to_speech_tool(
                text=self.SSML,
                output_path=str(tmp_path / "out.wav"),
            )

        assert calls == [self.SSML]

    def test_normal_text_still_splits_into_ordered_chunks(self, tmp_path):
        """The no-split contract is scoped to the raw-markup opt-out: default
        providers keep the long-form chunking behavior."""
        calls = []

        def fake_command(text, output_path, provider_name, config, tts_config):
            calls.append(text)
            from pathlib import Path
            Path(output_path).write_bytes(b"fake-wav")
            return output_path

        cfg = _command_tts_config(max_text_length=48)
        long_plain = "One sentence here. " * 10
        with patch("tools.tts_tool._generate_command_tts", side_effect=fake_command), \
             patch("tools.tts_tool._load_tts_config", return_value=cfg):
            text_to_speech_tool(
                text=long_plain,
                output_path=str(tmp_path / "out.wav"),
            )

        assert len(calls) > 1
        assert all(len(c) <= 48 for c in calls)
