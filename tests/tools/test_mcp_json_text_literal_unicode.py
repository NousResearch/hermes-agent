"""MCP text blocks holding ASCII-escaped JSON reach the model with literal non-ASCII text.

Slack's MCP server (and any server serializing with ``ensure_ascii=True``) sends Thai/CJK/emoji as
six-character ``\\uXXXX`` escapes, several times the tokens of the literal text. The rewrite must be
lossless: same JSON value, escapes that are not real non-ASCII characters left alone.
"""

import json
from types import SimpleNamespace

from tools.mcp_tool_content import _literal_json_unicode


def _block(text):
    return SimpleNamespace(type="text", text=text)


class TestLiteralJsonUnicode:
    def test_thai_and_emoji_decoded(self):
        value = {"messages": "THREAD: เรื่อง SCALE \U0001f600 ok", "n": [1, 2.5, None, True]}
        escaped = json.dumps(value)
        assert "\\u0e40" in escaped and "\\ud83d\\ude00" in escaped
        out = _literal_json_unicode(escaped)
        assert "เรื่อง" in out and "\U0001f600" in out and "\\u" not in out
        assert json.loads(out) == value
        assert len(out) < len(escaped)

    def test_controls_ascii_and_lone_surrogates_stay_escaped(self):
        text = '{"a": "\\u0000\\u001f\\u0041\\u0085\\ud800 x", "b": "\\u00e9"}'
        out = _literal_json_unicode(text)
        assert "\\u0000" in out and "\\u001f" in out and "\\u0041" in out
        assert "\\u0085" in out and "\\ud800" in out
        assert "é" in out
        assert json.loads(out) == json.loads(text)

    def test_escaped_backslash_before_u_is_not_an_escape(self):
        value = {"path": "C:\\u0e40dir", "real": "ไทย"}
        text = json.dumps(value)  # path becomes "C:\\\\u0e40dir": a backslash then the letters u0e40
        out = _literal_json_unicode(text)
        assert json.loads(out) == value
        assert "ไทย" in out and "\\\\u0e40dir" in out

    def test_non_json_and_plain_text_untouched(self):
        for text in ("plain \\u0e40 text", "{not json \\u0e40", "", "no escapes here", "[1, 2]"):
            assert _literal_json_unicode(text) == text

    def test_json_string_literal(self):
        text = json.dumps("สวัสดี")
        assert json.loads(_literal_json_unicode(text)) == "สวัสดี"
        assert _literal_json_unicode(text) == '"สวัสดี"'


class TestRenderedContentBlocks:
    def test_text_block_rendered_literal(self):
        from tools.mcp_tool_handlers import _render_content_blocks

        result = SimpleNamespace(content=[_block(json.dumps({"messages": "ข้อความ"}))])
        text, usable, images = _render_content_blocks(result, "slack")
        assert json.loads(text) == {"messages": "ข้อความ"} and "\\u" not in text
        assert usable == 1 and images == []

    def test_escaped_tag_characters_are_still_stripped(self):
        """Decoding runs before tag stripping: an escaped invisible TAG must not slip through."""
        from tools.mcp_tool_handlers import _render_content_blocks

        hidden = "ev\U000e0069\U000e0067il"
        result = SimpleNamespace(content=[_block(json.dumps({"t": hidden}))])
        text, _, _ = _render_content_blocks(result, "srv")
        assert not any("\U000e0000" <= c <= "\U000e007f" for c in text)
        assert "\\udb40" not in text.lower()
