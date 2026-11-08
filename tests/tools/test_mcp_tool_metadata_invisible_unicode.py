"""MCP tool metadata (description + every string in inputSchema) reaches the model on every API
call, so invisible TAG chars and bidi controls must not survive registration, while a clean
schema stays byte-identical (schemas are frozen wire bytes for prompt caching)."""

import json
from types import SimpleNamespace

from tools.ansi_strip import strip_tool_metadata_text, strip_unicode_tags_deep
from tools.mcp_tool_schema import _convert_mcp_schema

HIDDEN = "\U000e0069\U000e0067\u202e\u2066"  # TAG 'i','g' + RLO + LRI: invisible to the user
FAMILY = "\U0001f468\u200d\U0001f469\u200d\U0001f467"  # ZWJ emoji sequence: must survive


def _invisible(text: str) -> bool:
    return any("\U000e0000" <= c <= "\U000e007f" or c in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
               for c in text)


def test_tool_schema_strips_hidden_chars_everywhere_but_keeps_zwj():
    tool = SimpleNamespace(name="lookup", description=f"Look up{HIDDEN} a record {FAMILY}", inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "title": f"Q{HIDDEN}", "description": f"Search text{HIDDEN}",
                      "default": f"d{HIDDEN}"},
            "mode": {"anyOf": [{"type": "string", "enum": [f"fast{HIDDEN}", "slow"]}, {"type": "null"}]},
        },
        "required": ["query"],
    })
    wire = json.dumps(_convert_mcp_schema("srv", tool), ensure_ascii=False)
    assert not _invisible(wire)
    assert "Search text" in wire and '"fast"' in wire and FAMILY in wire


def test_clean_schema_is_returned_unchanged():
    schema = {"type": "object", "properties": {"q": {"type": "string", "description": f"emoji {FAMILY}",
                                                     "enum": ["a", "b"]}}, "required": ["q"]}
    assert strip_unicode_tags_deep(schema, strip_tool_metadata_text) is schema
