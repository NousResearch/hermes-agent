"""Tests for tools/mcp_result_filter.py -- per-server opt-in filtering of rendered MCP results."""

import json

from tools.mcp_result_filter import apply_result_filter

_PAGE_ID = "gbrain-page:v1:WyJkZWZhdWx0Iiwid2lraS9wZXJzb25hbC9yZWZsZWN0aW9ucyJd"
_CONFIG = {"url": "http://127.0.0.1:3131/mcp",
           "result_filter": {"drop_string_prefixes": ["gbrain-page:v1:"]}}


def _rendered(rows):
    """Handler JSON as _render_call_tool_result emits it: a JSON text block plus structuredContent."""
    return json.dumps({"result": json.dumps(rows, indent=2), "structuredContent": {"results": rows}})


def test_configured_prefix_values_are_dropped_from_text_and_structured_content():
    """Opaque values the server config names never reach the model (OpenRouter rejects long
    base64 runs as prompt injection); the rest of each row, including the slug, survives."""
    rows = [{"slug": "wiki/personal/reflections", "source_id": "default", "id": _PAGE_ID,
             "refs": [_PAGE_ID, "keep-me"]}]
    out = json.loads(apply_result_filter(_rendered(rows), _CONFIG))
    expected = [{"slug": "wiki/personal/reflections", "source_id": "default", "refs": ["keep-me"]}]
    assert json.loads(out["result"]) == expected
    assert out["structuredContent"] == {"results": expected}


def test_without_result_filter_config_output_is_untouched():
    rendered = _rendered([{"slug": "a", "id": _PAGE_ID}])
    assert apply_result_filter(rendered, {"url": "http://127.0.0.1:3131/mcp"}) == rendered
    assert apply_result_filter(rendered, None) == rendered
