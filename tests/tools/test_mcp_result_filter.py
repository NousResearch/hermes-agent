"""Tests for tools/mcp_result_filter.py -- per-server opt-in filtering of MCP results before rendering."""

import json
from types import SimpleNamespace

from mcp.types import CallToolResult

from tools.mcp_result_filter import apply_result_filter
from tools.mcp_tool_content import _MCP_HARD_RESULT_CAP_CHARS
from tools.mcp_tool_handlers import _render_call_tool_result

_PREFIX = "gbrain-page:v1:"
_PAGE_ID = _PREFIX + "WyJkZWZhdWx0Iiwid2lraS9wZXJzb25hbC9yZWZsZWN0aW9ucyJd"
_CONFIG = {"url": "http://127.0.0.1:3131/mcp", "result_filter": {"drop_string_prefixes": [_PREFIX]}}


def _text(text):
    return {"type": "text", "text": text}


def _result(*blocks, structured=None):
    wire = {"content": list(blocks)}
    if structured is not None:
        wire["structuredContent"] = structured
    return CallToolResult.model_validate(wire)


def _render(result, config=_CONFIG):
    return json.loads(_render_call_tool_result(apply_result_filter(result, config), "gbrain"))


def test_configured_prefix_values_are_dropped_from_text_and_structured_content():
    """Opaque values the server config names never reach the model (OpenRouter rejects long
    base64 runs as prompt injection); the rest of each row, including the slug, survives."""
    rows = [{"slug": "wiki/personal/reflections", "source_id": "default", "id": _PAGE_ID,
             "refs": [_PAGE_ID, "keep-me"]}]
    original = _result(_text("Found 1 page."), _text(json.dumps(rows, indent=2)),
                       structured={"results": rows})
    filtered = apply_result_filter(original, _CONFIG)
    expected = [{"slug": "wiki/personal/reflections", "source_id": "default", "refs": ["keep-me"]}]
    assert json.loads(filtered.content[1].text) == expected
    assert filtered.content[1].text.count("\n") > 1  # pretty-printed stays pretty-printed
    assert filtered.content[0] is original.content[0]  # the status line is untouched
    assert _PAGE_ID in original.content[1].text  # the server's own result object is never mutated
    rendered = _render(original)
    assert rendered["result"].startswith("Found 1 page.\n")
    assert rendered["structuredContent"] == {"results": expected}
    assert _PREFIX not in json.dumps(rendered)


def test_filter_runs_before_the_hard_cap_truncates_the_text():
    """A result over the cap is truncated with a notice — no longer valid JSON — so the filter must
    have run first, or every id in the kept head and tail would reach the model."""
    rows = [{"slug": f"wiki/page-{i}", "id": _PAGE_ID} for i in range(30_000)]
    text = json.dumps(rows, indent=2)
    assert len(text) > _MCP_HARD_RESULT_CAP_CHARS
    rendered = _render(_result(_text(text)))
    assert _PREFIX not in rendered["result"]
    assert json.loads(rendered["result"]) == [{"slug": row["slug"]} for row in rows]


def test_bare_string_prefix_is_one_prefix_not_one_per_character():
    config = {"result_filter": {"drop_string_prefixes": _PREFIX}}
    rows = [{"slug": "notes", "title": "release", "id": _PAGE_ID}]
    filtered = apply_result_filter(_result(_text(json.dumps(rows))), config)
    assert json.loads(filtered.content[0].text) == [{"slug": "notes", "title": "release"}]


def test_without_a_match_or_config_the_result_object_is_returned_as_is():
    result = _result(_text(json.dumps([{"slug": "a", "id": _PAGE_ID}])), _text("not json"))
    for config in ({"url": "http://127.0.0.1:3131/mcp"}, None, {"result_filter": {}},
                   {"result_filter": {"drop_string_prefixes": ["other-prefix:"]}}):
        assert apply_result_filter(result, config) is result


def test_non_json_text_and_error_results_pass_through():
    prose = _result(_text(f"see {_PAGE_ID}"))
    assert apply_result_filter(prose, _CONFIG) is prose
    error = SimpleNamespace(content=[SimpleNamespace(text=json.dumps({"id": _PAGE_ID}))], isError=True)
    assert apply_result_filter(error, _CONFIG) is error


def test_plain_objects_are_copied_not_mutated():
    """Non-pydantic results (test doubles, older SDK shapes) get the same copy-on-write treatment."""
    block = SimpleNamespace(type="text", text=json.dumps({"slug": "a", "id": _PAGE_ID}))
    result = SimpleNamespace(content=[block], isError=False, structuredContent={"id": _PAGE_ID, "slug": "a"})
    filtered = apply_result_filter(result, _CONFIG)
    assert json.loads(filtered.content[0].text) == {"slug": "a"}
    assert filtered.structuredContent == {"slug": "a"}
    assert _PAGE_ID in block.text and result.structuredContent["id"] == _PAGE_ID
