import json

from tools import mcp_tool


def test_mcp_tool_result_under_limit_is_unchanged(monkeypatch):
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
    payload = json.dumps({"result": "short"})

    assert mcp_tool._truncate_mcp_tool_result(payload) == payload


def test_mcp_tool_result_over_limit_is_bounded_valid_json(monkeypatch):
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
    payload = json.dumps({"result": "x" * 500})

    truncated = mcp_tool._truncate_mcp_tool_result(payload)
    parsed = json.loads(truncated)

    assert set(parsed) == {"result"}
    assert "MCP TOOL RESULT TRUNCATED" in parsed["result"]
    assert len(parsed["result"]) < len(payload)
    assert parsed["result"].startswith(payload[:40])
    assert parsed["result"].endswith(payload[-60:])
