"""Error redaction must precede the upstream bounded head/tail rendering."""
from functools import partial
from types import SimpleNamespace

import pytest

from tools import mcp_tool_handlers as handlers
from tools.mcp_tool_content import _truncate_mcp_text_result


@pytest.mark.parametrize("boundary", ["head", "tail"])
def test_is_error_redacts_secret_straddling_truncation_boundary(boundary, monkeypatch):
    secret = "opaque-boundary-secret-7391"
    cap = 200
    monkeypatch.setattr(handlers, "_truncate_mcp_text_result", partial(_truncate_mcp_text_result, max_chars=cap))
    length = cap + 1000
    split = int(cap * 0.4) if boundary == "head" else length - (cap - int(cap * 0.4))
    prefix = split - len(secret) // 2
    text = "a" * prefix + secret + "z" * (length - prefix - len(secret))
    result = SimpleNamespace(isError=True, content=[SimpleNamespace(type="text", text=text)])
    rendered = handlers._render_call_tool_result(result, "private", (secret,))
    assert secret[:len(secret) // 2] not in rendered
    assert secret[len(secret) // 2:] not in rendered
    assert "MCP RESULT TRUNCATED" in rendered
