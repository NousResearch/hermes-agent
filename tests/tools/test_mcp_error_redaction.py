"""MCP registration errors redact URL credentials without changing URL paths."""

import pytest

from tools.mcp_tool_common import _sanitize_error


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://user:p%40ss@example.test/mcp", "https://[REDACTED]@example.test/mcp"),
        ("http://user:pass@[::1]:9000/mcp", "http://[REDACTED]@[::1]:9000/mcp"),
        ("https://token@example.test/mcp", "https://[REDACTED]@example.test/mcp"),
        ("https://example.test/a@b", "https://example.test/a@b"),
    ],
)
def test_url_userinfo_redacted_without_changing_path(text, expected):
    assert _sanitize_error(text) == expected
